# app.py — Streamlit app for diabetes_meta_pipeline.pkl
# -----------------------------------------------------
# This app expects a full scikit-learn Pipeline saved as
# `diabetes_meta_pipeline.pkl` (produced by pipeline_train.py).
# It supports:
#  - Loading the pipeline
#  - Batch predictions from CSV
#  - Optional single-row manual input (built from uploaded CSV columns)
#  - Download of prediction results
# -----------------------------------------------------

import os
import io
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import streamlit as st
from pathlib import Path
from PIL import Image
import streamlit as st

APP_DIR = Path(__file__).resolve().parent

# Logo yolu
logo_candidates = [
    APP_DIR / "SugarFlag_logo.png",
    APP_DIR.parent / "SugarFlag_logo.png",
    APP_DIR / "Data Sets" / "SugarFlag_logo.png",
]
logo_path = next((p for p in logo_candidates if p.exists()), None)

# Yan yana düzen
col1, col2 = st.columns([1, 4])

with col1:
    if logo_path:
        st.image(Image.open(logo_path), width=120)
    else:
        st.warning("Logo bulunamadı.")

with col2:
    st.markdown(
        """
        <h1 style='margin-bottom: 0;'>SugarFlag: Diyabet Tahmin Modeli</h1>
        <h4 style='margin-top: 0; font-weight: normal;'>The Gradient Gang</h4>
        """,
        unsafe_allow_html=True
    )

# Alt çizgi (divider)
st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)

# Optional SHAP
try:
    import shap  # noqa: F401
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

try:
    import joblib
except Exception:
    joblib = None

st.set_page_config(page_title="Diabetes Prediction (Meta Pipeline)", page_icon="🩺", layout="wide")
st.title("🩺 Diabetes Prediction • Meta Pipeline")
st.caption("Loads `diabetes_meta_pipeline.pkl` and runs predictions on your data.")

# -----------------------------------------------------
# 🔧 Custom classes used inside the pickled pipeline
#    They MUST be available at import-time for joblib.load
# -----------------------------------------------------
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score

class FeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        def bmi_category(bmi):
            if bmi < 18.5:
                return 'Underweight'
            elif bmi < 25:
                return 'Normal'
            elif bmi < 30:
                return 'Overweight'
            else:
                return 'Obese'
        if 'BMI' in df.columns:
            df['BMI_cat'] = df['BMI'].apply(bmi_category)
            df['BMI_cat_code'] = pd.Categorical(
                df['BMI_cat'],
                categories=['Underweight','Normal','Overweight','Obese'],
                ordered=True
            ).codes
            df.drop(columns=['BMI_cat'], inplace=True)
        if {'BMI','Age'}.issubset(df.columns):
            df['BMI_Age_interaction'] = df['BMI'] * df['Age']
        if 'Age' in df.columns and 'BMI_cat_code' in df.columns:
            df['HighRisk_Obese_Old'] = ((df['BMI_cat_code'] == 3) & (df['Age'] >= 10)).astype(int)
        if {'HighBP','HighChol'}.issubset(df.columns):
            df['HighBP_HighChol'] = df['HighBP'] * df['HighChol']
        if {'HighBP','DiffWalk'}.issubset(df.columns):
            df['HighBP_DiffWalk'] = df['HighBP'] * df['DiffWalk']
        if {'BMI','DiffWalk'}.issubset(df.columns):
            df['BMI_DiffWalk'] = df['BMI'] * df['DiffWalk']
        if {'GenHlth','HighRisk_Obese_Old'}.issubset(df.columns):
            df['GenHlth_HighRisk'] = df['GenHlth'] * df['HighRisk_Obese_Old']
        if {'BMI_cat_code','Age'}.issubset(df.columns):
            df['BMIcat_Age'] = df['BMI_cat_code'] * df['Age']
        cardio_cols = [c for c in ['HighBP','HighChol','HeartDiseaseorAttack'] if c in df.columns]
        if cardio_cols:
            df['CardioRiskScore'] = df[cardio_cols].sum(axis=1)
        mobility_cols = [c for c in ['DiffWalk','Stroke'] if c in df.columns]
        if 'PhysHlth' in df.columns:
            mobility_cols.append((df['PhysHlth'] > 30).astype(int))
        if mobility_cols:
            df['MobilityRiskScore'] = np.sum(np.column_stack(
                [(df[c] if not isinstance(c, pd.Series) else c) for c in mobility_cols]
            ), axis=1)
        lifestyle_cols = []
        if 'HvyAlcoholConsump' in df.columns:
            lifestyle_cols.append(df['HvyAlcoholConsump'])
        if 'Smoker' in df.columns:
            lifestyle_cols.append(df['Smoker'])
        if 'NoDocbcCost' in df.columns:
            lifestyle_cols.append(df['NoDocbcCost'])
        if lifestyle_cols:
            df['LifestyleRiskScore'] = np.sum(np.column_stack(lifestyle_cols), axis=1)
        if 'BMI_cat_code' in df.columns and 'Age' in df.columns:
            df['ObeseAndOld'] = ((df['BMI_cat_code'] == 3) & (df['Age'] > 10)).astype(int)
        if {'CardioRiskScore','MobilityRiskScore'}.issubset(df.columns):
            df['HighRiskCluster'] = ((df['CardioRiskScore'] + df['MobilityRiskScore']) > 2).astype(int)
        healthy_conditions = []
        if 'PhysActivity' in df.columns:
            healthy_conditions.append(df['PhysActivity'] == 1)
        if 'Fruits' in df.columns:
            healthy_conditions.append(df['Fruits'] > 0)
        if 'Veggies' in df.columns:
            healthy_conditions.append(df['Veggies'] > 0)
        if 'HvyAlcoholConsump' in df.columns:
            healthy_conditions.append(df['HvyAlcoholConsump'] == 0)
        if 'Smoker' in df.columns:
            healthy_conditions.append(df['Smoker'] == 0)
        if healthy_conditions:
            df['HealthyLifestyle'] = np.all(np.column_stack(healthy_conditions), axis=1).astype(int)
        if 'Age' in df.columns:
            df['AgeGroup'] = pd.cut(df['Age'], bins=[0,4,8,12,14], labels=['18-34','35-49','50-64','65+'])
        if 'Income' in df.columns:
            df['IncomeGroup'] = pd.cut(df['Income'], bins=[0,4,7,8], labels=['Low','Mid','High'])
        for col in ['risk_score','rule_pred','Diabetes_012']:
            if col in df.columns:
                df = df.drop(columns=[col])
        return df

class MetaProbaFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, rf: RandomForestClassifier | None = None, gnb: GaussianNB | None = None):
        self.rf = rf if rf is not None else RandomForestClassifier(random_state=17)
        self.gnb = gnb if gnb is not None else GaussianNB()
    def fit(self, X, y=None):
        self.rf_ = clone(self.rf)
        self.gnb_ = clone(self.gnb)
        self.rf_.fit(X, y)
        self.gnb_.fit(X, y)
        y_pred_rf = self.rf_.predict(X)
        y_pred_gnb = self.gnb_.predict(X)
        self.recall_pos_rf_ = recall_score(y, y_pred_rf, pos_label=1)
        self.precision_neg_gnb_ = precision_score(y, y_pred_gnb, pos_label=0)
        return self
    def transform(self, X):
        rf_proba = self.rf_.predict_proba(X)
        gnb_proba = self.gnb_.predict_proba(X)
        rf_scaled = rf_proba * np.array([1.0, self.recall_pos_rf_])
        gnb_scaled = gnb_proba * np.array([self.precision_neg_gnb_, 1.0])
        return np.hstack([X, rf_scaled, gnb_scaled])

# ---------------------
# Sidebar: Model Loader
# ---------------------
st.sidebar.header("⚙️ Settings")
model_path = st.sidebar.text_input("Model path (.pkl)", value="diabetes_meta_pipeline.pkl")

@st.cache_resource(show_spinner=True)
def load_model(path: str):
    if joblib is None:
        raise RuntimeError("joblib is not installed. Run: pip install joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

pipe = None
err = None

# ---- Helpers for feature names & meta input ----

def get_pre_feature_names(pipe):
    try:
        pre = pipe.named_steps.get("pre")
        if hasattr(pre, "get_feature_names_out"):
            return list(pre.get_feature_names_out())
    except Exception:
        pass
    return None


def get_meta_feature_names(pipe, names_pre):
    try:
        meta = pipe.named_steps.get("meta")
        mf = meta.named_steps.get("meta_features")
        rf_classes = list(getattr(mf.rf_, "classes_", [0, 1]))
        gnb_classes = list(getattr(mf.gnb_, "classes_", [0, 1]))
        rf_names = [f"rf_proba_{c}_scaled" for c in rf_classes]
        gnb_names = [f"gnb_proba_{c}_scaled" for c in gnb_classes]
        return (names_pre or [] ) + rf_names + gnb_names
    except Exception:
        return None


def transform_to_meta_input(pipe, df: pd.DataFrame):
    """Return Z (input to final RF), names for Z if possible, and a small note."""
    note = None
    try:
        features = pipe.named_steps.get("features")
        pre = pipe.named_steps.get("pre")
        meta = pipe.named_steps.get("meta")
        mf = meta.named_steps.get("meta_features")
        std = meta.named_steps.get("std")

        Xf = features.transform(df)
        Xpre = pre.transform(Xf)
        Xmeta = mf.transform(Xpre)
        Z = std.transform(Xmeta)

        names_pre = get_pre_feature_names(pipe)
        names_all = get_meta_feature_names(pipe, names_pre)
        return Z, names_all, note
    except Exception as e:
        note = str(e)
        return None, None, note
try:
    pipe = load_model(model_path)
except Exception as e:
    err = e

with st.expander("🔍 Model Info", expanded=True):
    if err:
        st.error(f"Model could not be loaded: {err}")
    else:
        try:
            from sklearn.pipeline import Pipeline
            is_pipe = isinstance(pipe, Pipeline)
        except Exception:
            is_pipe = False
        if is_pipe:
            steps = [(name, step.__class__.__name__) for name, step in pipe.steps]
            st.json({"is_pipeline": True, "steps": steps})
            try:
                final_est = pipe.named_steps.get("meta", None)
                final_rf = None
                if final_est is not None and hasattr(final_est, "named_steps"):
                    final_rf = final_est.named_steps.get("clf", None)
                if final_rf is not None and hasattr(final_rf, "get_params"):
                    st.write("**Final estimator:** RandomForestClassifier (meta)")
            except Exception:
                pass
        else:
            st.json({"is_pipeline": False, "type": type(pipe).__name__})

# ---------------------
# Tabs
# ---------------------
TAB_BATCH, TAB_SINGLE, TAB_EXPLAIN, TAB_SELFTEST = st.tabs(["📦 Batch Predict (CSV)", "🧍 Single Input (Manual)", "🧠 Explain (SHAP & Importance)", "📝 Risk Test (13 Q)"])

with TAB_BATCH:
    st.subheader("📤 Upload CSV")
    csv_file = st.file_uploader("Select a CSV file", type=["csv"], accept_multiple_files=False)

    sample_info = st.empty()
    if csv_file is not None and pipe is not None and err is None:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            st.error(f"CSV could not be read: {e}")
            st.stop()

        # Remove target columns if present (the pipeline expects only features)
        for col in ["Diabetes_binary", "Diabetes_012", "rule_pred", "risk_score"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        st.session_state._last_df_cols = list(df.columns)

        st.write("**Preview (top 8 rows):**")
        st.dataframe(df.head(8), use_container_width=True)
        st.caption(f"Rows: {len(df):,} • Columns: {len(df.columns)}")

        run = st.button("▶️ Run Predictions", type="primary")
        if run:
            with st.spinner("Predicting..."):
                try:
                    y_pred = pipe.predict(df)
                    out = pd.DataFrame({"prediction": y_pred})
                    proba_cols = None
                    try:
                        proba = pipe.predict_proba(df)
                        # Binary or multiclass agnostic column names
                        if proba.ndim == 2:
                            proba_cols = [f"proba_{i}" for i in range(proba.shape[1])]
                            proba_df = pd.DataFrame(proba, columns=proba_cols)
                            out = pd.concat([out, proba_df], axis=1)
                    except Exception:
                        pass

                    result = pd.concat([df.reset_index(drop=True), out], axis=1)
                    st.success("Predictions ready.")
                    st.dataframe(result.head(1000), use_container_width=True)

                    # Download button
                    csv_bytes = result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv_bytes,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )

                    # Quick summary
                    st.markdown("### 📊 Quick Summary")
                    try:
                        unique, counts = np.unique(y_pred, return_counts=True)
                        summary_df = pd.DataFrame({"class": unique, "count": counts})
                        st.dataframe(summary_df, use_container_width=True)
                    except Exception:
                        pass
                except Exception as e:
                    st.exception(e)

with TAB_SINGLE:
    st.subheader("✍️ Manual Input")
    st.caption("The form fields are inferred from an uploaded CSV (Batch tab). Alternatively, paste a JSON row.")

    if "_last_df_cols" not in st.session_state:
        st.session_state._last_df_cols = []

    uploaded_cols = []
    if 'csv_file' in locals() and csv_file is not None:
        try:
            tmp = pd.read_csv(csv_file, nrows=5)
            for c in ["Diabetes_binary", "Diabetes_012", "rule_pred", "risk_score"]:
                if c in tmp.columns:
                    tmp = tmp.drop(columns=[c])
            uploaded_cols = list(tmp.columns)
            st.session_state._last_df_cols = uploaded_cols
        except Exception:
            pass

    cols_for_form = st.session_state._last_df_cols

    # -------------------------------
    # A) FORM ÜZERİNDEN TEK SATIR GİRİŞ
    # -------------------------------
    if cols_for_form:
        st.write("**Form built from uploaded CSV columns**")
        chunk = 3 if len(cols_for_form) >= 9 else 2
        row_dict = {}
        for i in range(0, len(cols_for_form), chunk):
            cset = st.columns(chunk)
            for j, col in enumerate(cols_for_form[i:i+chunk]):
                with cset[j]:
                    is_num = True
                    try:
                        is_num = pd.api.types.is_numeric_dtype(tmp[col]) if 'tmp' in locals() and col in tmp.columns else True
                    except Exception:
                        pass
                    if is_num:
                        val = st.number_input(col, value=0.0)
                    else:
                        val = st.text_input(col, value="")
                    row_dict[col] = val

        # Kullanıcının girdiği form verisi
        X_one = pd.DataFrame([row_dict])

        # 👇 Eksik kolonları tamamla ve sıralamayı şablona göre yap
        if st.session_state._last_df_cols:
            for col in st.session_state._last_df_cols:
                if col not in X_one.columns:
                    X_one[col] = 0
            X_one = X_one[st.session_state._last_df_cols]

        st.dataframe(X_one, use_container_width=True)

        if st.button("🧪 Predict Single Row", type="primary"):
            try:
                y1 = pipe.predict(X_one)[0]
                st.success(f"Prediction: {int(y1)}")
                try:
                    p1 = pipe.predict_proba(X_one)[0]
                    st.write({f"proba_{i}": float(p) for i, p in enumerate(p1)})
                except Exception:
                    pass
            except Exception as e:
                st.exception(e)

    # -------------------------------
    # B) JSON ÜZERİNDEN TEK SATIR GİRİŞ
    # -------------------------------
    else:
        st.write("**Paste a JSON row** (keys=feature names)")
        json_text = st.text_area(
            "JSON row",
            value='{"HighBP":0, "HighChol":0, "BMI":27.5, "Smoker":0, "Age":9}'
        )

        if st.button("🧪 Predict from JSON", type="primary"):
            try:
                data = json.loads(json_text)
                X_one = pd.DataFrame([data])

                # 👇 Eksik kolonları CSV’den gelen kolon listesiyle tamamla
                template_cols = st.session_state.get("_last_df_cols", [])
                if not template_cols:
                    st.warning("Önce Batch sekmesinde bir CSV yükleyin ki kolon şablonu alınsın.")
                    st.stop()

                # Tüm kolonları 0 ile hazırla, JSON’daki değerlerle güncelle
                row_full = {c: 0 for c in template_cols}
                for k, v in data.items():
                    if k in row_full:
                        row_full[k] = v
                X_full = pd.DataFrame([row_full])[template_cols]

                y1 = pipe.predict(X_full)[0]
                st.success(f"Prediction: {int(y1)}")
                try:
                    p1 = pipe.predict_proba(X_full)[0]
                    st.write({f"proba_{i}": float(p) for i, p in enumerate(p1)})
                except Exception:
                    pass
            except Exception as e:
                st.exception(e)
with TAB_EXPLAIN:
    st.subheader("🧠 Explanations")
    if err:
        st.error("Load a valid model first in the sidebar.")
    else:
        if not _HAS_SHAP:
            st.info("`shap` is not installed. Run `pip install shap` to enable this tab.")
        else:
            st.caption("Upload a small CSV (≤ 2,000 rows recommended) to compute feature importance and SHAP plots.")
            csv_small = st.file_uploader("CSV for explanations", type=["csv"], key="csv_explain")
            if csv_small is not None:
                try:
                    df_s = pd.read_csv(csv_small)
                except Exception as e:
                    st.error(f"CSV could not be read: {e}")
                    st.stop()
                for col in ["Diabetes_binary", "Diabetes_012", "rule_pred", "risk_score"]:
                    if col in df_s.columns:
                        df_s = df_s.drop(columns=[col])

                # Limit rows for faster SHAP
                if len(df_s) > 2000:
                    st.warning("Taking first 2000 rows for speed.")
                    df_s = df_s.head(2000)

                # Build meta-input Z (the matrix used by final RF)
                Z, names_all, note = transform_to_meta_input(pipe, df_s)
                if Z is None:
                    st.error("Failed to transform data for explanations.")
                    if note:
                        st.code(note)
                    st.stop()

                meta = pipe.named_steps.get("meta")
                clf = meta.named_steps.get("clf")

                # --- Feature Importance ---
                st.markdown("### 📈 Feature Importance (Final RF)")
                try:
                    if hasattr(clf, "feature_importances_"):
                        import matplotlib.pyplot as plt
                        importances = clf.feature_importances_
                        if names_all is None or len(names_all) != len(importances):
                            names_all = [f"f{i}" for i in range(len(importances))]
                        imp_df = pd.DataFrame({"feature": names_all, "importance": importances}).sort_values("importance", ascending=False).head(40)
                        st.dataframe(imp_df, use_container_width=True)
                        fig, ax = plt.subplots(figsize=(10, 10))
                        imp_df.sort_values("importance").plot(kind="barh", x="feature", y="importance", ax=ax)
                        ax.set_title("Top Feature Importances (Final RF)")
                        st.pyplot(fig, clear_figure=True)
                    else:
                        st.info("Final estimator has no feature_importances_.")
                except Exception as e:
                    st.exception(e)

                # --- SHAP Summary ---
                st.markdown("### 🧩 SHAP Summary (Final RF)")
                try:
                    explainer = shap.TreeExplainer(clf)
                    # Use a smaller sample for SHAP to keep it fast
                    sample_n = min(500, Z.shape[0])
                    Z_sample = Z[:sample_n]
                    shap_vals = explainer.shap_values(Z_sample)
                    st.caption(f"Using {sample_n} rows for SHAP.")
                    shap.summary_plot(shap_vals, Z_sample, feature_names=names_all, show=False)
                    st.pyplot(bbox_inches='tight', clear_figure=True)
                except Exception as e:
                    st.info("SHAP summary failed; see error below.")
                    st.exception(e)

                # --- SHAP Waterfall for first row ---
                st.markdown("### 🌊 SHAP Waterfall (Row 0)")
                try:
                    shap_vals_full = explainer.shap_values(Z_sample)


                    # tek örnek ve pozitif sınıf (1) için bir vektör seç
                    def pick_single_sv(shap_vals, expected):
                        if isinstance(shap_vals, list):
                            sv = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
                            base = expected[1] if (isinstance(expected, (list, tuple, np.ndarray)) and len(
                                expected) > 1) else expected
                            return sv, base

                        arr = np.array(shap_vals)
                        if arr.ndim == 3:  # (n_samples, n_features, n_classes)
                            base = expected[1] if (isinstance(expected, (list, tuple, np.ndarray)) and arr.shape[
                                -1] > 1) else expected
                            return arr[0, :, -1], base  # son sınıfı (genelde pozitif) al
                        elif arr.ndim == 2:  # (n_samples, n_features)
                            return arr[0, :], expected
                        else:
                            return arr.squeeze(), expected


                    sv, base = pick_single_sv(shap_vals_full, explainer.expected_value)

                    ex = shap.Explanation(
                        values=sv,  # (n_features,)
                        base_values=base,  # skaler
                        data=Z_sample[0],  # (n_features,)
                        feature_names=names_all
                    )
                    shap.plots.waterfall(ex, max_display=20)
                    st.pyplot(bbox_inches='tight', clear_figure=True)
                except Exception as e:
                    st.info("Waterfall plot failed.")
                    st.exception(e)
with TAB_SELFTEST:
    st.subheader("📝 Diyabet Risk Testi (13 Soru)")
    st.caption("13 soruyu cevaplayın. Kalan ham kolonlar güvenli varsayılanlarla doldurulur ve eğitimli pipeline ile risk tahmini yapılır.")

    # 1) Eğitimde görülen ham kolon şablonu (eksikleri tamamlamak için)
    required_cols = None
    source = ""
    try:
        # Eğer pipeline eğitimde bu bilgiyi kaydettiyse
        required_cols = pipe.named_steps["features"].input_columns_
        source = "modelden (FeatureBuilder.input_columns_)"
    except Exception:
        pass

    # Batch sekmesinde yüklenen CSV'den alınan şablon (bir kez CSV yüklemen yeterli)
    if required_cols is None and "_last_df_cols" in st.session_state and st.session_state._last_df_cols:
        required_cols = st.session_state._last_df_cols
        source = "son yüklenen CSV'den"

    if required_cols is None:
        st.warning("Önce Batch sekmesinde bir CSV yükleyin ya da modeli FeatureBuilder.input_columns_ kaydedecek şekilde yeniden export edin.")
        st.stop()

    st.caption(f"Ham kolon şablonu: {source}. Toplam kolon: {len(required_cols)}")

    # BRFSS yaş kodu (1..13) dönüştürücü: gerçek yaştan koda çevirir
    def age_to_code(age_years: int) -> int:
        bins = [(18,24),(25,29),(30,34),(35,39),(40,44),(45,49),(50,54),(55,59),(60,64),(65,69),(70,74),(75,79),(80,120)]
        for i,(a,b) in enumerate(bins, start=1):
            if a <= age_years <= b:
                return i
        return 9  # makul default

    # 2) 13 soruluk form
    with st.form("risk_test_form"):
        q = {}
        q['HighBP'] = st.radio("1️⃣ Yüksek tansiyonunuz var mı?", [1, 0], format_func=lambda x: "Evet" if x==1 else "Hayır", index=0)
        q['HighChol'] = st.radio("2️⃣ Yüksek kolesterolünüz var mı?", [1, 0], format_func=lambda x: "Evet" if x==1 else "Hayır", index=0)
        q['CholCheck'] = st.radio("3️⃣ Son 5 yılda kolesterol kontrolü yaptırdınız mı?", [1, 0], format_func=lambda x: "Evet" if x==1 else "Hayır", index=1)
        q['BMI'] = st.number_input("4️⃣ Vücut kitle indeksiniz (BMI)", min_value=10.0, max_value=70.0, value=27.5, step=0.1)
        q['Smoker'] = st.radio("5️⃣ Sigara kullanıyor musunuz?", [1, 0], format_func=lambda x: "Evet" if x==1 else "Hayır", index=0)
        q['Stroke'] = st.radio("6️⃣ Felç geçirdiniz mi?", [1, 0], format_func=lambda x: "Evet" if x==1 else "Hayır", index=0)
        q['HeartDiseaseorAttack'] = st.radio("7️⃣ Kalp hastalığı / kalp krizi öyküsü var mı?", [1, 0], format_func=lambda x: "Evet" if x==1 else "Hayır", index=0)
        q['PhysActivity'] = st.radio("8️⃣ Düzenli fiziksel aktivite yapıyor musunuz?", [1, 0], format_func=lambda x: "Evet" if x==1 else "Hayır", index=1)
        q['Fruits'] = st.radio("9️⃣ Haftada en az bir kez meyve tüketiyor musunuz?", [1, 0], format_func=lambda x: "Evet" if x==1 else "Hayır", index=1)
        q['Veggies'] = st.radio("🔟 Haftada en az bir kez sebze tüketiyor musunuz?", [1, 0], format_func=lambda x: "Evet" if x==1 else "Hayır", index=1)
        q['HvyAlcoholConsump'] = st.radio("1️⃣1️⃣ Aşırı alkol tüketimi var mı?", [1, 0], format_func=lambda x: "Evet" if x==1 else "Hayır", index=0)
        q['DiffWalk'] = st.radio("1️⃣2️⃣ Yürümede zorluk çekiyor musunuz?", [1, 0], format_func=lambda x: "Evet" if x==1 else "Hayır", index=0)
        age_years = st.slider("1️⃣3️⃣ Yaşınız (yıl)", min_value=18, max_value=90, value=35)
        q['Age'] = age_to_code(int(age_years))  # pipeline eğitimine uygun 1..13 kodu

        # Veri setinde varsa cinsiyet (0: Kadın, 1: Erkek) — opsiyonel ama önemli
        if 'Sex' in required_cols:
            q['Sex'] = st.radio("⚥ Cinsiyetiniz", [0,1], format_func=lambda x: "Kadın" if x==0 else "Erkek", index=1)

        submitted = st.form_submit_button("🔍 Riski Hesapla")

    # 3) Tahmin
    if submitted:
        import pandas as pd

        # Tüm ham kolonları içeren tek satır oluştur ve seçilen cevaplarla doldur
        row = {c: 0 for c in required_cols}     # eksikler güvenli default: 0
        # bazı sayısallar için makul defaultlar
        for num_default in ["BMI", "PhysHlth", "MentHlth"]:
            if num_default in row:
                row[num_default] = 0
        for k, v in q.items():
            if k in row:
                row[k] = v
        X_user = pd.DataFrame([row])[required_cols]

        try:
            y_pred = int(pipe.predict(X_user)[0])
            st.success(f"Tahmin edilen diyabet sınıfı: **{y_pred}**")
            # Tahmine göre bilgilendirme mesajı
            if y_pred == 1:
                st.warning("🔴 Diyabet riskiniz *yüksek*. Lütfen profesyonel destek için doktora başvurunuz.")
            elif y_pred == 0:
                st.info("🟢 Diyabet riskiniz *düşük*. Ancak sağlıklı yaşam alışkanlıklarını sürdürmeniz önerilir.")
            try:
                p = pipe.predict_proba(X_user)[0]
                st.write({f"proba_{i}": float(pi) for i, pi in enumerate(p)})
            except Exception:
                pass
        except Exception as e:
            st.exception(e)
# ---------------
# Footer
# ---------------
with st.expander("ℹ️ Notes"):
    st.markdown(
        """
        - Remove target columns from your CSV (e.g., `Diabetes_binary`, `Diabetes_012`).
        - The pipeline performs feature engineering and encoding internally.
        - If the model fails to load due to environment mismatch, recreate the `.pkl` in the same environment used by this app.
        """
    )
