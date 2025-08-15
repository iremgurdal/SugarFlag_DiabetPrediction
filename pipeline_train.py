"""
Pipeline: BRFSS Binary Diabetes (0/1)
- Raw → Feature engineering → ColumnTransformer (RobustScaler + OneHotEncoder)
- Train two base models (RF, GNB) inside a transformer → make scaled proba features
- Concatenate [X_transformed, RF_proba_scaled, GNB_proba_scaled]
- Standardize → Final meta RandomForest (class_weight={0:1,1:15})

Bu dosya, senin verdiğin eğitim akışına UYUMLU olacak şekilde yeniden düzenlenmiş, tek parça
scikit-learn Pipeline üretir. Eğitimden sonra `joblib.dump(pipeline, 'diabetes_meta_pipeline.pkl')`
ile **tam akış** kaydedilir ve Streamlit app doğrudan bu pipeline'ı yükleyip `predict` çalıştırabilir.
"""
from __future__ import annotations

pip install scikit-learn

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.metrics import recall_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# ================================
# 1) Feature Engineering
# ================================
class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Senin kodundaki feature engineering adımlarını DataFrame → DataFrame olarak uygular.
    - BMI_cat_code, BMI_Age_interaction, HighRisk_Obese_Old
    - Çeşitli etkileşimler ve risk skorları (rule_pred/risk_score HARİÇ)
    - AgeGroup, IncomeGroup kategorik gruplar
    Not: Hedef (Diabetes_012/Diabetes_binary) yoksa zaten dokunulmaz.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # --- BMI bucket & code ---
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
                categories=['Underweight', 'Normal', 'Overweight', 'Obese'],
                ordered=True
            ).codes
            df.drop(columns=['BMI_cat'], inplace=True)

        if {'BMI', 'Age'}.issubset(df.columns):
            df['BMI_Age_interaction'] = df['BMI'] * df['Age']

        if 'Age' in df.columns and 'BMI_cat_code' in df.columns:
            df['HighRisk_Obese_Old'] = ((df['BMI_cat_code'] == 3) & (df['Age'] >= 10)).astype(int)

        # Interactions & scores
        if {'HighBP', 'HighChol'}.issubset(df.columns):
            df['HighBP_HighChol'] = df['HighBP'] * df['HighChol']
        if {'HighBP', 'DiffWalk'}.issubset(df.columns):
            df['HighBP_DiffWalk'] = df['HighBP'] * df['DiffWalk']
        if {'BMI', 'DiffWalk'}.issubset(df.columns):
            df['BMI_DiffWalk'] = df['BMI'] * df['DiffWalk']
        if {'GenHlth', 'HighRisk_Obese_Old'}.issubset(df.columns):
            df['GenHlth_HighRisk'] = df['GenHlth'] * df['HighRisk_Obese_Old']
        if {'BMI_cat_code', 'Age'}.issubset(df.columns):
            df['BMIcat_Age'] = df['BMI_cat_code'] * df['Age']

        cardio_cols = [c for c in ['HighBP', 'HighChol', 'HeartDiseaseorAttack'] if c in df.columns]
        if cardio_cols:
            df['CardioRiskScore'] = df[cardio_cols].sum(axis=1)

        mobility_cols = [c for c in ['DiffWalk', 'Stroke'] if c in df.columns]
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
        if {'CardioRiskScore', 'MobilityRiskScore'}.issubset(df.columns):
            df['HighRiskCluster'] = ((df['CardioRiskScore'] + df['MobilityRiskScore']) > 2).astype(int)

        # HealthyLifestyle flag
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

        # Groups
        if 'Age' in df.columns:
            df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 4, 8, 12, 14], labels=['18-34', '35-49', '50-64', '65+'])
        if 'Income' in df.columns:
            df['IncomeGroup'] = pd.cut(df['Income'], bins=[0, 4, 7, 8], labels=['Low', 'Mid', 'High'])

        # Drop artifacts if present
        for col in ['risk_score', 'rule_pred', 'Diabetes_012']:
            if col in df.columns:
                df = df.drop(columns=[col])

        return df


# ================================
# 2) Meta Proba Feature Transformer
# ================================
class MetaProbaFeatures(BaseEstimator, TransformerMixin):
    """
    fit:
      - X_arr (numpy) ve y ile RF ve GNB'yi fit eder (clone ile).
      - RF için recall_pos, GNB için precision_neg değerlerini eğitim üzerinde hesaplar.
    transform:
      - predict_proba(X) üretir, ilgili ölçeklerle çarpar ve [X, rf_proba_scaled, gnb_proba_scaled] döner.
    """
    def __init__(self,
                 rf: Optional[RandomForestClassifier] = None,
                 gnb: Optional[GaussianNB] = None):
        # DİKKAT: sklearn estimator'larında truthiness __len__ üzerinden gider ve fit öncesi AttributeError atar.
        # Bu yüzden 'rf or RandomForestClassifier(...)' şeklinde yazmayın.
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
        # scale as in your code: [1, recall_pos_rf] and [precision_neg_gnb, 1]
        rf_scaled = rf_proba * np.array([1.0, self.recall_pos_rf_])
        gnb_scaled = gnb_proba * np.array([self.precision_neg_gnb_, 1.0])
        return np.hstack([X, rf_scaled, gnb_scaled])


# ================================
# 3) Full Pipeline Builder
# ================================

def build_pipeline(numeric_cols: List[str] | None = None, categorical_cols: List[str] | None = None) -> Pipeline:
    """
    numeric_cols/categorical_cols verilmezse, FeatureBuilder sonrası oluşan DataFrame'e göre
    otomatik seçim yapılır (dtype bazlı). Bu, 'Diabetes_012' gibi sonradan silinen kolon
    yüzünden ColumnTransformer'ın hata vermesini engeller.
    """
    from sklearn.compose import make_column_selector as selector

    if numeric_cols is None or categorical_cols is None:
        num_sel = selector(dtype_include=np.number)
        cat_sel = selector(dtype_exclude=np.number)
        pre = ColumnTransformer(
            transformers=[
                ("num", RobustScaler(with_centering=False), num_sel),
                ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), cat_sel),
            ],
            remainder='drop'
        )
    else:
        drop_if_present = {"Diabetes_012", "risk_score", "rule_pred"}
        numeric_cols = [c for c in numeric_cols if c not in drop_if_present]
        categorical_cols = [c for c in categorical_cols if c not in drop_if_present]
        pre = ColumnTransformer(
            transformers=[
                ("num", RobustScaler(with_centering=False), numeric_cols),
                ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
            ],
            remainder='passthrough'
        )

    meta_stack = Pipeline([
        ("meta_features", MetaProbaFeatures(
            rf=RandomForestClassifier(random_state=17),
            gnb=GaussianNB()
        )),
        ("std", StandardScaler(with_mean=False)),
        ("clf", RandomForestClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=5,
            class_weight={0:1, 1:15}
        )),
    ])

    pipe = Pipeline([
        ("features", FeatureBuilder()),
        ("pre", pre),
        ("meta", meta_stack),
    ])
    return pipe


# ================================
# 4) Example Training Script
# ================================
if __name__ == "__main__":
    import joblib

    # 1) Load raw data
    df = pd.read_csv("Data Sets/diabetes_012_health_indicators_BRFSS2015.csv")

    # 2) Prepare target
    if 'Diabetes_binary' not in df.columns:
        df['Diabetes_binary'] = df['Diabetes_012'].apply(lambda x: 1 if x > 0 else 0)

    y = df['Diabetes_binary']
    X = df.drop(columns=['Diabetes_binary'])

    # 3) Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=17
    )

    # 4) Identify raw numeric/categorical (before FeatureBuilder)
    raw_numeric = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    raw_categorical = [c for c in X_train.columns if c not in raw_numeric]

    pipe = build_pipeline()

    # 5) Fit & evaluate quickly
    pipe.fit(X_train, y_train)
    print("Train score (acc):", pipe.score(X_train, y_train))
    print("Test  score (acc):", pipe.score(X_test, y_test))

    # 6) Save full pipeline (ready for Streamlit)
    joblib.dump(pipe, "diabetes_meta_pipeline.pkl")
    print("Saved: diabetes_meta_pipeline.pkl")
