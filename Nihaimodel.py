import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#1-Loading the data sets
df = pd.read_csv("Data Sets/diabetes_012_health_indicators_BRFSS2015.csv", sep =",", encoding ='utf-8')

def check_df(dataframe, head=5):
    print('##################### Shape #####################')
    print(dataframe.shape)
    print('##################### Types #####################')
    print(dataframe.dtypes)
    print('##################### Head #####################')
    print(dataframe.head(head))
    print('##################### Tail #####################')
    print(dataframe.tail(head))
    print('##################### NA #####################')
    print(dataframe.isnull().sum())
    print('##################### Quantiles #####################')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


#2-Capturing / Detecting Numeric and Categorical Variables
def grab_col_names(dataframe, cat_th=32, car_th=33):
    """

    Returns the names of categorical, numeric and categorical but cardinal variables in the data set.
    Note Categorical variables include categorical variables with numeric appearance.

    Parameters
    ------
        dataframe: dataframe
                Variable names of the dataframe to be taken
        cat_th: int, optional
                class threshold for numeric but categorical variables
        car_th: int, optinal
                class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                List of cardinal variables with categorical appearance

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The sum of the 3 return lists equals the total number of variables: cat_cols + num_cols + cat_but_car = number of variables

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car, num_but_cat

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

#Analysis

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print('##########################################')
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
for col in num_cols:
    num_summary(df, col)

def target_summary_with_cat(dataframe, target, categorical_col, plot=False):
    print(pd.DataFrame({'TARGET_MEAN': dataframe.groupby(categorical_col)[target].mean()}), end='\n\n\n')
    if plot:
        sns.barplot(x=categorical_col, y=target, data=dataframe)
        plt.show(block=True)
for col in cat_cols:
    target_summary_with_cat(df, 'Diabetes_012', col)

def target_summary_with_num(dataframe, target, numerical_col, plot=False):
    print(pd.DataFrame({numerical_col+'_mean': dataframe.groupby(target)[numerical_col].mean()}), end='\n\n\n')
    if plot:
        sns.barplot(x=target, y=numerical_col, data=dataframe)
        plt.show(block=True)
for col in num_cols:
    target_summary_with_cat(df, 'Diabetes_012', col)

#Correlation Analysis

def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    numeric_df = dataframe.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (12, 12)})
        plt.figure(figsize=(12, 12))  # Ekledik
        sns.heatmap(corr, cmap="RdBu", annot=True)
        plt.title("Correlation Heatmap")
        plt.show(block=True)

    return drop_list

high_correlated_cols(df, plot=True)


def high_correlated_cols(dataframe, head=10):
    numeric_df = dataframe.select_dtypes(include=[np.number])  # sadece sayısal sütunlar
    corr_matrix = numeric_df.corr().abs()

    # Üst üçgen matrisi al (kendisiyle ve tekrar edenleri dışla)
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Korelasyon değerlerini sıralayıp en yüksek olanları seç
    corr_cols = upper_triangle.stack().sort_values(ascending=False).head(head)

    return corr_cols


df["Diabetes_012"].hist(bins=100)
plt.show(block=True)


# Outlier Analysis
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
for col in num_cols:
    if col != "Diabetes_012":
        print(col, ':', check_outlier(df, col))

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
for col in num_cols:
    if col != "Diabetes_012":
        replace_with_thresholds(df, col)
for col in num_cols:
    if col != "Diabetes_012":
        print(col, ':', check_outlier(df, col))


#Missing Value Analysis

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns
missing_values_table(df)

#Rare Analysis
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ':', len(dataframe[col].value_counts()))
        print(pd.DataFrame({'COUNT': dataframe[col].value_counts(),
                            'RATIO': dataframe[col].value_counts() / len(dataframe),
                            'TARGET_MEAN': dataframe.groupby(col)[target].mean()}), end='\n\n\n')
rare_analyser(df, "Diabetes_012", cat_cols)

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
    return temp_df
rare_encoder(df, 0.01)

#Feature Extraction
# BMI Categories
def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

df['BMI_cat'] = df['BMI'].apply(bmi_category)

# BMI x Age (Age is coded 1-13, higher = older age group)
df['BMI_Age_interaction'] = df['BMI'] * df['Age']

# High_risk group: Obese + Age 10+ (65+ per BRFSS age codes)
df['HighRisk_Obese_Old'] = ((df['BMI_cat'] == 'Obese') & (df['Age'] >= 10)).astype(int)

# Convert BMI_cat to ordinal codes
df['BMI_cat_code'] = pd.Categorical(df['BMI_cat'],
                                    categories=['Underweight','Normal','Overweight','Obese'],
                                    ordered=True).codes
df = df.drop(columns=['BMI_cat'])

def add_new_features(df):
    # 1️⃣ ETKİLEŞİM FEATURE'LARI
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

    # 2️⃣ RİSK SKORLARI
    cardio_cols = [col for col in ['HighBP', 'HighChol', 'HeartDiseaseorAttack'] if col in df.columns]
    if cardio_cols:
        df['CardioRiskScore'] = df[cardio_cols].sum(axis=1)

    mobility_cols = [col for col in ['DiffWalk', 'Stroke'] if col in df.columns]
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

    # 3️⃣ BINARY FLAG’LER
    if {'BMI_cat_code', 'Age'}.issubset(df.columns):
        df['ObeseAndOld'] = ((df['BMI_cat_code'] == 3) & (df['Age'] > 10)).astype(int)

    if {'CardioRiskScore', 'MobilityRiskScore'}.issubset(df.columns):
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

    # 4️⃣ YAŞ & GELİR GRUPLAMALARI
    if 'Age' in df.columns:
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 4, 8, 12, 14], labels=['18-34', '35-49', '50-64', '65+'])

    if 'Income' in df.columns:
        df['IncomeGroup'] = pd.cut(df['Income'], bins=[0, 4, 7, 8], labels=['Low', 'Mid', 'High'])

    return df


# Kullanım
df = add_new_features(df)

# KURAL TABANLI RİSK SKORU
# ============================== #
def calculate_custom_risk_score(row):
    score = 0
    score += int(row['Age'] >= 5)
    score += int(row['BMI'] >= 25)
    score += int(row['Sex'] == 1)
    score += int(row['HighBP'] == 1)
    score += int(row['PhysActivity'] == 0)
    return score

def assign_risk_level(score):
    if score < 2:
        return 0
    else:
        return 1

df['risk_score'] = df.apply(calculate_custom_risk_score, axis=1)
df['rule_pred'] = df['risk_score'].apply(assign_risk_level)

df.shape


df['Diabetes_binary'] = df['Diabetes_012'].apply(lambda x: 1 if x > 0 else 0)

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)


#Encoding

drop_list = ["risk_score", "Diabetes_012"]
df.drop(drop_list, axis=1, inplace=True)
cat_cols, num_cols, cat_but_car,  num_but_cat = grab_col_names(df)

if 'Diabetes_binary' in cat_cols:
    cat_cols.remove('Diabetes_binary')
if 'Diabetes_binary' in num_cols:
    num_cols.remove('Diabetes_binary')

# Label Encoding

def binary_cols(dataframe):
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in ['int64', 'float64'] and dataframe[col].nunique() <= 2]
    return binary_cols
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
binary_cols = binary_cols(df)
for col in binary_cols:
    df = label_encoder(df, col)
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()

# num_cols
scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head(10)

print(df.columns)
#Modelling
# Örneğin diabetes 1 ve 2 için binary hedef oluşturalım
y = df["Diabetes_binary"]
X = df.drop(["Diabetes_binary"], axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=17)

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns


# Not: X_train, X_test, y_train, y_test değişkenlerinin
# daha önce doğru bir şekilde oluşturulduğunu varsayarız.

# --- Sınıflandırma Modellerini Tanımla ---
# Yeni modeller eklendi: CatBoost ve GaussianNB
classifiers = [
    ('LR', LogisticRegression(random_state=17, max_iter=500)),
    #('KNN', KNeighborsClassifier()),
    ('GNB', GaussianNB()), # Yeni model eklendi
    ('RF', RandomForestClassifier(random_state=17)),
    #('GBM', GradientBoostingClassifier(random_state=17)),
    ("XGB", XGBClassifier(random_state=17, use_label_encoder=False, eval_metric='logloss')),
    ("LGBM", LGBMClassifier(random_state=17)),
    #("CatBoost", CatBoostClassifier(verbose=False, random_state=17)) # Yeni model eklendi
]

# --- Sonuçları Saklamak İçin Listeler ---
roc_auc_scores = []
execution_times = []
model_names = []
print("--- Modeller Eğitiliyor ve Değerlendiriliyor ---")

for name, classifier in classifiers:
    start_time = time.time()
    model_names.append(name)

    # Modeli eğitim verisi üzerinde eğit
    classifier.fit(X_train, y_train)

    # Cross-validation ile ROC AUC skorunu hesapla (genelleme yeteneğini görmek için)
    cv_score = np.mean(cross_val_score(classifier, X_train, y_train, cv=5, scoring="roc_auc"))
    roc_auc_scores.append(cv_score)

    # Modelin test verisi üzerindeki tahminlerini yap
    y_pred = classifier.predict(X_test)

    # Modelin yürütme süresini hesapla
    execution_time = time.time() - start_time
    execution_times.append(execution_time)

    print(f"\n--- {name} Model Raporu ---")
    print(f"Eğitim ROC AUC (Çapraz Doğrulama): {round(cv_score, 4)}")
    print(f"Yürütme Süresi: {round(execution_time, 2)} saniye")

    # Test verisi üzerinde confusion matrix ve classification report yazdır
    print("\nTest Verisi Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nTest Verisi Classification Report:")
    print(classification_report(y_test, y_pred))

# --- Performans ve Yürütme Süresi Grafikleri ---
# (Bu kısım, önceki yanıttaki kodla aynıdır)
fig, ax = plt.subplots(1, 2, figsize=(20, 8))

sns.barplot(x=model_names, y=roc_auc_scores, ax=ax[0])
ax[0].set_xlabel("Model")
ax[0].set_ylabel("ROC AUC Skoru")
ax[0].set_title("Modellerin Performansı (5-Katlı Çapraz Doğrulama)")
ax[0].set_ylim(0.5, 1.0)
ax[0].grid(axis='y', linestyle='--')

sns.barplot(x=model_names, y=execution_times, ax=ax[1])
ax[1].set_xlabel("Model")
ax[1].set_ylabel("Yürütme Süresi (saniye)")
ax[1].set_title("Modellerin Yürütme Süreleri")

plt.tight_layout()
plt.show(block=True)


for name, classifier in classifiers:
    print(f"\n### {name} Modeli Değerlendiriliyor ###")

    # Modeli eğitim verisi üzerinde eğit
    classifier.fit(X_train, y_train)

    # Test verisi üzerinde tahmin yap
    y_test_pred = classifier.predict(X_test)

    # Karmaşıklık matrisini hesapla
    cm = confusion_matrix(y_test, y_test_pred)

    # Karmaşıklık matrisini görselleştir
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Tahmin: Yok', 'Tahmin: Var'],
                yticklabels=['Gerçek: Yok', 'Gerçek: Var'])
    plt.title(f'Optimizasyon Öncesi {name} Modeli Karmaşıklık Matrisi')
    plt.xlabel('Tahmin Edilen Değerler')
    plt.ylabel('Gerçek Değerler')
    plt.show(block=True)


##Hiperparametre Optimizasyonu###
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns

# Not: X_train, X_test, y_train, y_test değişkenlerinin
# daha önce doğru bir şekilde oluşturulduğunu varsayıyoruz.

# --- Modeller ve Hiperparametre Izgaraları ---
# Diyabet ikili sınıflandırması için uygun modeller ve param_grid'ler tanımlandı.

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Not: X_train, X_test, y_train, y_test değişkenlerinin
# daha önce doğru bir şekilde oluşturulduğunu varsayıyoruz.

# Sınıf dengesizliğini ele almak için manuel olarak sınıf ağırlıklarını belirleyelim
from sklearn.model_selection import GridSearchCV
import numpy as np
import time
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# Sınıf ağırlıkları ve scale_pos_weight
class_weights_dict = {0: 1, 1: 12}
scale_pos_weight_value = y_train.value_counts()[0] / y_train.value_counts()[1]

# Modeller ve hiperparametreler
models_to_optimize = {
    'GNB': {
        'model': GaussianNB(),
        'param_grid': {
            'var_smoothing': np.logspace(0, -9, num=10)
        }
    },
    #'KNN': {
     #   'model': KNeighborsClassifier(),
      #  'param_grid': {
       #     'n_neighbors': [3, 5, 9],
        #    'weights': ['uniform']
       # }
   # },
    'RF': {
        'model': RandomForestClassifier(random_state=17),
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth': [5, 9],
            'class_weight': [class_weights_dict]
        }
    },
    'XGB': {
        # use_label_encoder parametresi kaldırıldı
        'model': XGBClassifier(random_state=17, eval_metric='logloss', use_label_encoder=None),
        'param_grid': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'scale_pos_weight': [scale_pos_weight_value]
        }
    },
    'LGBM': {
        'model': LGBMClassifier(random_state=17),
        'param_grid': {
            'n_estimators': [100],
            'learning_rate': [0.1],
            'max_depth': [-1],
            'scale_pos_weight': [scale_pos_weight_value]
        }
    }
}

final_results = []
best_models = {}

print("--- Seçilen Modeller İçin Hiperparametre Optimizasyonu Başlıyor ---")

for model_name, config in models_to_optimize.items():
    print(f"\n### {model_name} Modeli Optimize Ediliyor ###")

    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['param_grid'],
        cv=3,
        scoring='roc_auc',
        # Paralel işlem sayısını 1 yaptık, böylece pickle hatası ve memory error önlenir
        n_jobs=1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_score = grid_search.best_score_
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_

    print(f"En iyi parametreler: {best_params}")
    print(f"En iyi doğrulama ROC AUC skoru: {best_score:.4f}")

    best_models[model_name] = best_estimator

    # Test seti üzerinde final değerlendirme
    y_test_pred = best_estimator.predict(X_test)
    y_test_proba = best_estimator.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, y_test_proba)

    final_results.append({
        'Model': model_name,
        'En İyi Parametreler': best_params,
        'Doğrulama ROC AUC': best_score,
        'Test ROC AUC': test_auc,
        'Yürütme Süresi': time.time() - start_time
    })

    print(f"Test seti üzerinde final ROC AUC skoru: {test_auc:.4f}")
    print("\nFinal Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nFinal Classification Report:")
    print(classification_report(y_test, y_test_pred))


# Tüm sonuçları bir DataFrame'de toplama
results_df = pd.DataFrame(final_results)
print("\n--- Tüm Modellerin Optimizasyon Sonuçları ---")
print(results_df.sort_values(by='Test ROC AUC', ascending=False))

# --- Performans ve Yürütme Süresi Grafikleri ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# final_results listesini DataFrame'e çevir
results_df = pd.DataFrame(final_results)

fig, ax = plt.subplots(1, 2, figsize=(18, 8))

sns.barplot(x=results_df['Model'], y=results_df['Test ROC AUC'], ax=ax[0])
ax[0].set_xlabel("Model")
ax[0].set_ylabel("ROC AUC Skoru")
ax[0].set_title("Modellerin Performansı (Test Verisi)")
ax[0].set_ylim(0.5, 1.0)
ax[0].grid(axis='y', linestyle='--')

sns.barplot(x=results_df['Model'], y=results_df['Yürütme Süresi'], ax=ax[1])
ax[1].set_xlabel("Model")
ax[1].set_ylabel("Yürütme Süresi (saniye)")
ax[1].set_title("Modellerin Yürütme Süreleri")

plt.tight_layout()
plt.show(block=True)

print("\n--- Modeller İçin Ayrı Ayrı Karmaşıklık Matrisleri ---")


# Best_models sözlüğündeki her model için döngü
for model_name, best_estimator in best_models.items():
    # Modelin test verisi üzerindeki tahminlerini al
    y_test_pred = best_estimator.predict(X_test)

    # Karmaşıklık matrisini hesapla
    cm = confusion_matrix(y_test, y_test_pred)

    # Karmaşıklık matrisini görselleştir
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Tahmin: Yok', 'Tahmin: Var'],
                yticklabels=['Gerçek: Yok', 'Gerçek: Var'])
    plt.title(f'{model_name} Modeli Karmaşıklık Matrisi')
    plt.xlabel('Tahmin Edilen Değerler')
    plt.ylabel('Gerçek Değerler')
    plt.show(block=True)
###############
import numpy as np
import pandas as pd
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
###################################################
#Stacking
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
import numpy as np

# 1️⃣ En iyi modelleri recall / precision'a göre seç
# Recall için en iyi RF, Precision için en iyi GNB
rf_model = best_models['RF']
gnb_model = best_models['GNB']

# --- Olasılık Tahminleri ---
rf_proba_train = best_models['RF'].predict_proba(X_train)
gnb_proba_train = best_models['GNB'].predict_proba(X_train)

rf_proba_test = best_models['RF'].predict_proba(X_test)
gnb_proba_test = best_models['GNB'].predict_proba(X_test)

# --- METRİK TABANLI ÖLÇEKLER ---
recall_pos_rf = recall_score(y_train, best_models['RF'].predict(X_train), pos_label=1)
precision_neg_gnb = precision_score(y_train, best_models['GNB'].predict(X_train), pos_label=0)

# --- Ölçekleme ---
rf_proba_train_scaled = rf_proba_train * [1, recall_pos_rf]
gnb_proba_train_scaled = gnb_proba_train * [precision_neg_gnb, 1]

rf_proba_test_scaled = rf_proba_test * [1, recall_pos_rf]
gnb_proba_test_scaled = gnb_proba_test * [precision_neg_gnb, 1]

# --- META MODEL VERİLERİ ---
meta_X_train = np.hstack([
    rf_proba_train_scaled,
    gnb_proba_train_scaled,
    X_train
])
meta_X_test = np.hstack([
    rf_proba_test_scaled,
    gnb_proba_test_scaled,
    X_test
])

# --- Standartlaştırma ---
scaler = StandardScaler()
meta_X_train = scaler.fit_transform(meta_X_train)
meta_X_test = scaler.transform(meta_X_test)

# --- META MODEL ---
#scale_pos_weight_value = y_train.value_counts()[0] / y_train.value_counts()[1]

#meta_model = XGBClassifier(
  #  n_estimators=1000,
  #  learning_rate=0.05,
   # max_depth=4,
    #subsample=0.8,
    #colsample_bytree=0.8,
    #random_state=42,
    #eval_metric='logloss',
    #use_label_encoder=False,
    #scale_pos_weight=scale_pos_weight_value
#)

# Early stopping için validation set ayarlayalım
#meta_model.fit(
 #   meta_X_train, y_train,
  #  eval_set=[(meta_X_test, y_test)],
   # verbose=True
#)

#y_pred_proba = meta_model.predict_proba(meta_X_test)[:, 1]
#print("Meta Model ROC AUC:", roc_auc_score(y_test, y_pred_proba))

# --- Performans ---
#y_pred = meta_model.predict(meta_X_test)
#print(classification_report(y_test, y_pred))
# Confusion Matrix
#cm = confusion_matrix(y_test, y_pred)

#plt.figure(figsize=(6,5))
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
 #           xticklabels=['Tahmin: Negatif', 'Tahmin: Pozitif'],
  #          yticklabels=['Gerçek: Negatif', 'Gerçek: Pozitif'])
#plt.title("Meta Model Confusion Matrix")
#plt.xlabel("Tahmin")
#plt.ylabel("Gerçek")
#plt.show(block=True)

#################################################
#
#RF meta
# Pozitif sınıfa yüksek ağırlık vererek recall'u artırmayı hedefliyoruz
class_weights = {0: 1, 1: 15}
meta_model_rf = RandomForestClassifier(
    random_state=42,
    n_estimators=200,
    max_depth=5,
    class_weight=class_weights
)

# Modeli eğit
meta_model_rf.fit(meta_X_train, y_train)

# Test seti üzerinde tahminler
y_pred_rf = meta_model_rf.predict(meta_X_test)
y_pred_proba_rf = meta_model_rf.predict_proba(meta_X_test)[:, 1]

# Performans metrikleri
print("Random Forest Meta Model Performansı (class_weight ile):")
print(classification_report(y_test, y_pred_rf))
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")

# Confusion Matrix görselleştirme
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6,5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Tahmin: Negatif', 'Tahmin: Pozitif'],
            yticklabels=['Gerçek: Negatif', 'Gerçek: Pozitif'])
plt.title("Random Forest Meta Model Confusion Matrix (class_weight ile)")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.show(block=True)
#################################################
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

rf_model = best_models['RF']
gnb_model = best_models['GNB']
lgbm_model = best_models['LGBM']


base_models = [
    ('RF', rf_model),
    ('GNB', gnb_model),
    ('LGBM', lgbm_model)
]


# Voting classifier oluştur
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('gnb', gnb_model),
        ('lgbm', lgbm_model)
    ],
    voting='soft',   # olasılık tabanlı oylama (daha iyi performans için)
    weights=[26, 2, 1]  # istersen ağırlık verebilirsin, RF ve LGBM'e daha fazla ağırlık gibi
)

# Eğit
voting_clf.fit(X_train, y_train)

# Tahmin
y_pred = voting_clf.predict(X_test)
y_proba = voting_clf.predict_proba(X_test)[:, 1]

# Performans raporu
print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Tahmin: Negatif', 'Tahmin: Pozitif'],
            yticklabels=['Gerçek: Negatif', 'Gerçek: Pozitif'])
plt.title("Voting Classifier Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.show(block=True)



# Stacking modeli oluştur
from sklearn.ensemble import StackingClassifier

#stacking_clf = StackingClassifier(
#    estimators=base_models,
#    final_estimator=meta_model,
#    cv=3,
#    n_jobs=-1,
#    passthrough=True  # base model feature’larını da meta modele aktarır
#)


# Modeli eğit
#stacking_clf.fit(X_train, y_train)

# Tahmin
#y_pred = stacking_clf.predict(X_test)
#y_proba = stacking_clf.predict_proba(X_test)[:, 1]

# Performans raporu
#print(classification_report(y_test, y_pred))
#print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Tahmin: Negatif', 'Tahmin: Pozitif'],
            yticklabels=['Gerçek: Negatif', 'Gerçek: Pozitif'])
plt.title("Stacking Classifier Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.show(block=True)


###################################################
# Not: Önceki kodunuzdan gelen 'X_train', 'y_train', 'X_test', 'y_test' ve 'df' değişkenlerinin
# zaten mevcut olduğunu varsayıyoruz.
# Ayrıca, 'best_models' sözlüğü de doldurulmuştur ve en iyi modelleri içermektedir.

# --- 1. En İyi Modeli Seçme ---
# Optimizasyon sonuçlarına göre en iyi modelin XGBoost olduğunu varsayalım.
# Bu durumda, 'best_models' sözlüğünden XGBoost modelini doğrudan alabiliriz.
final_model_name = 'RF'
final_model = best_models[final_model_name]
print(f"En iyi model olarak '{final_model_name}' seçildi.")

# --- 2. Özellik Önem Derecesini Analiz Etme (Feature Importance) ---
print("\n--- Özellik Önem Derecesi Analizi ---")

def plot_importance(model, features, num=50):
    if not hasattr(model, 'feature_importances_'):
        print(f"Hata: {model.__class__.__name__} modeli 'feature_importances_' özniteliğine sahip değil.")
        return

    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Özellik Önem Derecesi')
    plt.xlabel('Önem Derecesi')
    plt.ylabel('Özellikler')
    plt.tight_layout()
    plt.show(block=True)

# final_model ve X_train'in sütun adlarını kullanarak önem derecesini çizelim.
plot_importance(final_model, pd.DataFrame(X_train, columns=X_test.columns))

# SHAP analizi de yapabiliriz
print("\n--- SHAP Analizi: Global Özellik Önem Derecesi ---")

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)
plt.show(block=True)

print("\n--- Tekil Tahmin Açıklaması Grafiği (SHAP Waterfall) ---")


import numpy as np
import shap

# (İstersen probability alanında çalış: model_output='probability')
explainer = shap.TreeExplainer(final_model)  # veya TreeExplainer(final_model, model_output='probability')

sample_index = 42
sample_data = X_test.iloc[[sample_index]]   # tek örnek, DataFrame

shap_raw = explainer.shap_values(sample_data)
expected = explainer.expected_value

def pick_single_row_for_positive(shap_vals, expected_val):
    # base (skaler) seç
    if isinstance(expected_val, (list, np.ndarray)):
        base = expected_val[1]  # class=1
    else:
        base = expected_val     # skaler

    # SHAP değerini (n_features,) şeklinde tek satır olarak seç
    if isinstance(shap_vals, list):
        row = shap_vals[1][0]              # [class1][sample0] -> (n_features,)
    elif isinstance(shap_vals, np.ndarray):
        if shap_vals.ndim == 3:
            # (n_samples, n_features, n_classes)
            row = shap_vals[0, :, 1]       # sample0, class1 -> (n_features,)
        elif shap_vals.ndim == 2:
            # (n_samples, n_features)
            row = shap_vals[0, :]          # sample0 -> (n_features,)
        else:
            raise ValueError(f"Beklenmeyen shap_values boyutu: {shap_vals.shape}")
    else:
        # güvenli dönüşüm
        arr = np.array(shap_vals)
        if arr.ndim == 3:
            row = arr[0, :, 1]
        elif arr.ndim == 2:
            row = arr[0, :]
        else:
            raise ValueError(f"Beklenmeyen shap_values tipi/şekli: {type(shap_vals)}, {arr.shape}")

    return row, base

shap_row, base_val = pick_single_row_for_positive(shap_raw, expected)

# --- Waterfall ---
ex = shap.Explanation(
    values=shap_row,                         # (n_features,)
    base_values=base_val,                    # skaler
    data=sample_data.iloc[0],                # Series
    feature_names=sample_data.columns.tolist()
)
shap.plots.waterfall(ex, max_display=20)

# --- Force (matplotlib=True ile stabil) ---
shap.plots.force(base_val, shap_row, sample_data.iloc[0], matplotlib=True)

# Kontrol bilgisi (gerekirse)
#print(type(shap_raw), np.array(shap_raw).shape, expected)

# --- 3. Ensemble (Birleştirilmiş) Model Oluşturma ve Değerlendirme ---
print("\n--- Ensemble Modeli Oluşturuluyor ve Değerlendiriliyor ---")

# a) Makine öğrenmesi modelinin olasılık tahminlerini al
# Model ve özellik sütunları
model = best_models['RF']
feature_cols = [col for col in df.columns if col != 'Diabetes_binary']  # hedef hariç

# Tahminleri yap ve DataFrame'e ekle
df['rule_pred'] = model.predict(df[feature_cols])

# Test seti indexlerine göre tahminleri al
rule_preds = df.loc[X_test.index, 'rule_pred'].values

# ml_probs zaten tanımlıysa
rule_probs = np.zeros_like(ml_probs)
rule_probs[np.arange(len(rule_preds)), rule_preds] = 1



# c) Ensemble (Birleştirme) işlemi
ensemble_probs = 0.7 * ml_probs + 0.3 * rule_probs
ensemble_preds = np.argmax(ensemble_probs, axis=1)

# d) Ensemble modelin performansını değerlendir
print(f"Ensemble Model F1 Skoru (weighted): {f1_score(y_test, ensemble_preds, average='weighted'):.4f}")

# Detaylı raporlar
print("\nEnsemble Model Sınıflandırma Raporu:")
print(classification_report(y_test, ensemble_preds))

# Karmaşıklık matrisi görselleştirmesi
cm = confusion_matrix(y_test, ensemble_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Tahmin: Yok', 'Tahmin: Var'],
            yticklabels=['Gerçek: Yok', 'Gerçek: Var'])
plt.title('Ensemble Modeli Karmaşıklık Matrisi')
plt.xlabel('Tahmin Edilen Değerler')
plt.ylabel('Gerçek Değerler')
plt.show(block=True)
###############
import joblib

# En iyi modelinizi final_model değişkenine atadığınızı varsayıyoruz
# final_model = best_models['XGB']

# Modeli bir dosyaya kaydedin
import joblib

# Modeli kaydet
joblib.dump(meta_model_rf, 'meta_model_rf.pkl')

# Daha sonra tekrar yüklemek için
loaded_model = joblib.load('meta_model_rf.pkl')

# Yüklenen modelle tahmin yapabilirsin
y_pred = loaded_model.predict(meta_X_test)

print("Final model başarıyla 'final_model.pkl' olarak kaydedildi.")