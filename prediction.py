import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
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
from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# read the train, test and submission data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')    
submission_df = pd.read_csv('sample_submission.csv')

# concat train and test data
df = pd.concat([train_df, test_df], ignore_index=True)

# display the first 5 rows of the data
print(df.head())    

# show the shape of the data
print(df.shape)

# change the date column to datetime
df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])

# display the first 5 rows of the data
print(df.head())    

# add features to the data
df['Year'] = df['Policy Start Date'].dt.year
df['Quarter'] = df['Policy Start Date'].dt.quarter
df['Month'] = df['Policy Start Date'].dt.month
df['Day'] = df['Policy Start Date'].dt.day
df['day_of_week'] = df['Policy Start Date'].dt.dayofweek
df['week_of_year'] = df['Policy Start Date'].dt.isocalendar().week

df['day_sin'] = np.sin(2 * np.pi * df['Day'] / 365)
df['day_cos'] = np.cos(2 * np.pi * df['Day'] / 365)

df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

df['year_sin'] = np.sin(2 * np.pi * df['Year'] / 7.0)
df['year_cos'] = np.cos(2 * np.pi * df['Year'] / 7.0)
df['Group'] = (df['Year'] - 2010) * 48 + df['Month'] * 4 + df['Day'] // 7

df['Quarter'] = df['Quarter'].astype(str) 
df['Month'] = df['Month'].astype(str)
df['day_of_week'] = df['day_of_week'].astype(str)
df['week_of_year'] = df['week_of_year'].astype(str)


# drop the policy start date column
df.drop(columns=['Policy Start Date'], axis=1, inplace=True)

# display the first 5 rows of the data
print(df.head())    

# info of the data
print(df.info())

# describe the data
def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

check_df(df)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    num_cols = [col for col in num_cols if col not in ['Premium Amount', 'id']]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')


    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

#df['Premium Amount'] = np.log1p(df['Premium Amount'])

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

for col in cat_cols:
    cat_summary(df, col)


def cat_summary(dataframe,col_name,plot=False):
  print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                      'Ratio':100*dataframe[col_name].value_counts()/len(dataframe)}))
  print('#################################################################')

  if plot:
    sns.countplot(data=dataframe,x=dataframe[col_name])
    plt.show(block=True) 

for col in cat_cols:
    cat_summary(df, col, plot=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df,col, plot=True)
    else:
        cat_summary(df,col, plot=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("#################################################################")

    if plot:
        sns.histplot(data=dataframe, x=numerical_col)
        plt.show(block=True)

num_summary(df,"Premium Amount")        

for col in num_cols:
  num_summary(df,col,plot=True)


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col, observed=True)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Premium Amount", col)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Premium Amount", col)


df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n\n\n")
    if na_name:
        return na_columns
    
missing_values_table(df)

def quick_missing_imp(data, num_method="median", cat_length=20, target='Premium Amount'):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]

    temp_target = data[target]

    print("# Before")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis = 0)

    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis = 0)
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis = 0)

    data[target] = temp_target

    print("# After \n Imputation method is 'MODE' for categorical variables")
    print("Imputation method is ' "     " 'MEAN' for numerical variables")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data

df = quick_missing_imp(df, num_method="median", cat_length=17)

df.isnull().sum()

def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    low_limit = quantile_one - 1.5 * interquantile_range
    up_limit = quantile_three + 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    if col != "Premium Amount":
        print(col, check_outlier(df, col))

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    if col != "Premium Amount":
        replace_with_thresholds(df, col)

for col in num_cols:
    if col != "Premium Amount":
        print(col, check_outlier(df, col))

label_encoder = {col : LabelEncoder() for col in cat_cols}

for col in cat_cols:
    le = label_encoder[col]
    le.fit(df[col])
    df[col] = le.transform(df[col])

df.head()

df['week_of_year'] = df['week_of_year'].astype(int)

from lightgbm import LGBMClassifier

train_df.shape, test_df.shape

train_df = df[df['Premium Amount'].notna()].copy()
test_df = df[df['Premium Amount'].isna()].copy()

train_df.shape, test_df.shape

test_df.drop(columns=['Premium Amount'], axis=1, inplace=True)

train_df.shape, test_df.shape

X = train_df.drop(columns=['Premium Amount'])
y = train_df['Premium Amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import mean_absolute_percentage_error

def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

lgbm_params = {
    'num_leaves': 71,
    'learning_rate': 0.05412467152424433,
    'n_estimators': 595,
    'max_depth': 12,
    'min_data_in_leaf': 97,
    'bagging_fraction': 0.5200288825838669,
    'feature_fraction': 0.9881738491942492,
    'n_jobs': -1,
    'verbose': -1
}

lgbm_model = LGBMRegressor(**lgbm_params)

lgbm_model.fit(X_train, y_train)

y_pred = lgbm_model.predict(X_test)

lgbm_mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"LightGBM MAPE: {lgbm_mape: .4f}")

test_preds = lgbm_model.predict(test_df)

submission = pd.DataFrame({'id': test_df['id'], 'Premium Amount': test_preds})

summission_filename = 'submission_lgbm1.csv'
submission.to_csv(summission_filename, index=False)
print(f"Submission file saved as {summission_filename}")