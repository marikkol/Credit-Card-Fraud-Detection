import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN





# Data Preparation


train_df1 = pd.read_csv('train_identity.csv')   # [144233 rows x 41 columns]
train_df2 = pd.read_csv('train_transaction.csv')   # [590540 rows x 394 columns]
train_df = train_df2.join(train_df1.set_index('TransactionID'), on='TransactionID').set_index('TransactionID')   # [590540 rows x 433 columns]
print(train_df)

# not labeled test!
test_df1 = pd.read_csv('test_identity.csv')
test_df2 = pd.read_csv('test_transaction.csv')
test_df = test_df2.join(test_df1.set_index('TransactionID'), on='TransactionID').set_index('TransactionID')   # [506691 rows x 432 columns]
print(test_df)


# rename test cols (id_01...id_38)
rename_dic = {}
for col_name in test_df.columns:
    if '-' in col_name:
        new_name = col_name[:2] + '_' + col_name[3:]
        rename_dic[col_name] = new_name
test_df = test_df.rename(columns=rename_dic)
test_df = test_df.reindex(columns=train_df.columns.tolist().remove('isFraud'))   # reordering as the train



# Missing Values


missing_val_df = train_df.isnull().sum().to_frame()
missing_val_df = missing_val_df.apply(lambda x: x/len(train_df)*100, axis=1)\
    .rename(columns={0: 'Missing Ratio'}).sort_values(by=['Missing Ratio'])
print(missing_val_df)   # some attributes reached up to 99%

# removing the missing values that reaches above 60%
above_60_df = missing_val_df[missing_val_df['Missing Ratio'] > 60]
print(f'mun missing values features: {len(missing_val_df)}\n'
      f'mun above 60 features: {len(above_60_df)}')
remove_cols = above_60_df.index.values
train_df = train_df.drop(columns=remove_cols)
print(train_df)

# median for numeric features
print(f"Count total NaN in a DataFrame : {train_df.isnull().sum().sum()}")
num_cols = train_df._get_numeric_data().columns.tolist()
train_df[num_cols] = train_df[num_cols].fillna(train_df[num_cols].median())
print(f"Count total NaN in a DataFrame : {train_df.isnull().sum().sum()}")

# mode for categorical features
cat_cols = list(set(train_df.columns)-set(num_cols))
train_df[cat_cols] = train_df[cat_cols].fillna(train_df[cat_cols].mode().iloc[0])
print(f"Count total NaN in a DataFrame : {train_df.isnull().sum().sum()}"


# Encoding Catigorical Features


nunique = train_df[cat_cols].nunique()
print(nunique)   # 8 of them contained only two levels of cardinality (e.g., attributes with false and true)

# replacing the two level features by 0 and 1
two_level_features = ['M9','M8','M7','M2','M6','M3','M5','M1']
for feature in two_level_features:
    uni_vals = train_df[feature].unique()
    print(uni_vals)   # the unique values for these col are: ['T' 'F']
train_df[two_level_features] = train_df[two_level_features].replace({'T': 1, 'F': 0})

# straightforward one-hot encoding for the rest of the features
rets_cat_features = [col for col in cat_cols if col not in two_level_features]
df = pd.get_dummies(train_df, columns=rets_cat_features)
print(df)



# Feature Scaling


# MinMaxScaler scaling:  x' = (x - min(x))/max(x) - min(x)
num_cols.remove('isFraud')
min_max_scaler = MinMaxScaler()
df[num_cols] = min_max_scaler.fit_transform(df[num_cols])
print(df)



# Feature Selection


# Filter method - hybrid feature selection for numerical features - correlation matrix

corrmat = df[num_cols].corr().abs()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
# plt.show()
upper_tri = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(bool))   # selecting the upper traingular
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.80)]
print(f"num of columns to drop: {len(to_drop)}")
df.drop(to_drop, axis=1, inplace=True)
print(df)

# Train-Test Split
df_train, df_test = train_test_split(df, test_size=0.3, random_state=1)

# Dividing data into X and y variables
y_train, y_test = df_train.pop('isFraud'), df_test.pop('isFraud')
X_train, X_test = df_train, df_test

# Wrapper method - SVM-RFE

rand_col_select = random.sample(list(X_train.columns), 30)
X_train, X_test = X_train[rand_col_select], X_test[rand_col_select]

svc = SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=0.2, min_features_to_select=10, verbose=1)
pipeline = Pipeline(steps=[('s',rfecv),('m',svc)])
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=cv, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
final_features = X_train.columns[rfecv.support_]
print(f"Selected features: {final_features}")
n_features = rfecv.n_features_
print(f"Optimal number of features {n_features}")

# Plot showing the Cross Validation score
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

X_train_rfe = X_train[final_features]
X_test_rfe = X_test[final_features]
print(X_train_rfe)
print(X_test_rfe)



# Data Resampling


# The classes are heavily skewed
print('No Frauds', round(y_train.value_counts()[0]/len(y_train) * 100,2), '% of the dataset')
print('Frauds', round(y_train.value_counts()[1]/len(y_train) * 100,2), '% of the dataset')
colors = ["#0101DF", "#DF0101"]
sns.countplot('isFraud', data=y_train, palette=colors)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
plt.show()

X_res, y_res = sme.fit_resample(X_train_rfe, y_train)




