import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler




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
print(f"Count total NaN in a DataFrame : {train_df.isnull().sum().sum()}")


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
rand_col_select = random.sample(list(X_train.columns), 15)
X_train, X_test = X_train[rand_col_select], X_test[rand_col_select]

svc = SVC(kernel="linear")
min_features_to_select = 10
rfecv = RFECV(estimator=svc, step=0.2, cv=StratifiedKFold(2), min_features_to_select=min_features_to_select, verbose=1)
rfecv = rfecv.fit(X_train, y_train)
print('Optimal number of features :', rfecv.n_features_)
# pipeline = Pipeline(steps=[('s',rfecv),('m',svc)])
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
# n_scores = cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=5, error_score='raise')
# print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
final_features = X_train.columns[rfecv.support_]
print(f"Selected features: {final_features}")

# Plot showing the Cross Validation score
num_feat_list = [len(rand_col_select)]   # list of Number of features selected for plotting
while (num_feat_list[-1]-num_feat_list[-1]*0.2) >= 10:
    num_feat_list.append(round(num_feat_list[-1]-num_feat_list[-1]*0.2))
if num_feat_list[-1] != min_features_to_select:
    num_feat_list.append(min_features_to_select)

print(num_feat_list[::-1])
print(rfecv.grid_scores_)
print(type(rfecv.grid_scores_))

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (accuracy)")
plt.plot(num_feat_list[::-1], np.mean(rfecv.grid_scores_, axis=1))   # taking the mean of cv scores

plt.show()

X_train_rfe = X_train[final_features]
X_test_rfe = X_test[final_features]
print(X_train_rfe)
print(X_test_rfe)


# Data Resampling

# The classes are heavily skewed
colors = ["#0101DF", "#DF0101"]
def plot_dist(y_data):
    print('No Frauds', round(y_data.value_counts()[0]/len(y_data) * 100,2), '% of the dataset')
    print('Frauds', round(y_data.value_counts()[1]/len(y_data) * 100,2), '% of the dataset')
    print(y_data)
    ser_count_n = y_data.value_counts(normalize=True)
    print(ser_count_n)
    ser_count_n = ser_count_n.mul(100)
    print(ser_count_n)
    df_count_n = ser_count_n.rename('percent').reset_index().rename(columns={"index": "Is Fraud"})
    print(df_count_n)
    g = sns.catplot(x='Is Fraud',y='percent',kind='bar',data=df_count_n)
    plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
    g.ax.set_ylim(0,100)
    for p in g.ax.patches:
        txt = str(p.get_height().round(2)) + '%'
        txt_x = p.get_x()
        txt_y = p.get_height()
        g.ax.text(txt_x,txt_y,txt)
    plt.tight_layout()
    plt.show()

    ser_count = y_data.value_counts()
    print(ser_count)
    df_count = ser_count.rename('examples').reset_index().rename(columns={"index": "Is Fraud"})
    print(df_count)
    g = sns.catplot(x='Is Fraud',y='examples',kind='bar',data=df_count)
    plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
    for p in g.ax.patches:
        txt = str(int(p.get_height().round()))
        txt_x = p.get_x()
        txt_y = p.get_height()
        g.ax.text(txt_x,txt_y,txt)
    plt.tight_layout()
    plt.show()

plot_dist(y_train)


# sns.countplot(y=y_train, palette=colors)
# plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
# plt.show()

# Resample the dataset using SMOTE-ENN
sme = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='all'), random_state=42)
X_res, y_res = sme.fit_resample(X_train_rfe, y_train)

# The classes dist after SMOTE-ENN
plot_dist(y_res)

# print('No Frauds', round(y_res.value_counts()[0]/len(y_res) * 100,2), '% of the dataset')
# print('Frauds', round(y_res.value_counts()[1]/len(y_res) * 100,2), '% of the dataset')
# ser_count_n = y_res.value_counts(normalize=True)
# print(ser_count_n)
# ser_count_n = ser_count_n.mul(100)
# print(ser_count_n)
# df_count_n = ser_count_n.rename('percent').reset_index().rename(columns={"index": "Is Fraud"})
# print(df_count_n)
# g = sns.catplot(x='Is Fraud',y='percent',kind='bar',data=df_count_n)
# plt.title('Class Distributions SMOTE-ENN \n (0: No Fraud || 1: Fraud)', fontsize=14)
# g.ax.set_ylim(0,100)
# for p in g.ax.patches:
#     txt = str(p.get_height().round(2)) + '%'
#     txt_x = p.get_x()
#     txt_y = p.get_height()
#     g.ax.text(txt_x,txt_y,txt)
# plt.tight_layout()
# plt.show()
#
# ser_count = y_res.value_counts()
# print(ser_count)
# df_count = ser_count.rename('examples').reset_index().rename(columns={"index": "Is Fraud"})
# print(df_count)
# g = sns.catplot(x='Is Fraud',y='examples',kind='bar',data=df_count)
# plt.title('Class Distributions SMOTE-ENN \n (0: No Fraud || 1: Fraud)', fontsize=14)
# for p in g.ax.patches:
#     txt = str(p.get_height().round())
#     txt_x = p.get_x()
#     txt_y = p.get_height()
#     g.ax.text(txt_x,txt_y,txt)
# plt.tight_layout()
# plt.show()

# Under-sample the majority class
rus = RandomUnderSampler(random_state=42)
rus.fit(X_res, y_res)
X_res, y_res = rus.fit_resample(X_res, y_res)


# The classes dist after Under-sample
plot_dist(y_res)


# print('No Frauds', round(y_res.value_counts()[0]/len(y_res) * 100,2), '% of the dataset')
# print('Frauds', round(y_res.value_counts()[1]/len(y_res) * 100,2), '% of the dataset')
# ser_count_n = y_res.value_counts(normalize=True)
# print(ser_count_n)
# ser_count_n = ser_count_n.mul(100)
# print(ser_count_n)
# df_count_n = ser_count_n.rename('percent').reset_index().rename(columns={"index": "Is Fraud"})
# print(df_count_n)
# g = sns.catplot(x='Is Fraud',y='percent',kind='bar',data=df_count_n)
# plt.title('Class Distributions SMOTE-ENN \n (0: No Fraud || 1: Fraud)', fontsize=14)
# g.ax.set_ylim(0,100)
# for p in g.ax.patches:
#     txt = str(p.get_height().round(2)) + '%'
#     txt_x = p.get_x()
#     txt_y = p.get_height()
#     g.ax.text(txt_x,txt_y,txt)
# plt.tight_layout()
# plt.show()
#
# ser_count = y_res.value_counts()
# print(ser_count)
# df_count = ser_count.rename('examples').reset_index().rename(columns={"index": "Is Fraud"})
# print(df_count)
# g = sns.catplot(x='Is Fraud',y='examples',kind='bar',data=df_count)
# plt.title('Class Distributions SMOTE-ENN \n (0: No Fraud || 1: Fraud)', fontsize=14)
# for p in g.ax.patches:
#     txt = str(p.get_height().round())
#     txt_x = p.get_x()
#     txt_y = p.get_height()
#     g.ax.text(txt_x,txt_y,txt)
# plt.tight_layout()
# plt.show()

# save arrays to one file in compressed format
X_train = X_res.to_numpy()
y_train = y_res.to_numpy()
np.savez_compressed('preprossesing_data.npz', X_train, y_train, X_test, y_test)

