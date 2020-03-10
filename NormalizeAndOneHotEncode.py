import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from collections import Counter

#load_location = "C:\\Users\\xg16137\\PycharmProjects\\NormalizationOneHot\\Data\\"
load_location="C:\\Users\\xg16060\\OneDrive - APG\\tanguy table\\"
save_location = "C:\\Users\\xg16137\\PycharmProjects\\NormalizationOneHot\\ModifiedData\\"


dfTopLocCustEmb = pd.read_csv(load_location + "GLoc_topic_Loc_cust_embeddings.csv")
dfTopLocEmb = pd.read_csv(load_location + "GLocation_topic_Location_embeddings.csv")
dfprovinceEmb = pd.read_csv(load_location + "Gprovince_embeddings.csv")
dfTrain_balanced_rose_dz = pd.read_csv(load_location + "training_balanced_rose_dz.csv", encoding ="ISO-8859-1")
dfTrain_balanced_rose_pk = pd.read_csv(load_location + "training_balanced_rose_pk.csv", encoding ="ISO-8859-1")
dftest_dataset = pd.read_csv(load_location + "testing.csv", encoding ="ISO-8859-1")

# one hot encode with weight 1/#ofdifferentvalues
def OneHotEncoding(df):
    ohe = pd.DataFrame(index=df.index)
    for col in df:
        dummies = pd.get_dummies(df[col], prefix=col)
        ohe = pd.concat([ohe, dummies.div(dummies.shape[1])], axis=1)
    return ohe

# normalize from 0 to 1
def normalization(df):
    # print(df.drop(['Unnamed: 0'],axis=1).transpose())
    min_max_scaler = preprocessing.MinMaxScaler()
    df_normalized = pd.DataFrame(min_max_scaler.fit_transform(df))
    df_normalized.columns = df.columns
    df_normalized.index = df.index
    return df_normalized

# one-hot and normalise the data
def OneHotNormalize(df):
    df_continuous_normalized = df.select_dtypes(include=['number'])
    if not df_continuous_normalized.empty: df_continuous_normalized = normalization(df_continuous_normalized)
    df_categorical_hotencoded = df.select_dtypes(include=['object', 'category'])
    if not df_categorical_hotencoded.empty: df_categorical_hotencoded = OneHotEncoding(df_categorical_hotencoded)
    return pd.concat([df_continuous_normalized,df_categorical_hotencoded], axis=1)

def display_ROC_curve(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    #plt.show()

# display the different evaluation metrics (# get accuracy the confusion matrix and ROC of the test dataset)
def evaluatePred(y_test, y_pred):
    print('')
    print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
    print('accuracy score: ', accuracy_score(y_test, y_pred))
    # print('classification_report', classification_report(y_test, y_pred))
    print('roc auc: ', roc_auc_score(y_test, y_pred))
    # roc curve
    display_ROC_curve(y_test, y_pred)
    print('')


# features not to be normalized/OneHotEncoded
not_OneHotNormalized = ['KLTID','Target','PROVINCIE','TargetDZ','TargetPK']

# norm/OHE each datasets
# for Location embeddings
dfTopLocCust_normOneHot = OneHotNormalize(dfTopLocCustEmb.drop(['Unnamed: 0'],axis=1).transpose())
dfTopLocEmb_normOneHot = OneHotNormalize(dfTopLocEmb.drop(['Unnamed: 0'],axis=1).transpose())
dfprovinceEmb_normOneHot = OneHotNormalize(dfprovinceEmb.drop(['Unnamed: 0'],axis=1).transpose())
# for train data
dfTrain_balanced_rose_dz_normOneHot = OneHotNormalize(dfTrain_balanced_rose_dz.drop(not_OneHotNormalized, axis=1)).join(dfTrain_balanced_rose_dz[not_OneHotNormalized])
dfTrain_balanced_rose_pk_normOneHot = OneHotNormalize(dfTrain_balanced_rose_pk.drop(not_OneHotNormalized, axis=1)).join(dfTrain_balanced_rose_pk[not_OneHotNormalized])
# for test data
dftest_dataset_normOneHot = OneHotNormalize(dftest_dataset.drop(not_OneHotNormalized, axis=1)).join(dftest_dataset[not_OneHotNormalized])

##save the modified datasets
# dfTopLocCust_normOneHot.to_csv((save_location + 'GLoc_topic_Loc_cust_embeddings' + '_normOneHot.csv'))
# dfTopLocEmb_normOneHot.to_csv((save_location + 'GLocation_topic_Location_embeddings' + '_normOneHot.csv'))
# dfprovinceEmb_normOneHot.to_csv((save_location + 'Gprovince_embeddings' + '_normOneHot.csv'))
# dfTrain_balanced_rose_dz_normOneHot.to_csv((save_location + 'training_balanced_rose_dz' + '_normOneHot.csv'))
# dfTrain_balanced_rose_pk_normOneHot.to_csv((save_location + 'training_balanced_rose_pk' + '_normOneHot.csv'))
# dftest_dataset_normOneHot.to_csv((save_location + 'testing' + '_normOneHot.csv'))

# add the "provicie" feature for merging
dfTopLocCust_normOneHot['PROVINCIE'] = dfTopLocCust_normOneHot.index
dfTopLocEmb_normOneHot['PROVINCIE'] = dfTopLocEmb_normOneHot.index
dfprovinceEmb_normOneHot['PROVINCIE'] = dfprovinceEmb_normOneHot.index

# add the location embeddings to the costumer data
# for train
df_CustData_TopLocCust_included_dz = pd.merge(dfTrain_balanced_rose_dz_normOneHot, dfTopLocCust_normOneHot, how='left', on='PROVINCIE')
df_CustData_TopLocEmb_included_dz = pd.merge(dfTrain_balanced_rose_dz_normOneHot, dfTopLocEmb_normOneHot, how='left', on='PROVINCIE')
df_CustData_provinceEmb_included_dz = pd.merge(dfTrain_balanced_rose_dz_normOneHot, dfprovinceEmb_normOneHot, how='left', on='PROVINCIE')
# for test
df_Test_TopLocCust_included_dz = pd.merge(dftest_dataset_normOneHot, dfTopLocCust_normOneHot, how='left', on='PROVINCIE')
df_Test_TopLocEmb_included_dz = pd.merge(dftest_dataset_normOneHot, dfTopLocEmb_normOneHot, how='left', on='PROVINCIE')
df_Test_provinceEmb_included_dz = pd.merge(dftest_dataset_normOneHot, dfprovinceEmb_normOneHot, how='left', on='PROVINCIE')

# function to delete the customers without any known location, and delete the location string feature
def finalize(df):
    df = df[df['PROVINCIE'] != 'Onbekend'].copy()
    df.drop('PROVINCIE', axis=1, inplace=True)
    return df

# for train data
df_CustData_TopLocCust_included_dz = finalize(df_CustData_TopLocCust_included_dz)
df_CustData_TopLocEmb_included_dz = finalize(df_CustData_TopLocEmb_included_dz)
df_CustData_provinceEmb_included_dz = finalize(df_CustData_provinceEmb_included_dz)
# for test data
df_Test_TopLocCust_included_dz = finalize(df_Test_TopLocCust_included_dz)
df_Test_TopLocEmb_included_dz = finalize(df_Test_TopLocEmb_included_dz)
df_Test_provinceEmb_included_dz = finalize(df_Test_provinceEmb_included_dz)


def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

def f_importances_abs(coef, names):
    imp = abs(coef)
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    # plt.show()

## Create the KNN models now



# get X data for both PK and DZ
X_train_TopLocCust = df_CustData_TopLocCust_included_dz.drop(['KLTID', 'Target', 'TargetDZ', 'TargetPK'], axis=1)
X_train_TopLocEmb = df_CustData_TopLocEmb_included_dz.drop(['KLTID', 'Target', 'TargetDZ', 'TargetPK'], axis=1)
X_train_provinceEmb = df_CustData_provinceEmb_included_dz.drop(['KLTID', 'Target', 'TargetDZ', 'TargetPK'], axis=1)

X_test_TopLocCust = df_Test_TopLocCust_included_dz.drop(['KLTID', 'Target', 'TargetDZ', 'TargetPK'], axis=1)
X_test_TopLocEmb = df_Test_TopLocEmb_included_dz.drop(['KLTID', 'Target', 'TargetDZ', 'TargetPK'], axis=1)
X_test_provinceEmb = df_Test_provinceEmb_included_dz.drop(['KLTID', 'Target', 'TargetDZ', 'TargetPK'], axis=1)

Diff_Embeddings = [[X_train_TopLocCust, X_test_TopLocCust], [X_train_TopLocEmb, X_test_TopLocEmb], [X_train_provinceEmb, X_test_provinceEmb]]

y_train_PK = df_CustData_TopLocCust_included_dz['TargetPK']
y_test_PK = df_Test_TopLocCust_included_dz['TargetPK']

y_train_DZ = df_CustData_TopLocCust_included_dz['TargetDZ']
y_test_DZ = df_Test_TopLocCust_included_dz['TargetDZ']


# print(Diff_Embeddings[0][0].describe().transpose().drop(['min', 'max'], axis=1))
# print(Diff_Embeddings[0][1].describe().transpose().drop(['min', 'max'], axis=1))

def compare_algorithms(X_train, y_train, X_test, y_test):

    ##model: naive model 
    print('NB')
    print(y_test.value_counts()[y_train.value_counts()[:1].index.tolist()[0]]/sum(y_test.value_counts().tolist()))

    ## model: k-nearest-neighbors
    print('KNN')
    n_neighbors_values = {'n_neighbors': range(4, 6)}
    KNNclassifier_PK = KNeighborsClassifier()
    model_KNN_PK = GridSearchCV(KNNclassifier_PK, param_grid=n_neighbors_values, scoring='accuracy').fit(X_train, y_train_PK)
    y_pred_KNN = model_KNN_PK.predict(X_test)
    evaluatePred(y_test, y_pred_KNN)

    ## model: logitic regression(print features coefficients + stats)
    print('LR')
    model_LR = LogisticRegression(random_state=0, solver='saga', multi_class='ovr').fit(X_train, y_train)
    # no convergence
    y_pred_LR = model_LR.predict(X_test)
    evaluatePred(y_test, y_pred_LR)
    f_importances_abs(np.array(model_LR.coef_[0]), X_test.columns)

    ## model: decision tree (
    print('Tree')
    model_treeClassifier = tree.DecisionTreeClassifier().fit(X_train, y_train)
    y_pred_treeClassifier = model_treeClassifier.predict(X_test)
    evaluatePred(y_test, y_pred_treeClassifier)

    ##model: random forest print feature importance)
    print('RF')
    model_RF = RandomForestClassifier().fit(X_train, y_train)
    y_pred_RF = model_RF.predict(X_test)
    evaluatePred(y_test, y_pred_RF)

    ##model: xgBoost
    print('XGBoost')
    model_XGB = xgb.XGBClassifier().fit(X_train, y_train)
    y_pred_XGB = model_XGB.predict(X_test)
    evaluatePred(y_test, y_pred_XGB)

    ##model: SVM
    print('SVM')
    model_SVM = SVC(kernel='linear').fit(X_train, y_train)
    y_pred_SVM = model_SVM.predict(X_test)
    evaluatePred(y_test, y_pred_SVM)
    f_importances_abs(np.array(model_SVM.coef_[0]), X_test.columns)

for emb in Diff_Embeddings:
    compare_algorithms(emb[0], y_train_PK, emb[1], y_test_PK)

# print(df_Test_TopLocCust_included_dz)
# print(df_Test_TopLocEmb_included_dz)
# print(df_Test_provinceEmb_included_dz)

# print(df_CustData_TopLocCust_included_dz['PROVINCIE'].unique())
# print(pd.crosstab(index=df_CustData_TopLocCust_included_dz['PROVINCIE'], columns="count"))


