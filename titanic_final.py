import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import titanic_tsne
import titanic_random_forest
import titanic_svm


# Importing train and test data
train = pd.read_csv(r'C:\Users\91944\Downloads\neo4j-community-3.5.4\import\Titanic\train.csv')
test = pd.read_csv(r'C:\Users\91944\Downloads\neo4j-community-3.5.4\import\Titanic\test.csv')
passengerId = test['PassengerId']

# Checking for nulls
def checkNull(c):
    m = pd.DataFrame(c.isna().sum())
    return m

# Heatmap
def heatmap(df):
    plt.figure(figsize=(20, 30))
    # calculate the correlation matrix
    corr = df.corr()
    # plot the heatmap
    sns.heatmap(corr,
                annot=True,
                xticklabels=corr.columns,
                yticklabels=corr.columns)
    plt.show()
    return df

# Data cleaning for training data
def dataCleaningForTrainingData():
    train.dropna(axis=0, subset=['Embarked'], inplace=True)  # dropping the two rows in 'Embarked' that have nulls
    print("Shape after dropping the two columns in Embarked: ", train.shape)
    train['Age'].fillna(train['Age'].mean(), inplace=True)
    print(checkNull(train))
    train['Cabin'] = train['Cabin'].str[1:]
    train['Cabin'] = train['Cabin'].fillna(0)
    train['Cabin'] = train['Cabin'].astype(bool).astype(int)
    print(checkNull(train))

# Data cleaning for test data
def dataCleaningForTestData():
    test['Fare'].fillna(test['Fare'].mean(), inplace=True)
    test['Age'].fillna(test['Age'].mean(), inplace=True)
    print(checkNull(test))
    test['Cabin'] = test['Cabin'].str[1:]
    test['Cabin'] = test['Cabin'].fillna(0)
    test['Cabin'] = test['Cabin'].astype(bool).astype(int)


# Feature engineering
def featureEngineeringForTrainingData():
    global title, train
    title = train['Name'].str.extract(r'([A-Za-z]+)\.')
    train['Title'] = title
    train['Title'].value_counts()
    train['Title'].replace(
        ['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don',
         'Dona'], ['Miss', 'Mrs', 'Miss', 'RespectMale', 'RespectMale', 'Miss', 'Miss', 'RespectMale', 'RespectMale',
                   'RespectMale', 'RespectMale', 'RespectMale', 'Mr', 'Mrs'], inplace=True)
    train['Title'].value_counts()
    train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
    train = train.drop(columns=['Name', 'Ticket', 'PassengerId'])

# Feature engineering for test data
def featureEngineeringForTestData():
    global test
    test['Title'] = test['Name'].str.extract(r'([A-Za-z]+)\.')
    test['Title'].value_counts()
    test['Title'].replace(
        ['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don',
         'Dona'], ['Miss', 'Mrs', 'Miss', 'RespectMale', 'RespectMale', 'Miss', 'Miss', 'RespectMale', 'RespectMale',
                   'RespectMale', 'RespectMale', 'RespectMale', 'Mr', 'Mrs'], inplace=True)
    test['Title'].value_counts()
    test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
    test = test.drop(columns=['Name', 'Ticket', 'PassengerId'])

# One hot encoding
def ohe(ohe_df):
    ohe_df = pd.get_dummies(ohe_df, drop_first=True)
    return ohe_df

# Standardisation
def standardisation(data):
    global train
    data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']] = StandardScaler().fit_transform(
        data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])
    print(data.head())

# Splitting data
def dataSplitting():
    global X_train, X_val, y_train, y_val, X, y
    dum1 = train.drop(columns='Survived')
    X = dum1
    y = train['Survived']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=0)

# Scores for random forest
def scores(y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return accuracy, precision, recall, f1

# Voting classifier
def votingClassifier():
    global ensemble, y_pred_ensemble
    ensemble = VotingClassifier(estimators=[('rf', titanic_random_forest.rf), ('svm', titanic_svm.svm)])
    ensemble.fit(X_train, y_train)
    y_pred_ensemble = ensemble.predict(X_val)

# PCA
def pcaForTrainingData():
    global principalDf, principalComponents
    #target = train['Survived']
    train1 = train.drop(columns=['Survived'])
    train1 = StandardScaler().fit_transform(train1)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(train1)
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['principal component 1', 'principal component 2'])
    # PCA scatter plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
    # plt.show()

def pcaForTestData():
    global pca_test_df
    pca = PCA(n_components=2)
    pca_test = pca.fit_transform(test)
    pca_test_df = pd.DataFrame(data=pca_test, columns=['principal component 1', 'principal component 2'])

# Elbow method for K means clustering
def elbowMethodForKMeansClustering():
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(principalDf)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(7, 6))
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.xticks(range(1, 11))
    plt.show()

# K means clustering
def kmeans(cluster_number):
    global kmeans1, y_means
    pcaForTrainingData()
    pcaForTestData()
    elbowMethodForKMeansClustering()

    kmeans1 = KMeans(n_clusters=cluster_number, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans1.fit_predict(principalDf)
    plt.figure(figsize=(8, 7))
    plt.scatter(principalComponents[y_kmeans == 0, 0], principalComponents[y_kmeans == 0, 1], s=90, c='red',
                label='Cluster 1')
    plt.scatter(principalComponents[y_kmeans == 1, 0], principalComponents[y_kmeans == 1, 1], s=90, c='blue',
                label='Cluster 2')
    plt.scatter(kmeans1.cluster_centers_[:, 0], kmeans1.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.legend()
    plt.show()

# Predicting
def predictingTheTargetVariable(model, test_data):
    preds = model.predict(test_data)
    print(preds)
    final = pd.DataFrame()
    final['PassengerId'] = passengerId
    final['Survived'] = preds
    print("Target values of \n%s \nare:\n " % model, final['Survived'])

def dataVisualization():
    # Shape
    print("Shape of training data: ", train.shape)
    print("Shape of test data: ", test.shape)
    # Head
    pd.options.display.max_columns = 30
    print("Train data: ", train.head())
    print("Test data: ", test.head())

    # Heatmap
    heatmap(train)

def dataPreprocessing():
    global train, test
    # Nulls
    print("Number of nulls in training data are: ", checkNull(train))
    print("Number of nulls in test data are: ", checkNull(test))
    dataCleaningForTrainingData()
    dataCleaningForTestData()
    featureEngineeringForTrainingData()
    featureEngineeringForTestData()
    train = ohe(train)
    test = ohe(test)
    print("Standardising train data: ")
    standardisation(train)
    print("Standardising test data: ")
    standardisation(test)

    #t-SNE
    y_tsne = train['Survived']
    x_tsne = train.drop(columns=['Survived'])
    titanic_tsne.t_sne(x_tsne, y_tsne)

    # K means
    kmeans(2)

    dataSplitting()

def modelling():
    # randomForest()
    titanic_random_forest.randomForest(X_train, y_train, X_val)
    print("The scores for training data, RANDOM FOREST: \n\tAccuracy, Precision, Recall, F1 score\n", scores(y_val, titanic_random_forest.y_pred_rf))
    #svm()
    titanic_svm.svm(X_train, y_train, X_val)
    print("The scores for training data, SVM: \n\tAccuracy, Precision, Recall, F1 score\n", scores(y_val, titanic_svm.y_pred_svm))
    votingClassifier()

def predictions():
    predictingTheTargetVariable(titanic_random_forest.rf, test)
    predictingTheTargetVariable(titanic_svm.svm, test)
    predictingTheTargetVariable(ensemble, test)
    predictingTheTargetVariable(kmeans1, pca_test_df)
    #titanic_kmeans.kmeansPredictions()

dataVisualization()
dataPreprocessing()
modelling()
predictions()