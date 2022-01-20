import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from collections import Counter

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier,VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,learning_curve

sns.set(font_scale=1.5) # 스타일 지정
warnings.filterwarnings('ignore')

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# 이상치 탐지
def detect_outliers(df, n, features):
    outlier_indices = []
    # iterate over features(columns)
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers

def solve():

    # 1. 데이터 읽어오기
    df_train = pd.read_csv('train_titanic.csv')
    df_test = pd.read_csv('test_titanic.csv')
    IDtest = df_test['PassengerId']

    Outliers_to_drop = detect_outliers(df_train, 2, ["Age", "SibSp", "Parch", "Fare"])

    print(df_train.loc[Outliers_to_drop]) # 이상치 탐지 결과

    df_train = df_train.drop(Outliers_to_drop,axis=0).reset_index(drop=True) # 이상치 컬럼에서 제거

    # 테스트 데이터, 학습 데이터 합치기
    train_len = len(df_train)
    dataset = pd.concat(objs=[df_train,df_test], axis=0).reset_index(drop=True)

    # null, 빈값 체크
    dataset = dataset.fillna(np.nan)
    print(dataset.isnull().sum())

    df_train.info()
    print(df_train.isnull().sum())

    print(df_train.head())
    print(df_train.dtypes)
    print(df_train.describe())

    # 2. Feature 분석

    # 히트맵으로 수치화
    g = sns.heatmap(df_train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True,fmt=".2f",cmap="coolwarm")

    # SibSp 분석
    g = sns.factorplot(x="SibSp",y='Survived',data=df_train,kind='bar',size=6,palette='muted')
    g.despine(left=True)
    g = g.set_ylabels("survival probability")

    # Parch 분석
    g = sns.factorplot(x="Parch", y='Survived', data=df_train, kind='bar', size=6, palette='muted')
    g.despine(left=True)
    g = g.set_ylabels("survival probability")

    # Age 분석, 데이터의 특성에 따라 다르게 그래프를 시각화해야함
    g = sns.FacetGrid(df_train,col='Survived')
    g = g.map(sns.distplot,"Age")

    g = sns.kdeplot(df_train["Age"][(df_train["Survived"] == 0) & (df_train["Age"].notnull())], color="Red", shade=True)
    g = sns.kdeplot(df_train["Age"][(df_train["Survived"] == 1) & (df_train["Age"].notnull())], ax=g, color="Blue", shade=True)
    g.set_xlabel("Age")
    g.set_ylabel("Frequency")
    g = g.legend(["Not Survived", "Survived"])

    dataset["Fare"].isnull().sum()
    dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())

    g = sns.distplot(dataset["Fare"], color="m", label="Skewness : %.2f" % (dataset["Fare"].skew()))
    g = g.legend(loc="best")

    # 이상치의 영향을 줄이기 위해 Fare에 log를 취한다.
    dataset["Fare"] = dataset["Fare"].map(lambda i : np.log(i) if i > 0 else 0)

    g = sns.distplot(dataset["Fare"],color="b",label="SKewness : %.2f"%(dataset["Fare"].skew()))
    g = g.legend(loc="best")

    # 카테고리화

    # Sex
    g = sns.barplot(x="Sex",y="Survived",data=df_train)
    g = g.set_ylabel("Survival Probability")

    print(df_train[['Sex','Survived']].groupby('Sex').mean())

    # Pclass
    g = sns.factorplot(x='Pclass',y='Survived',data=df_train,kind='bar',size=6,palette='muted')
    g.despine(left=True)
    g = g.set_ylabels('Survival Probability')

    g = sns.factorplot(x='Pclass',y='Survived',hue='Sex',data=df_train,size=6,kind='bar',palette='muted')
    g.despine(left=True)
    g = g.set_ylabels('Survival Probability')

    # Embarked
    dataset['Embarked'].isnull().sum()
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    g = sns.factorplot(x='Embarked',y='Survived',data=df_train,size=6,kind='bar',palette='muted')
    g.despine(left=True)
    g = g.set_ylabels('Survival Probability')

    # Embarked 와 Pclass 비교
    g = sns.factorplot('Pclass',col='Embarked',data=df_train,size=6,kind='count',palette='muted')
    g.despine(left=True)
    g = g.set_ylabels('Count')

    # 빈값 채우기
    g = sns.factorplot(y='Age',x='Sex',data=dataset,kind='box')
    g = sns.factorplot(y='Age', x='Sex',hue='Pclass', data=dataset, kind='box')
    g = sns.factorplot(y='Age', x='Parch', data=dataset, kind='box')
    g = sns.factorplot(y='Age', x='SibSp', data=dataset, kind='box')

    dataset['Sex'] = dataset['Sex'].map({'male':0,'female':1})
    g = sns.heatmap(dataset[['Age','Sex','SibSp','Parch','Pclass']].corr(),cmap='BrBG',annot=True)

    index_NaN_age = list(dataset['Age'][dataset['Age'].isnull()].index)

    for i in index_NaN_age:
        age_med = dataset['Age'].median()
        age_pred = dataset['Age'][((dataset['SibSp'] == dataset.iloc[i]['SibSp']) & (dataset['Parch'] == dataset.iloc[i]['Parch']) & (dataset['Pclass'] == dataset.iloc[i]['Pclass']))].median()
        if not np.isnan(age_pred):
            dataset['Age'].iloc[i] = age_pred
        else:
            dataset['Age'].iloc[i] = age_med

    g = sns.factorplot(x='Survived',y='Age',data=df_train,kind='box')
    g = sns.factorplot(x='Survived',y='Age',data=df_train,kind='violin')

    print(dataset['Name'].head())

    dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
    dataset['Title'] = pd.Series(dataset_title)
    dataset['Title'].head()

    g = sns.countplot(x='Title',data=dataset)
    g = plt.setp(g.get_xticklabels(),rotation=45)

    # 제목 카테고리화
    dataset["Title"] = dataset["Title"].replace(
        ['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
        'Rare')
    dataset["Title"] = dataset["Title"].map(
        {"Master": 0, "Miss": 1, "Ms": 1, "Mme": 1, "Mlle": 1, "Mrs": 1, "Mr": 2, "Rare": 3})

    dataset['Title'] = dataset['Title'].astype(int)

    g = sns.countplot(dataset['Title'])
    g = g.set_xticklabels(['Master','Miss/Ms/Mme/Mlle/Mrs','Mr','Rare'])

    g = sns.factorplot(x='Title',y='Survived',data=dataset,kind='bar')
    g = g.set_xticklabels(['Master','Miss-Mrs','Mr','Rare'])
    g = g.set_ylabels('Survival probability')

    # Name 컬럼 삭제
    dataset.drop(labels=['Name'],axis=1,inplace=True)

    # famliy 컬럼 생성 SibSp + Parch
    dataset['Famliy'] = dataset['SibSp'] + dataset['Parch'] + 1

    g = sns.factorplot(x='Famliy',y='Survived',data=dataset)
    g = g.set_ylabels('Survival Probability')

    # famliy 컬럼에서 특징 추출해서 새로운 컬럼 추출
    dataset['Single'] = dataset['Famliy'].map(lambda s: 1 if s == 1 else 0)
    dataset['SmallF'] = dataset['Famliy'].map(lambda s: 1 if s == 2 else 0)
    dataset['MedF'] = dataset['Famliy'].map(lambda s: 1 if 3<= s <= 4 else 0)
    dataset['LargeF'] = dataset['Famliy'].map(lambda s: 1 if s >= 5 else 0)

    g = sns.factorplot(x='Single',y='Survived',data=dataset,kind='bar')
    g = g.set_ylabels('Survival Probability')
    g = sns.factorplot(x='SmallF',y='Survived',data=dataset,kind='bar')
    g = g.set_ylabels('Survival Probability')
    g = sns.factorplot(x='MedF',y='Survived',data=dataset,kind='bar')
    g = g.set_ylabels('Survival Probability')
    g = sns.factorplot(x='LargeF',y='Survived',data=dataset,kind='bar')
    g = g.set_ylabels('Survival Probability')

    # 더미 데이터 생성
    dataset = pd.get_dummies(dataset,columns=['Title'])
    dataset = pd.get_dummies(dataset,columns=['Embarked'],prefix='Em')

    print(dataset.head())

    print(dataset['Cabin'].head())
    print(dataset['Cabin'].describe())
    print(dataset['Cabin'].isnull().sum())

    print(dataset['Cabin'][dataset['Cabin'].notnull()].head())

    dataset['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin']])
    g = sns.countplot(dataset['Cabin'],order=['A','B','C','D','E','F','G','T','X'])

    g = sns.factorplot(y='Survived',x='Cabin',data=dataset,kind='bar',order=['A','B','C','D','E','F','G','T','X'])
    g = g.set_ylabels('Survival Probability')

    dataset = pd.get_dummies(dataset,columns=['Cabin'],prefix='Cabin')

    print(dataset['Ticket'].head())

    Ticket = []
    for i in list(dataset.Ticket):
        if not i.isdigit():
            Ticket.append(i.replace('.','').replace('/','').strip().split(' ')[0])
        else:
            Ticket.append('X')

    dataset['Ticket'] = Ticket
    print(dataset['Ticket'].head())

    dataset = pd.get_dummies(dataset,columns=['Ticket'],prefix='T')
    dataset['Pclass'] = dataset['Pclass'].astype('category')
    dataset = pd.get_dummies(dataset,columns=['Pclass'],prefix='Pc')

    dataset.drop(labels=['PassengerId'], axis=1,inplace=True)

    print(dataset.head())

    # 모델링
    train = dataset[:train_len]
    test = dataset[train_len:]
    test.drop(labels=['Survived'],axis=1,inplace=True)

    # 학습 데이터 분리
    train['Survived'] = train['Survived'].astype(int)
    y_train = train['Survived']
    X_train = train.drop(labels=['Survived'],axis=1)

    kfold = StratifiedKFold(n_splits=10)

    # 각 머신러닝 알고리즘별 학습
    random_state = 2
    classifiers = []
    classifiers.append(SVC(random_state=random_state))
    classifiers.append(DecisionTreeClassifier(random_state=random_state))
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    classifiers.append(ExtraTreesClassifier(random_state=random_state))
    classifiers.append(GradientBoostingClassifier(random_state=random_state))
    classifiers.append(MLPClassifier(random_state=random_state))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(LogisticRegression(random_state=random_state))
    classifiers.append(LinearDiscriminantAnalysis())

    cv_results = []
    for classifier in classifiers:
        cv_results.append(cross_val_score(classifier, X_train, y=y_train, scoring="accuracy", cv=kfold, n_jobs=4))

    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    cv_res = pd.DataFrame(
        {"CrossValMeans": cv_means, "CrossValerrors": cv_std, "Algorithm": ["SVC", "DecisionTree", "AdaBoost",
                                                                            "RandomForest", "ExtraTrees",
                                                                            "GradientBoosting",
                                                                            "MultipleLayerPerceptron", "KNeighboors",
                                                                            "LogisticRegression",
                                                                            "LinearDiscriminantAnalysis"]})

    g = sns.barplot("CrossValMeans", "Algorithm", data=cv_res, palette="Set3", orient="h", **{'xerr': cv_std})
    g.set_xlabel("Mean Accuracy")
    g = g.set_title("Cross validation scores")

    ##############################################
    # Adaboost
    DTC = DecisionTreeClassifier()

    adaDTC = AdaBoostClassifier(DTC,random_state=7)

    ada_param_grid = {"base_estimator__criterion": ["gini", "entropy"],
                      "base_estimator__splitter": ["best", "random"],
                      "algorithm": ["SAMME", "SAMME.R"],
                      "n_estimators": [1, 2],
                      "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}

    gsadaDTC = GridSearchCV(adaDTC,param_grid=ada_param_grid,cv=kfold,scoring='accuracy',n_jobs=4,verbose=1)
    gsadaDTC.fit(X_train,y_train)

    ada_best = gsadaDTC.best_estimator_

    print(gsadaDTC.best_score_)

    ##############################################
    ExtC = ExtraTreesClassifier()

    ex_param_grid = {"max_depth": [None],
                     "max_features": [1, 3, 10],
                     "min_samples_split": [2, 3, 10],
                     "min_samples_leaf": [1, 3, 10],
                     "bootstrap": [False],
                     "n_estimators": [100, 300],
                     "criterion": ["gini"]}

    gsExtC = GridSearchCV(ExtC,param_grid=ex_param_grid,cv=kfold,scoring='accuracy',n_jobs=4,verbose=1)
    gsExtC.fit(X_train,y_train)

    ExtC_best = gsExtC.best_estimator_
    print(gsExtC.best_score_)

    ##############################################
    RFC = RandomForestClassifier()

    rf_param_grid = {"max_depth": [None],
                     "max_features": [1, 3, 10],
                     "min_samples_split": [2, 3, 10],
                     "min_samples_leaf": [1, 3, 10],
                     "bootstrap": [False],
                     "n_estimators": [100, 300],
                     "criterion": ["gini"]}

    gsRFC = GridSearchCV(RFC,param_grid=rf_param_grid,cv=kfold,scoring='accuracy',n_jobs=4,verbose=1)

    gsRFC.fit(X_train,y_train)

    RFC_best = gsRFC.best_estimator_

    print(gsRFC.best_score_)

    ##############################################
    GBC = GradientBoostingClassifier()
    gb_param_grid = {'loss': ["deviance"],
                     'n_estimators': [100, 200, 300],
                     'learning_rate': [0.1, 0.05, 0.01],
                     'max_depth': [4, 8],
                     'min_samples_leaf': [100, 150],
                     'max_features': [0.3, 0.1]
                     }

    gsGBC = GridSearchCV(GBC, param_grid=gb_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsGBC.fit(X_train, y_train)

    GBC_best = gsGBC.best_estimator_

    print(gsGBC.best_score_)
    ##############################################

    SVMC = SVC(probability=True)
    svc_param_grid = {'kernel': ['rbf'],
                      'gamma': [0.001, 0.01, 0.1, 1],
                      'C': [1, 10, 50, 100, 200, 300, 1000]}
    gsSVMC = GridSearchCV(SVMC,param_grid=svc_param_grid,cv=kfold,scoring='accuracy',n_jobs=4,verbose=1)
    gsSVMC.fit(X_train,y_train)

    SVMC_best = gsSVMC.best_estimator_

    print(gsSVMC.best_score_)

    g = plot_learning_curve(gsRFC.best_estimator_, "RF mearning curves", X_train, y_train, cv=kfold)
    g = plot_learning_curve(gsExtC.best_estimator_, "ExtraTrees learning curves", X_train, y_train, cv=kfold)
    g = plot_learning_curve(gsSVMC.best_estimator_, "SVC learning curves", X_train, y_train, cv=kfold)
    g = plot_learning_curve(gsadaDTC.best_estimator_, "AdaBoost learning curves", X_train, y_train, cv=kfold)
    g = plot_learning_curve(gsGBC.best_estimator_, "GradientBoosting learning curves", X_train, y_train, cv=kfold)

    fig,ax = plt.subplots(2,2,sharex='all',figsize=(15,15))

    names_classifiers = [("AdaBoosting", ada_best), ("ExtraTrees", ExtC_best), ("RandomForest", RFC_best),("GradientBoosting", GBC_best)]

    nclassifier = 0
    for row in range(2):
        for col in range(2):
            name = names_classifiers[nclassifier][0]
            classifier = names_classifiers[nclassifier][1]
            indices = np.argsort(classifier.feature_importances_)[::-1][:40]
            g = sns.barplot(y=X_train.columns[indices][:40], x=classifier.feature_importances_[indices][:40],
                            orient='h', ax=ax[row][col])
            g.set_xlabel("Relative importance", fontsize=12)
            g.set_ylabel("Features", fontsize=12)
            g.tick_params(labelsize=9)
            g.set_title(name + " feature importance")
            nclassifier += 1

    test_Survived_RFC = pd.Series(RFC_best.predict(test),name='RFC')
    test_Survived_ExtC = pd.Series(ExtC_best.predict(test),name='ExtC')
    test_Survived_SVMC = pd.Series(SVMC_best.predict(test),name='SVMC')
    test_Survived_AdaC = pd.Series(ada_best.predict(test),name='ada')
    test_Survived_GBC = pd.Series(GBC_best.predict(test),name='GBC')

    ensemble_results = pd.concat([test_Survived_RFC, test_Survived_ExtC, test_Survived_AdaC, test_Survived_GBC, test_Survived_SVMC], axis=1)
    g = sns.heatmap(ensemble_results.corr(), annot=True)

    # 앙상블 모델링
    votingC = VotingClassifier(estimators=[('rfc',RFC_best),('extc',ExtC_best),('svc',SVMC_best),('adac',ada_best),('gbc',GBC_best)],voting='soft',n_jobs=4)

    votingC = votingC.fit(X_train,y_train)

    # 예측
    test_Survived = pd.Series(votingC.predict(test),name='Survived')
    results = pd.concat([IDtest,test_Survived],axis=1)
    results.to_csv("ensemble_python_voting.csv",index=False)

class exec:
    def __init__(self):
        solve()

exec()