# 캐글 타이타닉 이진분류
# 데이터분석 전 프로세스
# 1. 데이터셋 확인 - 대부분의 캐글 데이터들은 정제된 데이터이다. 하지만 가끔 null값이 존재한다.
#                  이를 확인하고, 향후 수정한다.
# 2. 탐색적 데이터 분석(exploratory data analysis) - 여러 feature들을 개별적으로 분석하고, feature들 간의 상관관계를 확인한다.
#                                                 여러 시각화툴을 사용하여 insight를 얻는다.
# 3. feature engineering - 모델을 세우기에 앞서, 모델의 성능을 높일 수 있도록 feature들을 engineering 한다.
#                          one-hot encoding, class로 나누기, 구간으로 나누기, 텍스트 데이터 처리등을 한다.
# 4. model 만들기 - sklearn을 사용하여 모델을 만든다. 파이썬에서 머신러닝을 할 때는 sklearn을 사용하면 수많은 알고리즘을 일관된 문법으로 사용할 수 있다.
#                  물론 딥러닝을 위해 tensorflow, pytorch 등을 사용할 수 도 있다.
# 5. 모델 학습 및 예측 - trainset을 가지고 모델을 학습시킨 후, testset을 가지고 prediction을 한다.
# 6. 모델 평가 - 예측 성능이 원하는 수준인지 판단한다. 풀려는 문제에 따라 모델을 평가하는 방식도 달라진다.
#               학습된 모델이 어떤것을 학습 하였는지 확인해본다.

import numpy as np
import pandas as pd
from pandas import Series

import matplotlib.pyplot as plt # 그래프 작업을 위함
import seaborn as sns # 시각화
import missingno as msno # 결측치 제거를 위함
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics # 모델의 평가를 위함
from sklearn.model_selection import train_test_split # 학습, 테스트 데이터 분리

warnings.filterwarnings('ignore')
plt.style.use('seaborn')
sns.set(font_scale=2.5) # 스타일 지정

def category_age(x):
    if x < 10:
        return 0
    elif x < 20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    elif x < 60:
        return 5
    elif x < 70:
        return 6
    else:
        return 7

def titanic_solve():
    # 1. 데이터셋 확인
    df_train = pd.read_csv('train_titanic.csv')
    df_test = pd.read_csv('test_titanic.csv')
    # 상위 5개 항목 확인
    print(df_train.head())

    # 해당 문제에서 feature는 Pclass, Age, SibSp, Parch, Fare이며
    # 예측하려는 target label은 Survived 이다.
    print(df_train.describe())  # describe 메소드를 사용하면, 각 feature가 가진 통계치를 반환한다.
    print(df_test.describe())

    print()

    # 1-1. Null data check
    for col in df_train.columns:
        msg = 'column: {:>10}\t Percent of NaN value : {:.2f}%'.format(col, 100 * (
                df_train[col].isnull().sum() / df_train[col].shape[0]))
        print(msg)

    for col in df_test.columns:
        msg = 'column: {:>10}\t Percent of NaN value : {:.2f}%'.format(col, 100 * (
                df_test[col].isnull().sum() / df_test[col].shape[0]))
        print(msg)

    # msno.bar(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
    # msno.bar(df=df_test.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))

    # 1-2. Target label 확인
    # target label이 어떤 distribution을 가지고 있는지 확인해 보아야 한다.
    # 지금 같은 이진 분류 문제의 경우에서 1과 0의 분포가 어떠냐에 따라 모델의 평가 방법이 달라 질 수 있다.
    f, ax = plt.subplots(1, 2, figsize=(18, 8))

    df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
    ax[0].set_title('Pie plot - Survived')
    ax[0].set_ylabel('')
    sns.countplot('Survived', data=df_train, ax=ax[1])
    ax[1].set_title('Count plot - Survived')

    plt.show()
    # 그래프를 보면 38.4%만 살아남았다.
    # target label의 분포가 제법 균일하다.
    # 불균일한 경우, 예를 들어서 100중 1이 99, 0이 1개인 경우에는 만약 모델이 모든것을 1이라 해도
    # 정확도가 99%가 나오게 된다. 0을 찾는 문제라면 이 모델은 원하는 결과를 줄 수가 없다.
    # 지금 문제는 그렇지 않으니 진행한다.

    # 2. 탐색적 데이터 분석 : 이 많은 데이터 안에 숨겨진 사실을 찾기 위해선 적절한 시각화가 필요하다.
    # 2-1. Pclass
    # Pclass는 ordinal, 서수형 데이터이다. 카테고리면서, 순서가 있는 데이터 타입을 말한다.
    # 먼저 Pclass에 따른 생존률의 차이를 살펴보면, 엑셀의 피봇차트와 유사한 작업을 하게되는데
    # pandas dataframe에서는 groupby를 사용하면 쉽게 할 수 있다. 또한 pivot이라는 메소드도 있다.
    # Pclass, Survived를 가져온 후 pclass로 묶는다. 그러고 나면 각 pclass 마다 0,1이 count가 되는데,
    # 이를 평균내면 각 pclass 별 생존률이 나온다.

    print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count())
    print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum())

    # pd.crosstab(df_train['Pclass'],df_train['Survived'],margins=True).style.background_gradient(cmap='summer_r')

    df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived',
                                                                                           ascending=False).plot.bar()  # Pclass가 좋을 수록 생존률이 높은 것을 확인할 수 있다.

    # seaborn의 countplot를 이용하면, 특정 label에 따른 개수를 확인할 수 있다.
    y_position = 1.02
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32', '#FFDF00', '#D3D3D3'], ax=ax[0])
    ax[0].set_title('Number of Passengers By Pclass', y=y_position)
    ax[0].set_ylabel('Count')
    sns.countplot('Pclass', hue='Survived', data=df_train, ax=ax[1])
    ax[1].set_title('Pclass: Survived vs Dead', y=y_position)
    plt.show()

    # 그래프의 결과를 보면 클래스가 높을 수록, 생존 확률이 높다는 걸 알 수 있다.
    # 생존에 Pclass가 큰 영향을 미친다고 생각할 수 있고, 나중에 모델을 세울 때 이 feature를 사용하는 것이 좋을 것이라 판단할 수 있다.

    # 2-2. Sex : 이번에는 성별에 따른 생존률을 확인한다.
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])
    ax[0].set_title('Survived vs Sex')
    sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])
    ax[1].set_title('Sex: Survived vs Dead')
    plt.show()

    print(df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().sort_values(by='Survived',
                                                                                           ascending=False))
    # 여자의 생존확률이 높다.

    # 2-3. Both Sex and Pclass : 두가지에 관하여 생존이 어떻게 달라지는 지 확인한다.
    sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, size=6, aspect=1.5)
    # sns.factorplot(x='Sex',y='Survived',col='Pclass',data=df_train,satureation=.5,size=9,aspect=1)

    # 그래프의 결과를 보면 여자가 살 확률이 남자보다 높다. 또한 남여 상관없이 클래스가 높을 수록 살 확률이 높다.

    # 2-4. Age
    print("제일 나이 많은 탑승객 : {:.1f} Years".format(df_train['Age'].max()))
    print("제일 나이 어린 탑승객 : {:.1f} Years".format(df_train['Age'].min()))
    print("탑승객 평균 나이 : {:.1f} Years".format(df_train['Age'].mean()))

    # 생존에 따른 Age의 히스토그램 출력
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)
    sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)
    plt.legend(['Survived == 1', 'Survived == 0'])
    plt.show()

    # 그래프의 결과에 따르면, 생존자 중 나이가 어린 경우가 많다.

    plt.figure(figsize=(8, 6))
    df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')
    df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')
    df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')

    plt.xlabel('Age')
    plt.title('Age Distribution within classes')
    plt.legend(['1st class', '2nd Class', '3rd Class'])

    # Class가 높을 수록 나이 많은 사람의 비중이 커진다.

    cummulate_survival_ratio = []
    for i in range(1, 80):
        cummulate_survival_ratio.append(
            df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age'] < i]['Survived']))

    plt.figure(figsize=(7, 7))
    plt.plot(cummulate_survival_ratio)
    plt.title('Survival rate change depending on range of Age', y=1.02)
    plt.ylabel('Survival rate')
    plt.xlabel('Range of Age(0~x)')
    plt.show()

    # 그래프의 결과를 보면 나이가 어릴 수록 생존률이 확실히 높은 것을 확인할 수 있다.
    # 나이가 즁요한 feature로 쓰일 수 있음을 확인할 수 있다.

    # 2-5. Pclass, Sex, Age : 지금까지본 Sex, Pclass, Age, Survived 모두에 대해서 본다
    #                         쉽게 그려주는 것이 seaborn의 violinplot이다.
    #                         x축은 우리가 나눠서 보고싶어하는 case(Pclass, Sex)를 나타내고,
    #                         y축은 보고 싶어하는 distribution(Age)이다.
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    sns.violinplot('Pclass', 'Age', hue='Survived', data=df_train, scale='count', split=True, ax=ax[0])
    ax[0].set_title('Pclass and Age vs Survived')
    ax[0].set_yticks(range(0, 110, 10))
    sns.violinplot('Sex', 'Age', hue='Survived', data=df_train, scale='count', split=True, ax=ax[1])
    ax[1].set_title('Sex and Age vs Survived')
    ax[1].set_yticks(range(0, 110, 10))
    plt.show()

    # 그래프의 결과를 보면 여성과 아이를 먼저 챙긴 것을 볼 수 있다.

    # 2-6. Embarked : Embarked는 탑승한 항구를 말한다.
    #                 탑승한 곳에 따른 생존률을 측정한다.
    f, ax = plt.subplots(1, 1, figsize=(7, 7))
    df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived',
                                                                                               ascending=False).plot.bar(
        ax=ax)

    # 해당 특징이 모델에 얼마나 큰 영향을 미칠지 모르지만 추후에 모델을 만들고 생각을 하자.
    f, ax = plt.subplots(2, 2, figsize=(20, 15))
    sns.countplot('Embarked', data=df_train, ax=ax[0, 0])
    ax[0, 0].set_title('(1) No. Of Passengers Boarded')
    sns.countplot('Embarked', hue='Sex', data=df_train, ax=ax[0, 1])
    ax[0, 1].set_title('(2) Male-Female Split for Embarked')
    sns.countplot('Embarked', hue='Survived', data=df_train, ax=ax[1, 0])
    ax[1, 0].set_title('(3) Embarked vs Survived')
    sns.countplot('Embarked', hue='Pclass', data=df_train, ax=ax[1, 1])
    ax[1, 1].set_title('(4) Embarked vs Pclass')
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.show()

    # Figure(1) - 전체적으로 봤을 때 S에서 가장 많은 사람이 탑승했다.
    # Figure(2) - C와 Q는 남녀의 비율이 비슷하고, S는 남자가 더 많다.
    # Figure(3) - 생존확률이 S 경우 많이 낮은 걸 볼 수 있다.
    # Figure(4) - Class로 Split해서 보니, C가 생존확률이 높은건 클래스가 높은 사람이 많이타서 그렇다.

    # 2-7. Family - SibSp(형제 자매) + Parch(부모, 자녀)
    #       SibSp 와 Parch를 합하면 Family가 된다. Family로 합쳐서 분석해보자.
    df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1  # 자신을 포함해야하니 1을 더한다.
    df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1  # 자신을 포함해야하니 1을 더한다.

    print("Maximum size of Family:", df_train['FamilySize'].max())
    print("Mininum size of Family:", df_train['FamilySize'].min())

    f, ax = plt.subplots(1, 3, figsize=(40, 10))
    sns.countplot('FamilySize', data=df_train, ax=ax[0])
    ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)

    sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax[1])
    ax[1].set_title('(2) Survived countplot depending on FamilySize', y=1.02)

    df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived',
                                                                                                   ascending=False).plot.bar(
        ax=ax[2])
    ax[2].set_title('(3) Survived rate depending on FamilySize', y=1.02)

    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.show()

    # Figure(1) - 가족 크기가 1~11까지 있음을 볼수 있다. 대부분 1명이고 그다음 2,3,4 명이다.
    # Figure(2),(3) - 가족 크기에 따른 생존비교이다. 가족이 4명인 경우가 가장 생존확률이 높다.
    #                 가족수가 많아질수록, (5,6,7,8,11) 생존확률이 낮아진다.
    #                 가족수가 너무 작아도, 너무 커도 생존 확률이 낮다. 3~4명 선에서 생존확률이 높아진다.

    # 2-8. Fare : 탑승요금에 따른 생존 확률
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
    g = g.legend(loc='best')

    # 그래프의 결과의 분포가 매우 비대칭이다. 만약 이대로 모델에 넣는다면 잘못 학습할수도 있다.
    # 몇개 없는 이상치에 대해서 너무 민감하게 반응한다면, 실제 예측시에 좋지 못한 결과를 부를 수 도 있다.
    # 이상치의 영향을 줄이기 위해 Fare에 log를 취한다.
    # 여기서 판다스의 유용한 기능을 사용할 것이다. map, apply

    df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()  # test set에 있는 nan value를 평균값으로 치환한다.
    df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
    df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
    g = g.legend(loc='best')

    # log를 취하니 비대칭성이 많이 사라지는 것을 볼수 있다.
    # 해당 모델의 성능을 높이기 위해 이런식으로 특징들에 조작을 많이 한다.

    # 2-9. Cabin : 이 특징은 NaN이 대략 80% 이므로 적합한 특징이 아니다.
    # 2-10. Ticket : 이 특징은 string data이므로 사전 작업을 해야 모델에 사용이 가능하다.
    # df_train['Ticket'].value_counts()

    # 3. feature engineering
    # 가장 먼저, dataset에 존재하는 null data를 채우려고 합니다.
    # 아무 숫자로 채울 수는 없고, null data를 포함하는 feature의 statistics를 참고하거나,
    # 다른 아이디어를 짜내어 채울 수 있다.
    # null data를 어떻게 채우느냐에 따라 모델의 성능이 좌지우지될 수 있기 때문에, 신경써야 할 부분이다.
    # train 뿐만 아니라 test도 똑같이 적용해야 한다.

    # 3-1. Fill Null
    # 3-1-1. Fill Null in Age using title
    # Age에는 null data가 177개나 있다. 이를 채울 수 있는 여러 아이디어가 있을 것인데,
    # 여기서 title + statistics를 사용해 보겠다.
    # 영어에서는 Miss, Mrr, Mrs 같은 title이 존재한다. 각 탑승객의 이름에는 꼭 이런 title이 들어가게 되는데
    # 이를 사용해 보겠다.
    # pandas series에는 data를 string으로 바꿔주는 str method, 거기에 정규표현식을 적용하게 해주는
    # extract method가 있다. 이를 사용하여 title을 쉽게 추출할 수 있다.
    df_train['Initial'] = df_train.Name.str.extract('([A-Za-z]+)\.')
    df_test['Initial'] = df_test.Name.str.extract('([A-Za-z]+)\.')

    # pd.crosstab(df_train['Initial'],df_train['Sex']).T.style.background_gradient(cmap='summer_r')
    # 해당 테이블을 참고하여 남여가 쓰는 initial을 구분한다.

    df_train['Initial'].replace(
        ['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don',
         'Dona'],
        ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr', 'Mr'],
        inplace=True)

    df_test['Initial'].replace(
        ['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don',
         'Dona'],
        ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr', 'Mr'],
        inplace=True)

    # 치환한다.
    print(df_train.groupby('Initial').mean())
    # 여성과 관련있는 Miss, Mrs가 생존률이 높다.
    df_train.groupby('Initial')['Survived'].mean().plot.bar()

    # Age의 평균을 이용하여 Null value를 채우도록 한다.
    df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Mr'), 'Age'] = 33
    df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Mrs'), 'Age'] = 36
    df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Master'), 'Age'] = 5
    df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Miss'), 'Age'] = 22
    df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Other'), 'Age'] = 46

    df_test.loc[(df_test.Age.isnull()) & (df_test.Initial == 'Mr'), 'Age'] = 33
    df_test.loc[(df_test.Age.isnull()) & (df_test.Initial == 'Mrs'), 'Age'] = 36
    df_test.loc[(df_test.Age.isnull()) & (df_test.Initial == 'Master'), 'Age'] = 5
    df_test.loc[(df_test.Age.isnull()) & (df_test.Initial == 'Miss'), 'Age'] = 22
    df_test.loc[(df_test.Age.isnull()) & (df_test.Initial == 'Other'), 'Age'] = 46

    print('Embarked has ', sum(df_train['Embarked'].isnull()), ' Null values')
    df_train['Embarked'].fillna('S', inplace=True)  # Fill Null in Embarked

    # 3-2. Change Age(continuous to categorical)
    # Age는 현재 continuous feature이다. 이대로 써도 모델을 세울 수 있지만
    # Age를 몇 개의 group으로 나누어 카테고리화 시켜줄 수도 있다.
    # 하지만, continuous를 categorical로 바꾸면 자칫 information loss가 생길 수도 있다.

    df_train['Age_cat'] = df_train['Age'].apply(category_age)

    # 중복되는 Age_cat 컬럼과 원래 Age 컬럼 제거
    df_train.drop(['Age'], axis=1, inplace=True)
    df_test.drop(['Age'], axis=1, inplace=True)

    # 3-3. Change Initial, Embarked and Sex(string to numerical)
    # 현재 Initial은 Mr, Mrs, Miss, Master, Other 총 5개로 이루어져 있다.
    # 이것을 컴퓨터가 인식할 수 있게끔, 수치화 시키자.
    df_train['Initial'] = df_train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
    df_test['Initial'] = df_test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

    df_train['Embarked'].unique()
    df_train['Embarked'].value_counts()

    # Embarked 얘도 수치화
    df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    df_train['Embarked'].isnull().any()  # null 체크

    # Sex 얘도 수치화
    df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})
    df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})

    # 여러 feature를 가지고 있으니 이를 하나의 matrix 형태로 보면 편하다.
    # 이를 heatmap plot 이라고 하며, dataFrame의 corr() 메소드와 seaborn을 가지고 편하게 그릴 수 있다.

    heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Initial', 'Age_cat']]
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14, 12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})

    del heatmap_data
    # EDA에서 살펴봣듯, Sex와 Pclass가 Survived에 상관관계가 어느 정도 있음을 볼 수 있다.
    # 생각보다 fare와 Embarked 도 상관관계가 있다.
    # 또한 서로 강한 상관관계를 가지는 feature들이 없다는 것.
    # 이것은 우리가 모델을 학습시킬 때, 불필요한 feature가 없다는 것을 의미한다.

    # 3-4. One-hot encoding on Initial and Embarked
    #      수치화시킨 카테고리 데이터를 그대로 넣어도 되지만, 모델의 성능을 높이기 위해선 one-hot encoding을 해준다.
    #      수치화는 위의 코드 처럼 매핑해주는 것을 말하고
    #      one-hot encoding은 0,1로 이루어진 5차원의 벡터로 나타내는 것을 말한다.
    df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')
    df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')
    df_train.head()

    # Embarked도 적용
    df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')
    df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')

    # 3-5. Drop columns
    # 필요한 컬럼만 남기고 다 지우기.
    df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
    df_test.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

    print(df_train.head())
    print(df_test.head())

    # 4. 모델 만들기
    # 4-1. 준비 - 학습, 테스트 데이터 셋 분리
    X_train = df_train.drop('Survived', axis=1).values
    target_label = df_train['Survived'].values
    X_test = df_test.values

    X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)

    # 4-2. 모델 생성 및 예측
    model = RandomForestClassifier()
    model.fit(X_tr, y_tr)
    prediction = model.predict(X_vld)

    print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))

    # 4-3. Feature importance
    # 학습된 모델은 feature importance를 가지게 된다.
    # 이것을 확인하여 지금 만든 모델이 어떤 feature에 영향을 많이 받았는지 확인한다.

    feature_importance = model.feature_importances_
    Series_feat_imp = Series(feature_importance, index=df_test.columns)

    plt.figure(figsize=(8, 8))
    Series_feat_imp.sort_values(ascending=True).plot.barh()
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.show()
    # Fare가 가장 큰 영향력을 가진다.

    # 4-4. 테스트 데이터 예측
    submission = pd.read_csv('gender_submission.csv')
    submission.head()

    prediction = model.predict(X_test)
    submission['Survived'] = prediction

    submission.to_csv('./first_submission.csv', index=False)

class exec:
    def __init__(self):
        titanic_solve()

exec()
