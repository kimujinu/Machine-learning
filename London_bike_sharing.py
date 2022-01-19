import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # 그래프 작업을 위함
import seaborn as sns # 시각화
import missingno as msno # 결측치를 시각화

from sklearn.model_selection import train_test_split # 훈련용 데이터, 테스트 데이터 분리

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping # 학습중 과적합 잡아주는 라이브러리

from sklearn.metrics import mean_squared_error # mse (실제값과 예측값의 차이를 구해주는 라이브러리)
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def RMSE(y_test,y_predict): # mse의 루트를 씌운모양(평균 제곱근 편차)
    return np.sqrt(mean_squared_error(y_test,y_predict))

def is_outliners(s):# 아웃라이어 제거(시그마 이상치 제거) 모든 데이터에는 잡음이 존재함 그것을 제거하기 위한 함수
    lower_limit = s.mean() - (s.std()*3) # 양 극단 값을 이상치로 추정하여 제거
    upper_limit = s.mean() + (s.std()*3)
    return ~s.between(lower_limit,upper_limit)

def plot_bar(data,feature): # 그래프 함수 만들기
    fig = plt.figure(figsize=(12,3))
    sns.barplot(x=feature,y='cnt',data=data, palette='Set3', orient='v')

def analysis():
    df = pd.read_csv('london_merged.csv', parse_dates=['timestamp'])  # 데이터 불러오기, 시간 데이터 불러오기
    df.head()  # df에 데이터 상위 5개 불러오기

    # 데이터의 타입과 구조
    print('데이터의 구조는:', df.shape)  # 행 17414 열 10
    print('데이터의 타입은:', df.dtypes)  # 각 열별로 데이터의 타입불러오기
    print('데이터의 컬럼은:', df.columns)

    # 데이터의 결측치
    df.isna().sum()

    msno.matrix(df)
    plt.show()

    df['year'] = df['timestamp'].dt.year  # df에 year를 하나 새롭게 만들어주기
    df['month'] = df['timestamp'].dt.month
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['hour'] = df['timestamp'].dt.hour
    df.head()

    # 탐색적 분석
    print(df['year'].value_counts())
    print()
    print(df['weather_code'].value_counts())  # 코드에 따른 의미가 있음.

    # a,b = plt.subplots(1,1,figsize=(10,5)) # 그래프 그리기
    # sns.boxplot(df['year'],df['cnt']) # 가로축, 세로축 년도별 이용횟수 (디버그 형태로 실행해야 보임)

    # a, b = plt.subplots(1, 1, figsize=(10, 5))  # 그래프 그리기
    # sns.boxplot(df['month'], df['cnt'])  # 가로축, 세로축 월별 이용횟수 (디버그 형태로 실행해야 보임)

    # a, b = plt.subplots(1, 1, figsize=(10, 5))  # 그래프 그리기
    # sns.boxplot(df['dayofweek'], df['cnt'])  # 가로축, 세로축 요일별 이용횟수 (디버그 형태로 실행해야 보임)

    #plot_bar(df,'hour') # 시간에 따른 그래프 그리기
    df_out = df[~df.groupby('hour')['cnt'].apply(is_outliners)]

    print("이상치 제거전:",df.shape)
    print("이상치 제거후:",df_out.shape)

    print(df_out.dtypes)
    # 딥러닝, 머신러닝으로 학습을 하려면 기계가 학습할수 있도록 범주형 데이터여야 한다.
    df_out['weather_code'] = df_out['weather_code'].astype('category')
    df_out['season'] = df_out['season'].astype('category')
    df_out['year'] = df_out['year'].astype('category')
    df_out['month'] = df_out['month'].astype('category')
    df_out['hour'] = df_out['hour'].astype('category')

    print()
    # 범주형 데이터 변경 확인
    print(df_out.dtypes)

    # 머신러닝은 크게 영향이 없지만 딥러닝은 더미처리를 해야한다.
    # 더미처리란 컴퓨터가 학습이 용이하도록 0, 1과 같은 이진의 숫자로 바꾸어주는 것

    df_out = pd.get_dummies(df_out,columns=['weather_code','season','year','month','hour'])
    print(df_out.head())

    df_y = df_out['cnt'] # 종속 변수
    df_x = df_out.drop(['timestamp','cnt'],axis=1) # 독립 변수

    print(df_x.head())
    print(df_y.head())

    # 훈련 데이터, 테스트 데이터 분리
    X_train,X_test,y_train,y_test = train_test_split(df_x,df_y,random_state=60,test_size=0.3,shuffle=False)

    print("X_train의 구조는:",X_train.shape) # 데이터 확인
    print("y_train의 구조는:",y_train.shape) # 데이터 확인
    print("X_test의 구조는:",X_test.shape) # 데이터 확인
    print("y_test의 구조는:",y_test.shape) # 데이터 확인

    model = Sequential() # 딥러닝 계층형 학습법
    model.add(Dense(units=160,activation='relu',input_dim=57))
    model.add(Dense(units=60,activation='relu'))
    model.add(Dense(units=20, activation='relu'))
    model.add(Dense(units=10, activation='linear'))

    print(model.summary()) # 모델 Dense 정보

    model.compile(loss='mae',optimizer='adam',metrics=['mae'])
    early_stop = EarlyStopping(monitor='loss',patience=5,mode='min')# 과적합이 될때 loss가 5번이 올라가면 멈춰야함
    history = model.fit(X_train,y_train,epochs=5,batch_size=1,validation_split=0.1,callbacks=[early_stop]) # 학습한 모형 넣기
    # validation_split은 검증용 데이터 훈련이 잘되고 있는지 확인하기 위함, 과적합 방지

    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title("loss")
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.legend(['val_loss','loss'])
    plt.show()

    y_predict = model.predict(X_test)
    # 머신러닝 각 알고리즘 별 성능 비교

    rf = RandomForestRegressor(n_estimators=100,random_state=16)
    rf.fit(X_train,y_train)
    rf_result = rf.predict(X_test)
    print("RMSE:", RMSE(y_test, rf_result))  # 랜덤 포레스트를 거친후 RMSE

    xgb = XGBRegressor(n_estimators=100,random_state=16)
    xgb.fit(X_train,y_train)
    xgb_result = xgb.predict(X_test)
    print("RMSE:", RMSE(y_test, xgb_result))  # XGBRegressor를 거친후 RMSE

    lgb = LGBMRegressor(n_estimators=100,random_state=16)
    lgb.fit(X_train,y_train)
    lgb_result = lgb.predict(X_test)
    print("RMSE:", RMSE(y_test, lgb_result))  # LGBMRegressor를 거친후 RMSE

    xgb = pd.DataFrame(xgb_result)
    rf = pd.DataFrame(rf_result)
    lgb = pd.DataFrame(lgb_result)
    compare = pd.DataFrame(y_test).reset_index(drop=True)

    compare['xgb'] = xgb
    compare['rf'] = rf
    compare['lgb'] = lgb
    print(compare.head())

    sns.kdeplot(compare['cnt'],shade=True,color='r')
    sns.kdeplot(compare['xgb'],shade=True,color='b')
    sns.kdeplot(compare['rf'],shade=True,color='y')
    sns.kdeplot(compare['lgb'],shade=True,color='g')


class exec:
    def __init__(self):
        analysis()

exec()
