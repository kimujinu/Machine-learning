import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # 그래프 작업을 위함
import seaborn as sns # 시각화
import missingno as msno # 결측치를 시각화

from statsmodels.tsa.stattools import pacf # 시계열 모형에서 차수를 추정하는 라이브러리
from statsmodels.graphics.tsaplots import plot_pacf

from sklearn.preprocessing import MinMaxScaler # 데이터 스케일 작업
from sklearn.metrics import mean_squared_error # MSE

from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM,GRU


# 시계열 딥러닝 : 일반 딥러닝과 다름,
#               데이터의 구조, 독립변수가 없음, 데이터 자체만의 패턴을 학습해야함


df = pd.read_csv('london_merged.csv',parse_dates=['timestamp'])
print(df.head())

print(df['timestamp']) # 데이터 타입확인

print(df.shape)

train = df.iloc[:17000,1:2]
test = df.iloc[17000:17414,1:2] # 414개만 테스트 데이터 사용

print(train.shape)
print(test.shape)

# 데이터 구조 시각화
df['cnt'][:17000].plot(figsize=(15,4),legend=True)
df['cnt'][17000:].plot(figsize=(15,4),legend=True)
plt.legend(['train','test'])
plt.title('bike share demand')
plt.show()

# 데이터 전처리
pacf = pacf(df['cnt'],nlags=20,method='ols')
print(pacf)

plot_pacf(pacf,lags=9,method='ols',title='pa') # 시계열 차수 추정 그래프

# 데이터 스케일 작업(정규화 작업을 위해, 연산속도 증가)
sc = MinMaxScaler(feature_range=(0,1)) # 데이터를 0~1사이 값으로 정규화 시킨다.
train_scaled = sc.fit_transform(train)

print(train_scaled)

# 시계열 딥러닝은 자기 자신의 과거를 독립변수로 활용을 한다. 1시간 전 데이터를 독립변수로 사용해야 한다.
X_train = []
y_train = []
for i in range(1,17000):
    X_train.append(train_scaled[i-1:i,0])
    y_train.append(train_scaled[i,0])

X_train,y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1)) # 3차원화
print(X_train.shape)

# 시계열 딥러닝 (RNN) : 과거의 입력값을 네트워크에 남겨서 출력에 영향을 미치게하는 기법
rnn = Sequential()
rnn.add(SimpleRNN(activation='relu',units=6, input_shape =(1,1)))
rnn.add(Dense(activation='linear',units=1))

print(rnn.summary()) # 모델 확인

rnn.compile(loss='mse',optimizer='adam',metrics=['mse'])
rnn.fit(X_train,y_train,batch_size=1,epochs=2) # X는 1시간전 데이터, y는 원래 데이터

inputs = sc.transform(test) # 테스트 데이터도 스케일작업
print(inputs.shape)

X_test = []
for i in range(1,415):
    X_test.append(inputs[i-1:i,0])

X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
print(X_test.shape)

rnn = rnn.predict(X_test)
rnn = sc.inverse_transform(rnn) # 스케일 작업으로 인해 바뀐 값들을 원래 값으로 되돌리기

test1 = pd.DataFrame(test)
rnn1 = pd.DataFrame(rnn)

test1.plot(figsize=(15,4),legend=True)
plt.legend(['cnt'])
plt.title('bike share demand')
plt.show()

rnn1.plot(figsize=(15,4),legend=True)
plt.legend(['rnn'])
plt.title('bike share demand')
plt.show()

test = np.array(test)

plt.figure(figsize=(15,5))
plt.plot(test,marker='.',label='cnt',color='black')
plt.plot(rnn,marker=',',label='RNN',color='red')
plt.legend()

# RNN은 시간의 흐름에 따라 입력된 정보의 값이 소멸되는 vanish gradient 문제가 발생할수 있다.
# 이러한 문제를 해결하기 위해 장기적으로 기억할 수 있도록 메모리블록을 도입하여 LSTM이 탄생하였다. RNN + 메모리 블록 = LSTM

# LSTM
lstm = Sequential()
lstm.add(LSTM(units=6,activation='relu',input_shape=(1,1)))
lstm.add(Dense(units=1,activation='linear'))

print(lstm.summary()) # RNN에 비해 파라미터가 많기 때문에, 과적합문제가 있을수도 있다.

lstm.compile(loss='mse',optimizer='adam',metrics=['mse'])
lstm.fit(X_train,y_train,batch_size=1,epochs=2)

lstm = lstm.predict(X_test)
lstm = sc.inverse_transform(lstm)

plt.figure(figsize=(15,5))
plt.plot(test,marker='.',label='cnt',color='black')
plt.plot(lstm,marker=',',label='LSTM',color='green')
plt.legend()

# LSTM의 과적합 문제를 완화시킨 버전 GRU
# GRU

gru = Sequential()
gru.add(GRU(units=6,activation='relu',input_shape=(1,1)))
gru.add(Dense(units=1, activation='linear'))

print(gru.summary())

gru.compile(loss='mse',optimizer='adam',metrics=['mse'])
gru.fit(X_train,y_train,batch_size=1,epochs=2)

gru = gru.predict(X_test)
gru = sc.inverse_transform(gru)

plt.figure(figsize=(15,5))
plt.plot(test, marker='.',label='cnt',color='black')
plt.plot(gru, marker=',',label='GRU',color='blue')
plt.legend()

# 모형별 비교
plt.figure(figsize=(15,5))
plt.plot(test, marker='.',label='cnt',color='black')
plt.plot(rnn,marker=',',label='RNN',color='red')
plt.plot(lstm,marker=',',label='LSTM',color='green')
plt.plot(gru, marker=',',label='GRU',color='blue')
plt.legend()

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print("RNN RMSE:",RMSE(test,rnn))
print("LSTM RMSE:",RMSE(test,lstm))
print("GRU RMSE:",RMSE(test,gru))

