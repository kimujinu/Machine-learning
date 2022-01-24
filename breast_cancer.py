# 유방암 데이터로 로지스틱 회귀(이진 분류)를 만들어 봅시다.

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier

import numpy as np
import matplotlib.pyplot as plt

class LogisticNeuron: # 로지스틱 회귀 모델 구현

    def __init__(self):
        self.w = None
        self.b = None

    def forpass(self,x):
        z = np.sum(x * self.w) + self.b # 직선 방정식을 계산
        return z

    def backprop(self,x,err):
        w_grad = x * err # 가중치에 대한 그래디언트를 계산
        b_grad = 1 * err # 절편에 대한 그래디언트를 계산
        return w_grad, b_grad

    def activation(self,z):
        z = np.clip(z,-100,None) # 안전한 np.exp() 계산을 위해
        a = 1 / (1 + np.exp(-z)) # 시그모이드 계산
        return a

    def fit(self,x,y,epochs=100):
        self.w = np.ones(x.shape[1]) # 가중치 초기화
        self.b = 0                   # 절편을 초기화
        for i in range(epochs):      # epochs 만큼 반복
            for x_i,y_i in zip(x,y): # 모든 샘플에 대해 반복한다.
                z = self.forpass(x_i) # 정방향 계산
                a = self.activation(z) # 활성화 함수 적용
                err = -(y_i - a)       # 오차 계산
                w_grad,b_grad = self.backprop(x_i,err) # 역방향 계산
                self.w -= w_grad
                self.b -= b_grad

    def predict(self,x):
        z = [self.forpass(x_i) for x_i in x] # 정방향 계산
        a = self.activation(np.array(z))     # 활성화 함수 적용
        return a > 0.5

cancer = load_breast_cancer() # 데이터 불러오기

plt.boxplot(cancer.data)
plt.xlabel('feature')
plt.ylabel('value')
plt.show()

x = cancer.data
y = cancer.target

X_train,X_test, y_train,y_test = train_test_split(x,y,stratify=y,test_size=0.2,random_state=42) # 데이터 분리

neuron = LogisticNeuron()
neuron.fit(X_train,y_train)

print(np.mean(neuron.predict(X_test) == y_test))

class SingleLayer: # 단일층 신경망

    def __init__(self):
        self.w = None
        self.b = None
        self.losses = []

    def forpass(self,x):
        z = np.sum(x * self.w) + self.b # 직선 방정식을 계산
        return z

    def backprop(self,x,err):
        w_grad = x * err    # 가중치에 대한 그래디언트 계산
        b_grad = 1 * err    # 절편에 대한 그래디언트 계산
        return w_grad,b_grad

    def activation(self,z):
        z = np.clip(z,-100,None) # 안전한 np.exp() 계산
        a = 1 / (1 + np.exp(-z)) # 시그모이드 계산
        return a

    def fit(self,x,y,epochs=100):
        self.w = np.ones(x.shape[1]) # 가중치 초기화
        self.b = 0
        for i in range(epochs):
            loss = 0
            # 인덱스 섞기
            indexes = np.random.permutation(np.arange(len(x)))
            for i in indexes:           # 모든 샘플에 대해 반복
                z = self.forpass(x[i])  # 정방향 계산
                a = self.activation(z)  # 활성화 함수 적용
                err = -(y[i] - a)       # 오차 계산
                w_grad,b_grad = self.backprop(x[i],err) # 역방향 계산
                self.w -= w_grad        # 가중치 업데이트
                self.b -= b_grad        # 절편 업데이트
                # 안전한 로그 계산을 위해 글리핑한 후 손실을 누적시킨다
                a = np.clip(a,1e-10,1-1e-10)
                loss += -(y[i] * np.log(a)+(1-y[i]) * np.log(1-a))
            # 에포크마다 평균 손실을 저장한다.
            self.losses.append(loss/len(y))

    def predict(self,x):
        z = [self.forpass(x_i) for x_i in x]
        return np.array(z) > 0

    def score(self,x,y):
        return np.mean(self.predict(x) == y)


layer = SingleLayer()
layer.fit(X_train,y_train)
print(layer.score(X_test,y_test))

plt.plot(layer.losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

sgd = SGDClassifier(loss='log',max_iter=100,tol=1e-3,random_state=42)
sgd.fit(X_train,y_train)
print(sgd.score(X_test,y_test))



