# 검증 세트 나누기, 데이터 전처리

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler # 표준화 적용

import numpy as np
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
X_train_all,X_test,y_train_all,y_test = train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)

sgd = SGDClassifier(loss='log',random_state=42) # 확률적 경사하강법 적용 로지스틱 회귀
sgd.fit(X_train_all,y_train_all)
sgd.score(X_test,y_test)

sgd = SGDClassifier(loss='hinge',random_state=42) # 확률적 경사하강법 적용 SVM
sgd.fit(X_train_all,y_train_all)
sgd.score(X_test,y_test)

X_train,X_val,y_train,y_val = train_test_split(X_train_all,y_train_all,stratify=y_train_all,test_size=0.2,random_state=42) # 검증 데이터 분리

sgd = SGDClassifier(loss='log',random_state=42)
sgd.fit(X_train,y_train)
sgd.score(X_val,y_val)

class SingleLayer:

    def __init__(self,learning_rate=0.1,l1=0,l2=0):
        self.w = None
        self.b = None
        self.losses = []
        self.val_losses = []
        self.w_history = []
        self.lr = learning_rate
        self.l1 = l1
        self.l2 = l2

    def forpass(self,x):
        z = np.sum(x * self.w) + self.b # 직선 방정식 계산
        return z

    def backprop(self,x,err):
        w_grad = x * err    # 가중치에 대한 그래디언트 계산
        b_grad = 1 * err    # 절편에 대한 그래디언트 계산
        return w_grad,b_grad

    def activation(self,z):
        z = np.clip(z,-100,None) # 안전한 np.exp() 계산을 위해
        a = 1 / (1 + np.exp(-z)) # 시그모이드 계산
        return a

    def fit(self,x,y,epochs=100,x_val=None,y_val=None):
        self.w = np.ones(x.shape[1])    # 가중치 초기화
        self.b = 0                      # 절편 초기화
        self.w_history.append(self.w.copy())    # 가중치 기록
        np.random.seed(42)              # 랜덤 시드 지정
        for i in range(epochs):         # epochs 만큼 반복
            loss = 0
            indexes = np.random.permutation(np.arange(len(x)))
            for i in indexes:           # 모든 샘플에 대해 반복
                z = self.forpass(x[i])  # 정방향 계산
                a = self.activation(z)  # 활성화 함수 적용
                err = -(y[i] - a)       # 오차 계산
                w_grad, b_grad = self.backprop(x[i],err) # 역방향 계산
                # 그래디언트에서 페널티 항의 미분 값을 더한다.
                w_grad += self.l1 * np.sign(self.w) + self.l2 * self.w
                self.w -= self.lr * w_grad  # 가중치 업데이트
                self.b -= self.lr * b_grad  # 절편 업데이트
                # 가중치 기록
                self.w_history.append(self.w.copy())
                # 안전한 로그 계산을 위해 글리핑한 후 손실을 누적시킨다
                a = np.clip(a,1e-10,1-1e-10)
                loss += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
            # 에포크마다 평균 손실을 저장한다.
            self.losses.append(loss/len(y) + self.reg_loss())
            self.update_val_loss(x_val,y_val) # 검증 세트에 대한 손실 계산

    def predict(self,x):
        z = [self.forpass(x_i) for x_i in x] # 정방향 계산
        return np.array(z) >= 0              # 스텝 함수 적용

    def score(self,x,y):
        return np.mean(self.predict(x) == y)

    def reg_loss(self):
        return self.l1 * np.sum(np.abs(self.w)) + self.l2 / 2 * np.sum(self.w**2)

    def update_val_loss(self,x_val,y_val):
        if x_val is None:
            return
        val_loss = 0
        for i in range(len(x_val)):
            z = self.forpass(x_val[i]) # 정방향 계산
            a = self.activation(z)     # 활성화 함수 적용
            a = np.clip(a,1e-10,1-1e-10)
            val_loss += -(y_val[i]*np.log(a)+(1-y_val[i])*np.log(1-a))
        self.val_losses.append(val_loss/len(y_val) + self.reg_loss())

layer1 = SingleLayer()
layer1.fit(X_train,y_train)
print(layer1.score(X_val,y_val))

# 가중치 기록하고 학습률 적용하기
# 학습률이 너무 높으면 가중치의 변화가 크므로 전역 최솟값을 지나칠 수도 있다.
w2 = []
w3 = []
for w in layer1.w_history:
    w2.append(w[2])
    w3.append(w[3])
plt.plot(w2,w3)
plt.plot(w2[-1],w3[-1],'ro')
plt.xlabel('w[2]')
plt.ylabel('w[3]')
plt.show()

# 스케일을 조정하여 모델을 훈련한다.
# 표준화(standardization) : 평균 0, 분산 1
train_mean = np.mean(X_train,axis=0) # 평균 계산
train_std = np.std(X_train,axis=0) # 표준 편차 계산
X_train_scaled = (X_train - train_mean) / train_std

layer2 = SingleLayer()
layer2.fit(X_train_scaled,y_train)
w2 = []
w3 = []
for w in layer2.w_history:
    w2.append(w[2])
    w3.append(w[3])
plt.plot(w2,w3)
plt.plot(w2[-1],w3[-1],'ro')
plt.xlabel('w[2]')
plt.ylabel('w[3]')
plt.show()

# 검증 세트로 모델 성능 평가하기
# 검증 세트를 사용하는 이유: 테스트 데이터를 사용하지 않고 하이퍼파라미터 값을 찾기 위해
print(layer2.score(X_val,y_val))

val_mean = np.mean(X_val,axis=0)
val_std = np.std(X_val,axis=0)
x_val_scaled = (X_val - val_mean) / val_std

print(layer2.score(x_val_scaled,y_val))

# 올바르게 검증 세트 전처리하기
x_val_scaled = (X_val - train_mean) / train_std
plt.plot(X_train_scaled[:50,0],X_train_scaled[:50,1],'bo')
plt.plot(x_val_scaled[:50,0],x_val_scaled[:50,1],'ro')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.legend(['train set','val set'])
plt.show()

print(layer2.score(x_val_scaled,y_val))

# 과대적합, 과소적합
layer3 = SingleLayer()
layer3.fit(X_train_scaled,y_train,x_val=x_val_scaled,y_val=y_val)

plt.ylim(0,0.3)
plt.plot(layer3.losses)
plt.plot(layer3.val_losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'])
plt.show()

layer4 = SingleLayer()
layer4.fit(X_train_scaled,y_train,epochs=20) # 조기 종료
layer4.score(x_val_scaled,y_val)

# 규제 방법 : 모델의 학습 적합도를 조절하기 위해.
# 라쏘(Lasso) = L1 규제 + 선형 회귀, 그래디언트에서 alpha에 가중치의 부호를 곱하여 그레디언트에 더한다.
# w_grad += alpha * np.sign(w)
# 릿지(Ridge)(조금 더 선호) = L2 규제 + 선형 회귀, 그래디언트에서 alpha에 가중치를 곱하여 그레디언트에 더한다.
# w_grad += alpha * w
# 엘라스틱 넷(ElasticNet) = L1/L2 + 선형 회귀
l1_list = [0.0001, 0.001,0.01]

for l1 in l1_list:
    lyr = SingleLayer(l1=l1)
    lyr.fit(X_train_scaled,y_train,x_val=x_val_scaled,y_val=y_val)

    plt.plot(lyr.losses)
    plt.plot(lyr.val_losses)
    plt.title('Learning Curve (l1={})'.format(l1))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss','val_loss'])
    plt.ylim(0,0.3)
    plt.show()

    plt.plot(lyr.w,'bo')
    plt.title('Weight (l1={})'.format(l1))
    plt.ylabel('value')
    plt.xlabel('weight')
    plt.ylim(-4,4)
    plt.show()

layer5 = SingleLayer(l1=0.001)
layer5.fit(X_train_scaled,y_train,epochs=20)
print(layer5.score(x_val_scaled,y_val))

l2_list = [0.0001,0.001,0.01]

for l2 in l2_list:
    lyr = SingleLayer(l2=l2)
    lyr.fit(X_train_scaled,y_train,x_val=x_val_scaled,y_val=y_val)

    plt.plot(lyr.losses)
    plt.plot(lyr.val_losses)
    plt.title('Learning Curve (l2={})'.format(l2))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss','val_loss'])
    plt.ylim(0,0.3)
    plt.show()

    plt.plot(lyr.w,'bo')
    plt.title('Weight (l2={})'.format(l2))
    plt.ylabel('value')
    plt.xlabel('weight')
    plt.ylim(-4,4)
    plt.show()

layer6 = SingleLayer(l2=0.01)
layer6.fit(X_train_scaled,y_train,epochs=50)
print(layer6.score(x_val_scaled,y_val))

np.sum(layer6.predict(x_val_scaled)==y_val)

sgd = SGDClassifier(loss='log',penalty='l2',alpha=0.001,random_state=42)
sgd.fit(X_train_scaled,y_train)
print(sgd.score(x_val_scaled,y_val))

# 교차 검증 후, 사이킷런을 수행한다.
# 교차 검증 과정
# 1. 훈련 세트를 k개의 폴드(fold)로 나눈다.
# 2. 첫 번째 폴드를 검증 세트로 사용하고 나머지 폴드 (k-1개)를 훈련 세트로 사용한다.
# 3. 모델을 훈련한 다음에 검증 세트로 평가한다
# 4. 차례대로 다음 폴드를 검증 세트로 사용하여 반복한다.
# 5. k개의 검증 세트로 k번 성능을 평가한 후 계산된 성능의 평균을 내어 최종 성능을 계산한다.
validation_scores = []

k = 10
bins = len(X_train_all) // k

for i in range(k):
    start = i*bins
    end = (i+1)*bins
    val_fold = X_train_all[start:end]
    val_target = y_train_all[start:end]

    train_index = list(range(0,start))+list(range(end,len(X_train_all)))
    train_fold = X_train_all[train_index]
    train_target = y_train_all[train_index]

    train_mean = np.mean(train_fold,axis=0)
    train_std = np.std(train_fold,axis=0)
    train_fold_scaled = (train_fold - train_mean) / train_std
    val_fold_scaled = (val_fold - train_mean) / train_std

    lyr = SingleLayer(l2=0.01)
    lyr.fit(train_fold_scaled,train_target,epochs=50)
    score = lyr.score(val_fold_scaled,val_target)
    validation_scores.append(score)

print(np.mean(validation_scores))

sgd = SGDClassifier(loss='log',penalty='l2',alpha=0.001,random_state=42)
scores = cross_validate(sgd,X_train_all,y_train_all,cv=10)
print(np.mean(scores['test_score']))

print(type(scores)) # dict

# 전처리 단계를 포함해 교차 검증을 수행한다.
# 검증 세트를 나누지 않고 훈련 세트 전체를 사용한다.
# 전처리와 모델을 하나의 파이프라인으로 정의한다.
pipe = make_pipeline(StandardScaler(),sgd)
scores = cross_validate(pipe,X_train_all,y_train_all,cv=10,return_train_score=True)
print(np.mean(scores['test_score']))

print(np.mean(scores['train_score']))
