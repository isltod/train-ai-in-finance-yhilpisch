import numpy as np
import pandas as pd
from pylab import plt, mpl
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 환경 설정들
plt.style.use("seaborn-v0_8")
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "serif"
np.set_printoptions(precision=4, suppress=True)

# 비지도 학습 - K-Means 클러스터링
# 가우시안 정규분포로 클러스터링 가상 데이터 생성, random_state는 seed 같은 거 같네..
# x는 x1, x2, y는 그룹
x, y = make_blobs(n_samples=100, centers=4, random_state=500, cluster_std=1.25)
model = KMeans(n_clusters=4, random_state=0, n_init="auto")
model.fit(x)
y_ = model.predict(x)
print(y_)
# plt.figure(figsize=(10, 6))
# plt.scatter(x[:, 0], x[:, 1], c=y_, cmap="coolwarm")
# plt.show()

# 강화학습
# 이게 그냥 무작위로 찍을 때이고...
# 이게 상태 공간이고
ssp = [1, 1, 1, 1, 0]
# 이건 행동 공간인 모양...
asp = [1, 0]


def epoch():
    tr = 0
    for _ in range(100):
        a = np.random.choice(asp)
        s = np.random.choice(ssp)
        if a == s:
            tr += 1
    return tr


rl = np.array([epoch() for _ in range(15)])
print(rl)
print(rl.mean())


# 이게 학습이 되는 거라는데...
def epoch():
    tr = 0
    asp = [0, 1]
    for _ in range(100):
        a = np.random.choice(asp)
        s = np.random.choice(ssp)
        if a == s:
            tr += 1
            # 여기서 학습이 된다는 거 같은데...논리상 if 문으로 들어가야 할 거 같은데...
            # 그래야 성공하면 그게 더해져서 성공했던 숫자를 뽑을 확률이 높아지는거 아닌가?
            asp.append(s)
    return tr


rl = np.array([epoch() for _ in range(15)])
print(rl)
print(rl.mean())

# 회귀
from sklearn.neural_network import MLPRegressor


def f(x):
    return 2 * x**2 - x**3 / 3


x = np.linspace(-2, 4, 25)
y = f(x)
print("x:", x.shape)
print("y:", y.shape)

model = MLPRegressor(
    hidden_layer_sizes=3 * [256], learning_rate_init=0.03, max_iter=5000
)
model.fit(x.reshape(-1, 1), y)
y_ = model.predict(x.reshape(-1, 1))
MSE = ((y - y_) ** 2).mean()
print(MSE)
# plt.figure(figsize=(10, 6))
# plt.plot(x, y, "ro", label="sample data")
# plt.plot(x, y_, lw=3.0, label="dnn estimation")
# plt.legend()
# plt.show()

# Keras 신경망 회귀
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.random.set_seed(100)
from keras.layers import Dense
from keras.models import Sequential

print(tf.config.list_physical_devices("GPU"))

# model = Sequential()
# # (?,256) 출력
# model.add(Dense(256, activation="relu", input_dim=1))
# model.add(Dense(1, activation="linear"))
# model.compile(loss="mse", optimizer="rmsprop")

# plt.figure(figsize=(10, 6))
# plt.plot(x, y, "ro", label="sample data")
# for _ in range(1, 6):
#     model.fit(x, y, epochs=100, verbose=False)
#     y_ = model.predict(x)
#     MSE = ((y - y_.flatten()) ** 2).mean()
#     # for 문에서 _ 도 i 처럼 써먹을 수가 있네...
#     print(f"round={_} | MSE={MSE:.5f}")
# 이렇게 하면 iter마다 예측 결과를 비교해서 그래프로 그릴 수가 있구나..
#     plt.plot(x, y_, "--", label=f"round={_}")
# plt.legend()
# plt.show()

# 보편적 근사? Universal Approximation
np.random.seed(0)
x = np.linspace(-1, 1)
y = np.random.random(len(x)) * 2 - 1
# plt.figure(figsize=(10, 6))
# plt.plot(x, y, "ro", label="sample data")
# for deg in [1, 5, 9, 11, 13, 15]:
#     # 이것도 최소자승법 이용한 선형회귀
#     reg = np.polyfit(x, y, deg=deg)
#     y_ = np.polyval(reg, x)
#     MSE = ((y - y_) ** 2).mean()
#     print(f"deg={deg:2d} | MSE={MSE:.5f}")
#     plt.plot(x, np.polyval(reg, x), label=f"deg={deg}")
# plt.legend()
# plt.show()

model = Sequential()
model.add(Dense(256, activation="relu", input_dim=1))
for _ in range(3):
    model.add(Dense(256, activation="relu"))
model.add(Dense(1, activation="linear"))
model.compile(loss="mse", optimizer="rmsprop")
print(model.summary())
import sys

sys.path.append(".")
from util import Timer

# with Timer():
#     plt.figure(figsize=(10, 6))
#     plt.plot(x, y, "ro", label="sample data")
#     for _ in range(1, 8):
#         model.fit(x, y, epochs=500, verbose=False)
#         y_ = model.predict(x)
#         MSE = ((y - y_.flatten()) ** 2).mean()
#         print(f"round={_} | MSE={MSE:.5f}")
#         plt.plot(x, y_, "--", label=f"round={_}")
#     plt.legend()
# plt.show()

# 제목이 빅 데이터의 중요성이라고...
# 이건 너무 적은데...그리고 난리인 랜덤 데이터를 학습하니, 되도 뭐가 뭔지...
f = 5
n = 10
np.random.seed(100)
# 0/1 중에 하나를 (10,5)
x = np.random.randint(0, 2, (n, f))
y = np.random.randint(0, 2, n)
# model = Sequential()
# model.add(Dense(256, activation="relu", input_dim=f))
# model.add(Dense(1, activation="sigmoid"))
# model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["acc"])
# hist = model.fit(x, y, epochs=50, verbose=False)
# y_ = np.where(model.predict(x).flatten() > 0.5, 1, 0)
# res = pd.DataFrame(hist.history)
# res.plot(figsize=(10, 6))
# plt.show()

# 아무튼 데이터를 좀 늘리지만...
f = 10
n = 250
np.random.seed(100)
x = np.random.randint(0, 2, (n, f))
y = np.random.randint(0, 2, n)
# 각 패턴 별로 데이터가 다 있을 수가 없다...
print("나타날 수 있는 x의 패턴 수:", 2**f)
# f0~9로 컬럼 만들어서 x를 데이터프레임으로...컬럼 10개 레코드 250개...
fcols = [f"f{_}" for _ in range(f)]
print(fcols)
data = pd.DataFrame(x, columns=fcols)
# 라벨 컬럼이겠지...
data["l"] = y
print(data.info())
grouped = data.groupby(list(data.columns))
# 마지막 "l" 컬럼과 나머지 컬럼들과 피봇 테이블 만들기...
freq = grouped["l"].size().unstack(fill_value=0)
# l 컬럼이 0인 경우와 1인 경우를 모두 더하고...
freq["sum"] = freq[0] + freq[1]
print(freq.head(10))
print(freq["sum"].describe().astype(int))

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

model = MLPClassifier(
    hidden_layer_sizes=[128, 128, 128], max_iter=1000, random_state=100
)
# 이런 적은 데이터를 신경망으로 학습시키면...95% 이상 잘 맞춘다고 하지만...
model.fit(data[fcols], data["l"])
print(accuracy_score(data["l"], model.predict(data[fcols])))
# 훈련 데이터와 테스트 데이터를 나누면...
split = int(len(data) * 0.7)
train = data[:split]
test = data[split:]
model.fit(train[fcols], train["l"])
# 테스트 데이터에 대한 정확도는 38%...찍어도 반인데 그것도 안된다...
# 가만...이게 내가 트레이딩에서 실패하는 이윤가?
print("훈련 정확도:", accuracy_score(train["l"], model.predict(train[fcols])))
print("테스트 정확도:", accuracy_score(test["l"], model.predict(test[fcols])))

# 50배 더 많은 데이터를 만들면
factor = 50
big = pd.DataFrame(np.random.randint(0, 2, (factor * n, f)), columns=fcols)
big["l"] = np.random.randint(0, 2, factor * n)
train = big[:split]
test = big[split:]
model.fit(train[fcols], train["l"])
# 그냥 똑 같이 해도, 별 규칙 없는 데이터인데도 테스트 정확도가 50%로 올라갔다...
# 근데 랜덤 데이터니까 테스트 정확도는 50%가 한계가 아닐까? 그런거 같다...
print("훈련 정확도:", accuracy_score(train["l"], model.predict(train[fcols])))
print("테스트 정확도:", accuracy_score(test["l"], model.predict(test[fcols])))
grouped = big.groupby(list(data.columns))
freq = grouped["l"].size().unstack(fill_value=0)
freq["sum"] = freq[0] + freq[1]
print(freq.head(10))
print(freq["sum"].describe().astype(int))
