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
plt.figure(figsize=(10, 6))
plt.plot(x, y, "ro", label="sample data")
for deg in [1, 5, 9, 11, 13, 15]:
    # 이것도 최소자승법 이용한 선형회귀
    reg = np.polyfit(x, y, deg=deg)
    y_ = np.polyval(reg, x)
    MSE = ((y - y_) ** 2).mean()
    print(f"deg={deg:2d} | MSE={MSE:.5f}")
    plt.plot(x, np.polyval(reg, x), label=f"deg={deg}")
plt.legend()
plt.show()

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

with Timer():
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, "ro", label="sample data")
    for _ in range(1, 8):
        model.fit(x, y, epochs=500, verbose=False)
        y_ = model.predict(x)
        MSE = ((y - y_.flatten()) ** 2).mean()
        print(f"round={_} | MSE={MSE:.5f}")
        plt.plot(x, y_, "--", label=f"round={_}")
    plt.legend()
plt.show()
