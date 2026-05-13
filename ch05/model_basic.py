import numpy as np
import pandas as pd
from pylab import plt, mpl
import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "6"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential

sys.path.append(".")
from util import Timer

np.random.seed(100)
plt.style.use("seaborn-v0_8")
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "serif"

url = "data/aiif_eikon_eod_data.csv"
raw = pd.read_csv(url, index_col=0, parse_dates=True)["EUR="]
# 월말 값으로 리샘플링
l = raw.resample("1ME").last()
print(l.tail())
# l.plot(figsize=(10, 6), title="EUR/USD monthly")
# plt.show()

# y값은 평균을 제거해주고
l = l.values
l -= l.mean()
# 별 의미없는 x축을 만들고...
f = np.linspace(-2, 2, len(l))
# plt.figure(figsize=(10, 6))
# plt.plot(f, l, "ro")
# plt.title("Sample Data Set")
# plt.xlabel("features")
# plt.ylabel("labels")
# plt.show()


# 성능 척도는 평균 제곱 오차
def MSE(l, p):
    return np.mean((l - p) ** 2)


# 5차 선형 회귀
reg = np.polyfit(f, l, deg=5)
print("5차 선형 회귀 계수", reg)
p = np.polyval(reg, f)
print("5차 선형 회귀 MSE", MSE(l, p))
# plt.figure(figsize=(10, 6))
# plt.plot(f, l, "ro", label="sample data")
# plt.plot(f, p, "--", label="regression")
# plt.legend()
# plt.show()

with Timer():
    for i in range(10, len(f) + 1, 20):
        # 처음부터 데이터 수를 늘려가면서 3차 회귀를 해보면...
        reg = np.polyfit(f[:i], l[:i], deg=3)
        # 예측은 전체에 대해서, 평균 제곱 오차도 전체에 대해서...
        p = np.polyval(reg, f)
        mse = MSE(l, p)
        # 평균 제곱 오차가 점점 줄어든다...
        print(f"{i:3d} | MSE={mse}")


tf.random.set_seed(100)
# 모델은 히든이 1개 층이다...
model = Sequential()
model.add(Dense(256, activation="relu", input_dim=1))
model.add(Dense(1, activation="linear"))
model.compile(loss="mse", optimizer="rmsprop")
model.summary()

# 일단 나는 책보다는 더 걸리는데...
with Timer():
    hist = model.fit(f, l, epochs=1500, verbose=False)
# 배치 차원을 없애는 모양...
p = model.predict(f).flatten()
# MSE는 낮게 나온다..
print("DNN MSE", MSE(l, p))
# plt.figure(figsize=(10, 6))
# plt.plot(f, l, "ro", label="sample data")
# plt.plot(f, p, "--", label="DNN approximation")
# plt.legend()
# plt.show()
#
import pandas as pd

# 평균 제곱 오차가 hist.history에 저장되어 있는 모양...
res = pd.DataFrame(hist.history)
res.tail()
res.iloc[100:].plot(figsize=(10, 6))
plt.ylabel("MSE")
plt.xlabel("epochs")
plt.show()
