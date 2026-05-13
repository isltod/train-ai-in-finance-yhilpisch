import pandas as pd
import numpy as np
from pylab import plt, mpl
import sys

sys.path.append(".")
from util import Timer
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"

import random
import tensorflow as tf
from keras.optimizers import Adam
from keras.layers import SimpleRNN, LSTM, Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.preprocessing.sequence import TimeseriesGenerator

plt.style.use("seaborn-v0_8")
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "serif"
pd.set_option("display.precision", 4)
np.set_printoptions(suppress=True, precision=4)
os.environ["PYTHONHASHSEED"] = "0"

url = "data/aiif_eikon_id_eur_usd.csv"
symbol = "EUR_USD"
raw = pd.read_csv(url, index_col=0, parse_dates=True)


def generate_data():
    data = pd.DataFrame(raw["CLOSE"])
    data.columns = [symbol]
    # 종가 30분 자료...
    data = data.resample("30min", label="right").last().ffill()
    return data


# 일단 가격 데이터로 RNN 돌려보는데...
data = generate_data()
# 정규화하고...
data = (data - data.mean()) / data.std()
# ndarray로 바꾸고 2차원으로...근데 첫 번째 차원이 배치 차원이 아니네?
p = data[symbol].values
p = p.reshape((len(p), -1))
lags = 5
# 1~5 시간 입력과 6번째 라벨을 연결해서 반환
g = TimeseriesGenerator(p, p, length=lags, batch_size=5)


def create_rnn_model(
    hu=100, lags=lags, layer="SimpleRNN", features=1, algorithm="estimation"
):
    model = Sequential()
    if layer == "SimpleRNN":
        model.add(SimpleRNN(hu, activation="relu", input_shape=(lags, features)))
    else:
        model.add(LSTM(hu, activation="relu", input_shape=(lags, features)))
    if algorithm == "estimation":
        model.add(Dense(1, activation="linear"))
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    else:
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
    return model


# model = create_rnn_model()
# with Timer():
#     model.fit(g, epochs=500, steps_per_epoch=10, verbose=False)
# y = model.predict(g, verbose=False)
# # 예측값을 pred 컬럼에 입력...
# data["pred"] = np.nan
# data["pred"].iloc[lags:] = y.flatten()
# data[[symbol, "pred"]].plot(figsize=(10, 6), style=["b", "r-."], alpha=0.75)
# plt.show()
# data[[symbol, "pred"]].iloc[50:100].plot(
#     figsize=(10, 6), style=["b", "r-."], alpha=0.75
# )
# plt.show()
# 익히 알고있는 것처럼 그냥 오늘 데이터로 내일 예측값이라고 내놓는다...

# 그래서 입력을 로그 수익률로 바꿔보면...
data = generate_data()
data["r"] = np.log(data / data.shift(1))
data.dropna(inplace=True)
data = (data - data.mean()) / data.std()
r = data["r"].values
r = r.reshape((len(r), -1))
g = TimeseriesGenerator(r, r, length=lags, batch_size=5)
model = create_rnn_model()
with Timer():
    model.fit(g, epochs=500, steps_per_epoch=10, verbose=False)
y = model.predict(g, verbose=False)
data["pred"] = np.nan
data["pred"].iloc[lags:] = y.flatten()
data.dropna(inplace=True)
# data[["r", "pred"]].iloc[50:100].plot(figsize=(10, 6), style=["b", "r-."], alpha=0.75)
# plt.axhline(0, c="grey", ls="--")
# plt.show()
# 뭔가 의미없는 진동만 있는거 같고...방향은 잘 맞추는거 같다고 하는데...

from sklearn.metrics import accuracy_score

# 일단 훈련 데이터에 대해서는 68%
print(
    "훈련 데이터 방향 정확도", accuracy_score(np.sign(data["r"]), np.sign(data["pred"]))
)
split = int(len(r) * 0.8)
train = r[:split]
test = r[split:]
g = TimeseriesGenerator(train, train, length=lags, batch_size=5)


def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seeds()
model = create_rnn_model(hu=100)
with Timer():
    model.fit(g, epochs=500, steps_per_epoch=10, verbose=False)
g_ = TimeseriesGenerator(test, test, length=lags, batch_size=5)
y = model.predict(g_)
# 시험 데이터도 66%...좋은건가...
print("시험 데이터 방향 정확도", accuracy_score(np.sign(test[lags:]), np.sign(y)))
