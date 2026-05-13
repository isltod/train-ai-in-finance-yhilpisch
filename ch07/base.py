import os
import numpy as np
import pandas as pd
from pylab import plt, mpl

plt.style.use("seaborn-v0_8")
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "serif"
pd.set_option("display.precision", 4)
np.set_printoptions(suppress=True, precision=4)
os.environ["PYTHONHASHSEED"] = "0"

url = "data/aiif_eikon_id_eur_usd.csv"
symbol = "EUR_USD"
raw = pd.read_csv(url, index_col=0, parse_dates=True)
print("원본 데이터\n", raw.head())
print("원본 데이터\n", raw.info())
data = pd.DataFrame(raw["CLOSE"].loc[:])
data.columns = [symbol]
data = data.resample("1h", label="right").last().ffill()
print("시간 단위 데이터", data.info())
# data.plot(figsize=(10, 6))
# plt.show()

lags = 5


def add_lags(data, symbol, lags, window=20):
    cols = []
    df = data.copy()
    df.dropna(inplace=True)
    # 로그 수익률
    df["r"] = np.log(df / df.shift())
    # 이동 평균, 최대, 최소
    df["sma"] = df[symbol].rolling(window).mean()
    df["min"] = df[symbol].rolling(window).min()
    df["max"] = df[symbol].rolling(window).max()
    # 수익률의 이동 평균과 분산
    df["mom"] = df["r"].rolling(window).mean()
    df["vol"] = df["r"].rolling(window).std()
    df.dropna(inplace=True)
    # 상승 하락 방향성 - 이걸 정답지로 쓸 모양...
    df["d"] = np.where(df["r"] > 0, 1, 0)
    features = [symbol, "r", "d", "sma", "min", "max", "mom", "vol"]
    # 5시간까지 쉬프트
    for f in features:
        for lag in range(1, lags + 1):
            col = f"{f}_lag_{lag}"
            df[col] = df[f].shift(lag)
            cols.append(col)
    df.dropna(inplace=True)
    return df, cols


data, cols = add_lags(data, symbol, lags)
print("데이터 수", len(data))
c = data["d"].value_counts()
print("방향성 up:1, down:0\n", c)


# 상승 738, 하락 1445를 비슷하게 만든다고...
def cw(df):
    c0, c1 = np.bincount(df["d"])
    w0 = (1 / c0) * (len(df)) / 2
    w1 = (1 / c1) * (len(df)) / 2
    return {0: w0, 1: w1}


class_weight = cw(data)
print(class_weight)
# 이렇게 곱해주면 둘 다 총 수의 평균이 된다...
print(class_weight[0] * c[0])
print(class_weight[1] * c[1])

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"

import random
import tensorflow as tf
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score


def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


optimizer = Adam(lr=0.001)


def create_model(hl=1, hu=128, optimizer=optimizer):
    model = Sequential()
    model.add(Dense(hu, input_dim=len(cols), activation="relu"))
    for _ in range(hl):
        model.add(Dense(hu, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


import sys

sys.path.append(".")
from util import Timer

set_seeds()
model = create_model(hl=1, hu=128)
with Timer():
    model.fit(data[cols], data["d"], epochs=50, verbose=False, class_weight=cw(data))

print("훈련 데이터 결과 평가\n", model.evaluate(data[cols], data["d"]))
# p는 0.5 이상으로 예측되면 상승 1, 아니면 하락 0
data["p"] = np.where(model.predict(data[cols]) > 0.5, 1, 0)
print("예측 결과 상승과 하락 수\n", data["p"].value_counts())

# 이번에는 훈련/시험 데이터 나눠서...
split = int(len(data) * 0.8)
train = data.iloc[:split].copy()
test = data.iloc[split:].copy()
set_seeds()
model = create_model(hl=1, hu=128)
with Timer():
    hist = model.fit(
        train[cols],
        train["d"],
        epochs=50,
        verbose=False,
        validation_split=0.2,
        shuffle=False,
        class_weight=cw(train),
    )

print("훈련 데이터 결과 평가\n", model.evaluate(train[cols], train["d"]))
print("시험 데이터 결과 평가\n", model.evaluate(test[cols], test["d"]))
test["p"] = np.where(model.predict(test[cols]) > 0.5, 1, 0)
print("시험 데이터 상승과 하락 예측\n", test["p"].value_counts())
res = pd.DataFrame(hist.history)
res[["accuracy", "val_accuracy"]].plot(figsize=(10, 6), style="--")
plt.show()
