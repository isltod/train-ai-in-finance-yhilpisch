import os
import math
import numpy as np
import pandas as pd
from pylab import plt, mpl
import sys

sys.path.append(".")
from util import Timer

plt.style.use("seaborn-v0_8")
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "serif"
pd.set_option("mode.chained_assignment", None)
pd.set_option("display.float_format", "{:.4f}".format)
np.set_printoptions(suppress=True, precision=4)
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"

# 데이터는 유로/달러 환율, 이번에는 5분 데이터...
url = "data/aiif_eikon_id_eur_usd.csv"
symbol = "EUR="
data = pd.DataFrame(pd.read_csv(url, index_col=0, parse_dates=True).dropna()["CLOSE"])
data.columns = [symbol]
data = data.resample("5min", label="right").last().ffill()
print(data.info())
print(data.head())
data[symbol].plot(figsize=(10, 6))
plt.show()

lags = 5


# 5시간 단위 쉬프트, 20일 이동 평균/최소/최대
def add_lags(data, symbol, lags, window=20):
    cols = []
    df = data.copy()
    df.dropna(inplace=True)
    # 로그 수익률
    df["r"] = np.log(df / df.shift(1))
    # 이동 평균/최소/최대
    df["sma"] = df[symbol].rolling(window).mean()
    df["min"] = df[symbol].rolling(window).min()
    df["max"] = df[symbol].rolling(window).max()
    # 로그 수익률의 이동평균/표준편차
    df["mom"] = df["r"].rolling(window).mean()
    df["vol"] = df["r"].rolling(window).std()
    df.dropna(inplace=True)
    # 방향성
    df["d"] = np.where(df["r"] > 0, 1, 0)
    features = [symbol, "r", "d", "sma", "min", "max", "mom", "vol"]
    # 이 컬럼들 5일 쉬프트
    for f in features:
        for lag in range(1, lags + 1):
            col = f"{f}_lag_{lag}"
            df[col] = df[f].shift(lag)
            cols.append(col)
    df.dropna(inplace=True)
    return df, cols


data, cols = add_lags(data, symbol, lags, window=20)

import random
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l1
from sklearn.metrics import accuracy_score


def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seeds()

optimizer = keras.optimizers.Adam(learning_rate=0.0001)


def create_model(
    hl=2,
    hu=128,
    dropout=False,
    rate=0.3,
    regularize=False,
    reg=l1(0.0005),
    optimizer=optimizer,
    input_dim=len(cols),
):
    if not regularize:
        reg = None
    model = Sequential()
    model.add(
        Dense(hu, input_dim=input_dim, activity_regularizer=reg, activation="relu")
    )
    if dropout:
        model.add(Dropout(rate, seed=100))
    for _ in range(hl):
        model.add(Dense(hu, activation="relu", activity_regularizer=reg))
        if dropout:
            model.add(Dropout(rate, seed=100))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


split = int(len(data) * 0.85)
# 훈련 데이터
train = data.iloc[:split].copy()
print("하락/상승", np.bincount(train["d"]))


def cw(df):
    c0, c1 = np.bincount(df["d"])
    w0 = (1 / c0) * (len(df)) / 2
    w1 = (1 / c1) * (len(df)) / 2
    return {0: w0, 1: w1}


# 정규화
mu, std = train.mean(), train.std()
train_ = (train - mu) / std

set_seeds()
model = create_model(hl=1, hu=128, reg=True, dropout=False)

with Timer():
    model.fit(
        train_[cols],
        train["d"],
        epochs=40,
        verbose=False,
        # 이걸 주면 그 안에서 20%를 검증용으로 쓴단 말인가?
        validation_split=0.2,
        shuffle=False,
        class_weight=cw(train),
    )

print("훈련 데이터 결과 평가\n", model.evaluate(train_[cols], train["d"]))

# 훈련 데이터에 대해서 예측하면
train["p"] = np.where(model.predict(train_[cols]) > 0.5, 1, -1)
print("훈련 데이터 예측 결과 상승과 하락 수\n", train["p"].value_counts())
# 거래비용 고려안한 수익률
train["s"] = train["p"] * train["r"]
train[["r", "s"]].sum().apply(np.exp)
train[["r", "s"]].sum().apply(np.exp) - 1
train[["r", "s"]].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()

# 이번엔 시험 데이터
test = data.iloc[split:].copy()
# 훈련 데이터 평균/표준편차로 정규화
test_ = (test - mu) / std
print("시험 데이터 결과 평가\n", model.evaluate(test_[cols], test["d"]))
test["p"] = np.where(model.predict(test_[cols]) > 0.5, 1, -1)
print("시험 데이터 예측 결과 상승과 하락 수\n", test["p"].value_counts())
# 훈련 데이터에서 수수료 고려안한 수익률
test["s"] = test["p"] * test["r"]
test[["r", "s"]].sum().apply(np.exp)
test[["r", "s"]].sum().apply(np.exp) - 1
test[["r", "s"]].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()

# 수수료를 고려하는 경우...
print("매매 횟수", sum(test["p"].diff() != 0) + 1)
# 호가 차이인가? 이걸 평균에 곱해서 실제 거래비용이 나온다고? 이러면 정말 얼마 안되는데...
spread = 0.00012
pc_1 = spread / test[symbol]
spread = 0.00006
pc_2 = spread / test[symbol]
# 아무튼 거래비용 적용하는데...
test["s_1"] = np.where(test["p"].diff() != 0, test["s"] - pc_1, test["s"])
# 여긴 여전히 모르겠다...
test["s_1"].iloc[-1] -= pc_1.iloc[0]
test["s_2"] = np.where(test["p"].diff() != 0, test["s"] - pc_2, test["s"])
test["s_2"].iloc[-1] -= pc_2.iloc[0]
test[["r", "s", "s_1", "s_2"]].sum().apply(np.exp)
test[["r", "s", "s_1", "s_2"]].sum().apply(np.exp) - 1
test[["r", "s", "s_1", "s_2"]].cumsum().apply(np.exp).plot(
    figsize=(10, 6), style=["-", "-", "--", "--"]
)
plt.show()
