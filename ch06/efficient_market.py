import numpy as np
import pandas as pd
import warnings
from pylab import plt, mpl

# from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
import os
import sys

sys.path.append(".")
from util import Timer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential

tf.random.set_seed(100)
np.random.seed(100)
np.set_printoptions(suppress=True, precision=4)
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "serif"
pd.set_option("display.precision", 4)
plt.style.use("seaborn-v0_8")
warnings.simplefilter("ignore")

# 데이터 읽고
url = "data/aiif_eikon_eod_data.csv"
data = pd.read_csv(url, index_col=0, parse_dates=True).dropna()
# (data / data.iloc[0]).plot(figsize=(10, 6), cmap="coolwarm")
# plt.show()


# ric 컬럼 뽑아서, lags까지 뒤로 미룬 df 만들어 반환
def add_lags(data, ric, lags):
    cols = []
    df = pd.DataFrame(data[ric])
    for lag in range(1, lags + 1):
        col = "lag_{}".format(lag)
        df[col] = df[ric].shift(lag)
        cols.append(col)
    df.dropna(inplace=True)
    return df, cols


# # 7일까지 뒤로 미룬 데이터 만들고
lags = 7

# # 그냥 가격가지고 선형회귀 시켜보면 1일 차이 데이터가 계수를 거의 다 먹는다...나머진 필요없다...
# dfs = {}
# for sym in data.columns:
#     df, cols = add_lags(data, sym, lags)
#     dfs[sym] = df
# print(dfs[sym].head(7))
# regs = {}
# for sym in data.columns:
#     df = dfs[sym]
#     reg = np.linalg.lstsq(df[cols], df[sym], rcond=-1)[0]
#     regs[sym] = reg
# rega = np.stack(tuple(regs.values()))
# regd = pd.DataFrame(rega, columns=cols, index=data.columns)
# print(regd)
# # regd.mean().plot(kind="bar", figsize=(10, 6))
# # plt.show()
# #
# # 당연히 지연시킨 데이터들끼리 상관은 엄청 높다...정상성도 없다...
# print(dfs[sym].corr())
# print(adfuller(data[sym].dropna()))


# 가격가지고 로그 수익률 만들고, 그걸 지연시켜서 예측해보기...
# 근데 로그 수익률을 7일까지 이동시킨 필드가 뭔 의미일까?
rets = np.log(data / data.shift(1))
rets.dropna(inplace=True)
dfs = {}
for sym in data:
    df, cols = add_lags(rets, sym, lags)
    # 가우스 정규화...
    mu, std = df[cols].mean(), df[cols].std()
    df[cols] = (df[cols] - mu) / std
    dfs[sym] = df
# 마지막으로 처리한 sym 데이터프레임 결과...
print(dfs[sym].head())
# 이건 정상성 통과, 상관도 없고...
# print(adfuller(dfs[sym]["lag_1"]))
print(dfs[sym].corr())

# # 다중 선형 회귀 - 대강 50%
# with Timer():
#     for sym in data:
#         df = dfs[sym]
#         reg = np.linalg.lstsq(df[cols], df[sym], rcond=-1)[0]
#         pred = np.dot(df[cols], reg)
#         acc = accuracy_score(np.sign(df[sym]), np.sign(pred))
#         print(f"OLS | {sym:10s} | acc={acc:.4f}")

# # sklearn 신경망 적합 - 대충 55~60%
# with Timer():
#     for sym in data.columns:
#         df = dfs[sym]
#         model = MLPRegressor(
#             hidden_layer_sizes=[512],
#             random_state=100,
#             max_iter=1000,
#             early_stopping=True,
#             validation_fraction=0.15,
#             shuffle=False,
#         )
#         model.fit(df[cols], df[sym])
#         pred = model.predict(df[cols])
#         acc = accuracy_score(np.sign(df[sym]), np.sign(pred))
#         print(f"MLP | {sym:10s} | acc={acc:.4f}")


# Keras 신경망으로 적합 - 대략 60% 정도...
def create_model(problem="regression"):
    model = Sequential()
    model.add(Dense(512, input_dim=len(cols), activation="relu"))
    if problem == "regression":
        model.add(Dense(1, activation="linear"))
        model.compile(loss="mse", optimizer="adam")
    else:
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam")
    return model


# with Timer():
#     for sym in data.columns[:]:
#         df = dfs[sym]
#         model = create_model()
#         model.fit(df[cols], df[sym], epochs=25, verbose=False)
#         pred = model.predict(df[cols])
#         acc = accuracy_score(np.sign(df[sym]), np.sign(pred))
#         print(f"DNN | {sym:10s} | acc={acc:.4f}")

# 이번에는 데이터를 훈련과 검증으로 나눠보는데, 이러니까 다 50%정도다...
split = int(len(dfs[sym]) * 0.8)
with Timer():
    for sym in data.columns:
        df = dfs[sym]
        train = df.iloc[:split]
        reg = np.linalg.lstsq(train[cols], train[sym], rcond=-1)[0]
        test = df.iloc[split:]
        pred = np.dot(test[cols], reg)
        acc = accuracy_score(np.sign(test[sym]), np.sign(pred))
        print(f"OLS | {sym:10s} | acc={acc:.4f}")

with Timer():
    for sym in data.columns:
        df = dfs[sym]
        train = df.iloc[:split]
        model = MLPRegressor(
            hidden_layer_sizes=[512],
            random_state=100,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            shuffle=False,
        )
        model.fit(train[cols], train[sym])
        test = df.iloc[split:]
        pred = model.predict(test[cols])
        acc = accuracy_score(np.sign(test[sym]), np.sign(pred))
        print(f"MLP | {sym:10s} | acc={acc:.4f}")

with Timer():
    for sym in data.columns:
        df = dfs[sym]
        train = df.iloc[:split]
        model = create_model()
        model.fit(train[cols], train[sym], epochs=50, verbose=False)
        test = df.iloc[split:]
        pred = model.predict(test[cols])
        acc = accuracy_score(np.sign(test[sym]), np.sign(pred))
        print(f"DNN | {sym:10s} | acc={acc:.4f}")
