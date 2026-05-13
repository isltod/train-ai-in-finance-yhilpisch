import numpy as np
import pandas as pd
import warnings
from pylab import plt, mpl

# from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
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


def add_lags(data, ric, lags, window=50):
    cols = []
    df = pd.DataFrame(data[ric])
    df.dropna(inplace=True)
    # 로그 수익률
    df["r"] = np.log(df / df.shift())
    # 이동평균, 이동 최소/최대
    df["sma"] = df[ric].rolling(window).mean()
    df["min"] = df[ric].rolling(window).min()
    df["max"] = df[ric].rolling(window).max()
    # 로그 수익률의 평균은 모멘텀? 운동량? 어떻게 이런 개념이 되지?
    df["mom"] = df["r"].rolling(window).mean()
    # 변동성은 수익률의 이동 표준편차
    df["vol"] = df["r"].rolling(window).std()
    df.dropna(inplace=True)
    # 방향성이라...수익이면 1, 손해면 0...
    df["d"] = np.where(df["r"] > 0, 1, 0)
    # 이런 특성들을 다들 5일까지 쉬프트시킨 컬럼들을 만든다..
    # 근데 이걸 쉬프트 시킨다는게 뭔 의미인지 모르겠다...
    features = [ric, "r", "d", "sma", "min", "max", "mom", "vol"]
    for f in features:
        for lag in range(1, lags + 1):
            col = f"{f}_lag_{lag}"
            df[col] = df[f].shift(lag)
            cols.append(col)
    df.dropna(inplace=True)
    return df, cols


# def add_lags(data, ric, lags):
#     cols = []
#     df = pd.DataFrame(data[ric])
#     for lag in range(1, lags + 1):
#         col = "lag_{}".format(lag)
#         df[col] = df[ric].shift(lag)
#         cols.append(col)
#     df.dropna(inplace=True)
#     return df, cols


# 데이터 85%를 훈련에 쓰고, 시험 정확도는 그 외 데이터에서...
def train_test_model(model):
    for ric in data:
        df, cols = dfs[ric]
        split = int(len(df) * 0.85)
        train = df.iloc[:split].copy()
        # 가우시안 정규화라고...
        mu, std = train[cols].mean(), train[cols].std()
        train[cols] = (train[cols] - mu) / std
        model.fit(train[cols], train["d"])
        test = df.iloc[split:].copy()
        test[cols] = (test[cols] - mu) / std
        pred = model.predict(test[cols])
        acc = accuracy_score(test["d"], pred)
        print(f"OUT-OF-SAMPLE | {ric:7s} | acc={acc:.4f}")


url = "data/aiif_eikon_id_data.csv"
data = pd.read_csv(url, index_col=0, parse_dates=True)  # .dropna()
print(data.tail())
print(data.info())
lags = 5
dfs = {}
for ric in data:
    df, cols = add_lags(data, ric, lags)
    dfs[ric] = df, cols

# 근데 Keras 신경망은 안쓰나?
model_mlp = MLPClassifier(
    hidden_layer_sizes=[512],
    random_state=100,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.15,
    shuffle=False,
)

with Timer():
    train_test_model(model_mlp)

# 배깅 앙상블을 사용하면 조금 더 높아진다...
base_estimator = MLPClassifier(
    hidden_layer_sizes=[256],
    random_state=100,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.15,
    shuffle=False,
)
model_bag = BaggingClassifier(
    estimator=base_estimator,
    n_estimators=35,
    max_samples=0.25,
    max_features=0.5,
    bootstrap=False,
    bootstrap_features=True,
    n_jobs=8,
    random_state=100,
)

# ZeroDivisionError: Weights sum to zero, can't be normalized
# with Timer():
#     train_test_model(model_bag)
