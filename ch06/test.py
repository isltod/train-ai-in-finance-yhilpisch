import numpy as np
import pandas as pd
from pylab import plt, mpl

plt.style.use("seaborn-v0_8")
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "serif"
pd.set_option("display.precision", 4)
np.set_printoptions(suppress=True, precision=4)
import warnings

warnings.simplefilter("ignore")
url = "http://hilpisch.com/aiif_eikon_eod_data.csv"
data = pd.read_csv(url, index_col=0, parse_dates=True).dropna()
(data / data.iloc[0]).plot(figsize=(10, 6), cmap="coolwarm")
lags = 7


def add_lags(data, ric, lags):
    cols = []
    df = pd.DataFrame(data[ric])
    for lag in range(1, lags + 1):
        col = "lag_{}".format(lag)
        df[col] = df[ric].shift(lag)
        cols.append(col)
    df.dropna(inplace=True)
    return df, cols


dfs = {}
for sym in data.columns:
    df, cols = add_lags(data, sym, lags)
    dfs[sym] = df
dfs[sym].head(7)
regs = {}
for sym in data.columns:
    df = dfs[sym]
    reg = np.linalg.lstsq(df[cols], df[sym], rcond=-1)[0]
    regs[sym] = reg
rega = np.stack(tuple(regs.values()))
regd = pd.DataFrame(rega, columns=cols, index=data.columns)
regd
regd.mean().plot(kind="bar", figsize=(10, 6))
dfs[sym].corr()
rets = np.log(data / data.shift(1))
rets.dropna(inplace=True)
dfs = {}
for sym in data:
    df, cols = add_lags(rets, sym, lags)
    mu, std = df[cols].mean(), df[cols].std()
    df[cols] = (df[cols] - mu) / std
    dfs[sym] = df
dfs[sym].head()
dfs[sym].corr()
from sklearn.metrics import accuracy_score

for sym in data:
    df = dfs[sym]
    reg = np.linalg.lstsq(df[cols], df[sym], rcond=-1)[0]
    pred = np.dot(df[cols], reg)
    acc = accuracy_score(np.sign(df[sym]), np.sign(pred))
    print(f"OLS | {sym:10s} | acc={acc:.4f}")
from sklearn.neural_network import MLPRegressor

for sym in data.columns:
    df = dfs[sym]
    model = MLPRegressor(
        hidden_layer_sizes=[512],
        random_state=100,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        shuffle=False,
    )
    model.fit(df[cols], df[sym])
    pred = model.predict(df[cols])
    acc = accuracy_score(np.sign(df[sym]), np.sign(pred))
    print(f"MLP | {sym:10s} | acc={acc:.4f}")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "6"
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

np.random.seed(100)
tf.random.set_seed(100)


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


for sym in data.columns[:]:
    df = dfs[sym]
    model = create_model()
    model.fit(df[cols], df[sym], epochs=25, verbose=False)
    pred = model.predict(df[cols])
    acc = accuracy_score(np.sign(df[sym]), np.sign(pred))
    print(f"DNN | {sym:10s} | acc={acc:.4f}")
