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
from sklearn.metrics import accuracy_score

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


data = generate_data()
data["r"] = np.log(data / data.shift(1))
window = 20
data["mom"] = data["r"].rolling(window).mean()
data["vol"] = data["r"].rolling(window).std()
data.dropna(inplace=True)
split = int(len(data) * 0.8)
train = data.iloc[:split].copy()
mu, std = train.mean(), train.std()
train = (train - mu) / std
test = data.iloc[split:].copy()
test = (test - mu) / std


def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seeds()
lags = 5

from keras.layers import Dropout


def create_deep_rnn_model(
    hl=2,
    hu=100,
    layer="SimpleRNN",
    optimizer="rmsprop",
    features=1,
    dropout=False,
    rate=0.3,
    seed=100,
):
    if hl <= 2:
        hl = 2
    if layer == "SimpleRNN":
        layer = SimpleRNN
    else:
        layer = LSTM
    model = Sequential()
    model.add(
        layer(
            hu,
            input_shape=(lags, features),
            return_sequences=True,
        )
    )
    if dropout:
        model.add(Dropout(rate, seed=seed))
    for _ in range(2, hl):
        model.add(layer(hu, return_sequences=True))
        if dropout:
            model.add(Dropout(rate, seed=seed))
    model.add(layer(hu))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


model = create_deep_rnn_model(
    hl=2, hu=50, layer="SimpleRNN", features=len(data.columns), dropout=True, rate=0.3
)
print(model.summary())


def cw(a):
    c0, c1 = np.bincount(a)
    w0 = (1 / c0) * (len(a)) / 2
    w1 = (1 / c1) * (len(a)) / 2
    return {0: w0, 1: w1}


train_y = np.where(train["r"] > 0, 1, 0)
g = TimeseriesGenerator(train.values, train_y, length=lags, batch_size=5)
with Timer():
    model.fit(
        g, epochs=200, steps_per_epoch=10, verbose=False, class_weight=cw(train_y)
    )

test_y = np.where(test["r"] > 0, 1, 0)
g_ = TimeseriesGenerator(test.values, test_y, length=lags, batch_size=5)

y = np.where(model.predict(g_, batch_size=None) > 0.5, 1, 0).flatten()
print(np.bincount(y))
print(accuracy_score(test_y[lags:], y))
