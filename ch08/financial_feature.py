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

lags = 5
g = TimeseriesGenerator(train.values, train["r"].values, length=lags, batch_size=5)


def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seeds()


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


model = create_rnn_model(hu=100, features=len(data.columns), layer="SimpleRNN")
with Timer():
    model.fit(g, epochs=100, steps_per_epoch=10, verbose=False)
g_ = TimeseriesGenerator(test.values, test["r"].values, length=lags, batch_size=5)
y = model.predict(g_).flatten()
from sklearn.metrics import accuracy_score

print(
    "시험 데이터 방향 정확도",
    accuracy_score(np.sign(test["r"].iloc[lags:]), np.sign(y)),
)
