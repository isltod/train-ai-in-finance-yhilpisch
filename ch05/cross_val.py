import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from pylab import plt, mpl
import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential

sys.path.append(".")
from util import Timer

tf.random.set_seed(100)
np.random.seed(100)
plt.style.use("seaborn-v0_8")
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "serif"


def PolynomialRegression(degree=None, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


url = "data/aiif_eikon_eod_data.csv"
raw = pd.read_csv(url, index_col=0, parse_dates=True)["EUR="]
# 월말 값으로 리샘플링
l = raw.resample("1ME").last()
# y값은 평균을 제거해주고
l = l.values
l -= l.mean()
# 별 의미없는 x축을 만들고...
f = np.linspace(-2, 2, len(l))

np.set_printoptions(suppress=True, formatter={"float": lambda x: f"{x:12.2f}"})
print("\nCross-validation scores")
print(74 * "=")
for deg in range(0, 10, 1):
    model = PolynomialRegression(deg)
    cvs = cross_val_score(model, f.reshape(-1, 1), l, cv=5)
    print(f"deg={deg} | " + str(cvs.round(2)))
np.random.seed(100)
tf.random.set_seed(100)
from scikeras.wrappers import KerasRegressor


def create_dnn_model(hl=1, hu=256):
    """Function to create Keras DNN model.
    Parameters
    ==========
    hl: int
        히든 층의 수
    hu: int
        출력 노드 수 (per layer)
    """
    model = Sequential()
    for _ in range(hl):
        # 입력 차원은 계속 1인가...입력 노드는 데이터보고 정한다치면, 입력 차원은 왜 지정해줘야 하지?
        model.add(Dense(hu, activation="relu", input_dim=1))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="rmsprop")
    return model


# 이건 다 에런데....뭘 고쳐야 할지 모르겠다...
model = KerasRegressor(model=create_dnn_model, verbose=False, epochs=1000, hl=1, hu=36)
with Timer():
    cross_val_score(model, f.reshape(-1, 1), l, cv=5)
model = KerasRegressor(model=create_dnn_model, verbose=False, epochs=1000, hl=3, hu=256)
with Timer():
    cross_val_score(model, f.reshape(-1, 1), l, cv=5)
