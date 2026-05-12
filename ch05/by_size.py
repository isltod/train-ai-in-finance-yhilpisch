import numpy as np
import pandas as pd
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


# 성능 척도는 평균 제곱 오차
def MSE(l, p):
    return np.mean((l - p) ** 2)


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


url = "data/aiif_eikon_eod_data.csv"
raw = pd.read_csv(url, index_col=0, parse_dates=True)["EUR="]
# 월말 값으로 리샘플링
l = raw.resample("1ME").last()
# y값은 평균을 제거해주고
l = l.values
l -= l.mean()
# 별 의미없는 x축을 만들고...
f = np.linspace(-2, 2, len(l))

reg = {}
# 2차씩 증가하면서 회귀식 만들어 MSE 측정
for d in range(1, 12, 2):
    reg[d] = np.polyfit(f, l, deg=d)
    p = np.polyval(reg[d], f)
    mse = MSE(l, p)
    print(f"{d:2d} | MSE={mse}")

# MSE는 0.0012까지 줄어든다..
plt.figure(figsize=(10, 6))
plt.plot(f, l, "ro", label="sample data")
for d in reg:
    p = np.polyval(reg[d], f)
    plt.plot(f, p, "--", label=f"deg={d}")
plt.legend()
plt.show()


model = create_dnn_model(3)
model.summary()
with Timer():
    hist = model.fit(f, l, epochs=2500, verbose=False)
p = model.predict(f).flatten()
# 이건 MSE가 0.00035까지 줄어든다...
print("DNN MSE", MSE(l, p))
plt.figure(figsize=(10, 6))
plt.plot(f, l, "r", label="sample data")
plt.plot(f, p, "--", label="DNN approximation")
plt.legend()
plt.show()
