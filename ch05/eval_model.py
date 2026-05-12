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
from keras.callbacks import EarlyStopping

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

# 훈련 50%, 검증과 시험 25% 분리...이건 셔플
te = int(0.25 * len(f))
va = int(0.25 * len(f))
ind = np.arange(len(f))
# 넘파이 셔플은 이렇게 쓰는구나...
np.random.shuffle(ind)
ind_te = np.sort(ind[:te])
ind_va = np.sort(ind[te : te + va])
ind_tr = np.sort(ind[te + va :])
f_te = f[ind_te]
f_va = f[ind_va]
f_tr = f[ind_tr]
l_te = l[ind_te]
l_va = l[ind_va]
l_tr = l[ind_tr]

# 이번에 회귀는 4차씩 증가시켜 22차까지...
reg = {}
mse = {}
min_mse = np.inf
best_d = 4
for d in range(1, 22, 4):
    reg[d] = np.polyfit(f_tr, l_tr, deg=d)
    p = np.polyval(reg[d], f_tr)
    # 훈련 데이터에 대한 MSE
    mse_tr = MSE(l_tr, p)
    p = np.polyval(reg[d], f_va)
    # 검증 데이터에 대한 MSE
    mse_va = MSE(l_va, p)
    mse[d] = (mse_tr, mse_va)
    # 여기서 검증 데이터에 대한 MSE가 가장 낮은 차수를 저장해놓자...
    if mse_va < min_mse:
        min_mse = mse_va
        best_d = d
    print(f"{d:2d} | MSE_tr={mse_tr:7.5f} | MSE_va={mse_va:7.5f}")

# 차트 그리긴데...위 반복에 넣으면 좀 복잡해도 중복을 없앨 수 있을 텐데...
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax[0].plot(f_tr, l_tr, "ro", label="training data")
ax[1].plot(f_va, l_va, "go", label="validation data")
for d in reg:
    p = np.polyval(reg[d], f_tr)
    ax[0].plot(f_tr, p, "--", label=f"deg={d} (tr)")
    p = np.polyval(reg[d], f_va)
    plt.plot(f_va, p, "--", label=f"deg={d} (va)")
ax[0].legend()
ax[1].legend()
plt.show()

# 이번에 신경망은 2개 히든, 출력 노드는 256
model = create_dnn_model(2, 256)
# 조기 종료는 콜백으로 만들어서 넣어주는 모양...
# 조기 종료 기준은 평균 제곱 오차...
# 100번 안에 어떤? 기준만큼 향상 못하면 종료? 조기 종료하면 그중 제일 괜찮았던 가중치 복원...
callbacks = [EarlyStopping(monitor="loss", patience=100, restore_best_weights=True)]
with Timer():
    hist = model.fit(
        f_tr,
        l_tr,
        epochs=3000,
        verbose=False,
        # 검증 데이터는 이렇게 따로...
        validation_data=(f_va, l_va),
        callbacks=callbacks,
    )
# 결과 그리기...
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
ax[0].plot(f_tr, l_tr, "ro", label="training data")
# 훈련데이터에 대한 예측
p = model.predict(f_tr)
ax[0].plot(f_tr, p, "--", label=f"DNN (tr)")
ax[0].legend()
ax[1].plot(f_va, l_va, "go", label="validation data")
# 검증 데이터에 대한 예측
p = model.predict(f_va)
ax[1].plot(f_va, p, "--", label=f"DNN (va)")
ax[1].legend()
plt.show()

# hist.history를 이용해서 평균 제곱 오차가 어떻게 변하는지 사후 확인...
res = pd.DataFrame(hist.history)
print(res.tail())
# res df에서 35번부터 끝까지, 25개씩 건너뛰며 선택해서 그리기...
res.iloc[35::25].plot(figsize=(10, 6))
plt.ylabel("MSE")
plt.xlabel("epochs")
plt.show()

# 회귀 베스트와 신경망 베스트? 비교해보자...
p_ols = np.polyval(reg[best_d], f_te)
p_dnn = model.predict(f_te).flatten()
# 이렇게 해도 신경망이 좀 더 낮긴하다...
print("베스트 회귀 MSE", MSE(l_te, p_ols))
print("베스트 신경망 MSE", MSE(l_te, p_dnn))
plt.figure(figsize=(10, 6))
plt.plot(f_te, l_te, "ro", label="test data")
plt.plot(f_te, p_ols, "--", label="OLS prediction")
plt.plot(f_te, p_dnn, "-.", label="DNN prediction")
plt.legend()
plt.show()
