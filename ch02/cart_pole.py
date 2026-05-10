# import gym
# 바뀐 버전
import gymnasium as gym
import numpy as np
import pandas as pd

np.random.seed(100)
import warnings

warnings.simplefilter("ignore")
# env = gym.make("CartPole-v0")
# 바뀐 버전
env = gym.make("CartPole-v1")
action_size = env.action_space.n
print("액션 개수:", action_size)
print("액션 샘플:", [env.action_space.sample() for _ in range(10)])
state_size = env.observation_space.shape[0]
print("상태 개수:", state_size)
state = env.reset()
# [cart position, cart velocity, pole angle, pole angular velocity]
print("상태 정보", state)
state, reward, done, trunc, _ = env.step(env.action_space.sample())
print("다음 상태 정보:", state, reward, done, trunc, _)
import sys

sys.path.append(".")
from util import Timer

with Timer():
    data = pd.DataFrame()
    state = env.reset()
    length = []
    for run in range(25000):
        done = False
        prev_state = env.reset()[0]
        treward = 1
        results = []
        while not done:
            action = env.action_space.sample()
            state, reward, done, trunc, _ = env.step(action)
            results.append(
                {
                    "s1": prev_state[0],
                    "s2": prev_state[1],
                    "s3": prev_state[2],
                    "s4": prev_state[3],
                    "a": action,
                    "r": reward,
                }
            )
            treward += reward if not done else 0
            prev_state = state
        # 게임이 끝났을 때 점수가 110점이 넘으면
        if treward >= 110:
            # 그 동안 받았던 상태정보, 선택했던 행동, 보상 기록
            data = pd.concat((data, pd.DataFrame(results)))
            # 게임 점수 기록
            length.append(treward)
# 게임 별 점수 평균
print("평균 점수:", np.array(length).mean())
print(data.tail())

# 이 데이터로 신경망을 학습시킨다...
import os

print("환경 설정---------------------------------------------------------")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pylab import plt

plt.style.use("seaborn-v0_8")
import tensorflow as tf

tf.random.set_seed(100)
from tensorflow.python.framework.ops import disable_eager_execution

# 2.0부터 session을 정의하고 run을 실행하는 과정이 없어졌는데...이걸 이전으로 돌리나?
disable_eager_execution()
from keras.layers import Dense
from keras.models import Sequential

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# 완전연결층이 2개...
model = Sequential()
model.add(Dense(64, activation="relu", input_dim=env.observation_space.shape[0]))
model.add(Dense(1, activation="sigmoid"))
# 이게 모델이 실행되게 준비하는 거라고...
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
with Timer():
    model.fit(
        data[["s1", "s2", "s3", "s4"]],
        data["a"],
        epochs=25,
        verbose=False,
        validation_split=0.2,
    )
# 결과는 이렇게 볼 수가 있나...
res = pd.DataFrame(model.history.history)
res.tail(3)
res.plot(figsize=(10, 6), style="--")
plt.show()


# 신경망으로 게임을 하나?
def epoch():
    print("|", end="")
    done = False
    state = env.reset()[0]
    trunc = False
    treward = 0
    while not done and not trunc:
        # 이게 학습된 신경망이 행동을 선택하게 하는 코드인거 같고...
        action = np.where(model.predict(np.atleast_2d(state))[0][0] > 0.5, 1, 0)
        state, reward, done, trunc, _ = env.step(action)
        treward += reward if not done else 0
    return treward


# with Timer():
#     res = np.array([epoch() for _ in range(100)])
#     print()
import time

start = time.perf_counter()
res = np.array([epoch() for _ in range(100)])
print()
print(f"소요 시간: {time.perf_counter() - start:.5f}초")

print(res)
print(res.mean())
