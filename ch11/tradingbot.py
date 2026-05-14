#
# Financial Q-Learning Agent
#
# (c) Dr. Yves J. Hilpisch
# Artificial Intelligence in Finance
#
import os
import random
import logging
import numpy as np
from pylab import plt, mpl
from collections import deque
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
plt.style.use("seaborn-v0_8")
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "serif"


def set_seeds(seed=100):
    # 난수 생성 시드 고정
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class TradingBot:
    def __init__(
        self,
        hidden_units,
        learning_rate,
        learn_env,
        valid_env=None,
        val=True,
        dropout=False,
    ):
        self.learn_env = learn_env
        self.valid_env = valid_env
        self.val = val
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.learning_rate = learning_rate
        self.gamma = 0.5
        self.batch_size = 128
        self.max_treward = 0
        self.averages = list()
        self.trewards = []
        self.performances = list()
        self.aperformances = list()
        self.vperformances = list()
        self.memory = deque(maxlen=2000)
        self.model = self._build_model(hidden_units, learning_rate, dropout)

    def _build_model(self, hu, lr, dropout):
        # 3개의 완전연결 층 사이에 드롭아웃
        model = Sequential()
        model.add(
            Dense(
                hu,
                input_shape=(self.learn_env.lags, self.learn_env.n_features),
                activation="relu",
            )
        )
        if dropout:
            model.add(Dropout(0.3, seed=100))
        model.add(Dense(hu, activation="relu"))
        if dropout:
            model.add(Dropout(0.3, seed=100))
        model.add(Dense(2, activation="linear"))
        model.compile(loss="mse", optimizer=RMSprop(learning_rate=lr))
        return model

    def act(self, state):
        # 엡실론 비율에 따라 무작위 행동 또는 지금까지 학습된 모델 예측...
        if random.random() <= self.epsilon:
            return self.learn_env.action_space.sample()
        action = self.model.predict(state)[0, 0]
        return np.argmax(action)

    def replay(self):
        # 각 에피소드 끝에, 경험을 리플레이해서 학습...배치는 랜덤으로 뽑고
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            if not done:
                reward += self.gamma * np.amax(self.model.predict(next_state)[0, 0])
            target = self.model.predict(state)
            # 이건 이해가 안되고 있고...
            target[0, 0, action] = reward
            self.model.fit(state, target, epochs=1, verbose=False)
        # 다음 도전 전에 엡실론은 감쇠
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, episodes):
        # 에피소드만큼 반복해서 각각 만 번 시도해서, 행동하고 상태받아서 경험으로 저장...
        for e in range(1, episodes + 1):
            state = self.learn_env.reset()
            state = np.reshape(
                state, [1, self.learn_env.lags, self.learn_env.n_features]
            )
            for _ in range(10000):
                action = self.act(state)
                next_state, reward, done, info = self.learn_env.step(action)
                next_state = np.reshape(
                    next_state, [1, self.learn_env.lags, self.learn_env.n_features]
                )
                self.memory.append([state, action, reward, next_state, done])
                state = next_state
                if done:
                    treward = _ + 1
                    self.trewards.append(treward)
                    av = sum(self.trewards[-25:]) / 25
                    perf = self.learn_env.performance
                    self.averages.append(av)
                    self.performances.append(perf)
                    self.aperformances.append(sum(self.performances[-25:]) / 25)
                    self.max_treward = max(self.max_treward, treward)
                    templ = "episode: {:2d}/{} | treward: {:4d} | "
                    templ += "perf: {:5.3f} | av: {:5.1f} | max: {:4d}"
                    print(
                        templ.format(e, episodes, treward, perf, av, self.max_treward),
                        end="\r",
                    )
                    break
            # 검증 정확도도 같은 모델에서 측정하고
            if self.val:
                self.validate(e, episodes)
            # 다음 에피소드에는 학습한 모델로 임하고...
            if len(self.memory) > self.batch_size:
                self.replay()
        print()

    def validate(self, e, episodes):
        # 검증 데이터로 정확도 측정...
        state = self.valid_env.reset()
        state = np.reshape(state, [1, self.valid_env.lags, self.valid_env.n_features])
        for _ in range(10000):
            action = np.argmax(self.model.predict(state)[0, 0])
            next_state, reward, done, info = self.valid_env.step(action)
            state = np.reshape(
                next_state, [1, self.valid_env.lags, self.valid_env.n_features]
            )
            if done:
                treward = _ + 1
                perf = self.valid_env.performance
                self.vperformances.append(perf)
                if e % int(episodes / 6) == 0:
                    templ = 71 * "="
                    templ += "\nepisode: {:2d}/{} | VALIDATION | "
                    templ += "treward: {:4d} | perf: {:5.3f} | eps: {:.2f}\n"
                    templ += 71 * "="
                    print(templ.format(e, episodes, treward, perf, self.epsilon))
                break


def plot_treward(agent):
    # 몇 번까지 살아남았나 마지막 25개 평균과 그 회귀 그래프...
    plt.figure(figsize=(10, 6))
    x = range(1, len(agent.averages) + 1)
    y = np.polyval(np.polyfit(x, agent.averages, deg=3), x)
    plt.plot(x, agent.averages, label="moving average")
    plt.plot(x, y, "r--", label="regression")
    plt.xlabel("episodes")
    plt.ylabel("total reward")
    plt.legend()
    plt.show()


def plot_performance(agent):
    # 검증 훈련과 검증 데이터로 수익률 그래프...
    plt.figure(figsize=(10, 6))
    x = range(1, len(agent.performances) + 1)
    y = np.polyval(np.polyfit(x, agent.performances, deg=3), x)
    plt.plot(x, agent.performances[:], label="training")
    plt.plot(x, y, "r--", label="regression (train)")
    if agent.val:
        y_ = np.polyval(np.polyfit(x, agent.vperformances, deg=3), x)
        plt.plot(x, agent.vperformances[:], label="validation")
        plt.plot(x, y_, "r-.", label="regression (valid)")
    plt.xlabel("episodes")
    plt.ylabel("gross performance")
    plt.legend()
    plt.show()
