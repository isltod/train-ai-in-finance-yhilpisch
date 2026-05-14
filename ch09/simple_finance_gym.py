import numpy as np
import pandas as pd
import os
import random
import sys
from pylab import plt

sys.path.append(".")
from util import Timer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"

import tensorflow as tf
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from sklearn.metrics import accuracy_score
from collections import deque
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


class DQLAgent:
    def __init__(
        self,
        gamma=0.95,
        hu=24,
        opt=Adam,
        lr=0.001,
        finish=False,
    ):
        self.finish = finish
        # 초기 탐색 비율
        self.epsilon = 1.0
        # 최소 탐색 비율
        self.epsilon_min = 0.01
        # 탐색 비율의 감쇠 비율
        self.epsilon_decay = 0.995
        # 지연 보상에 대한 할인율
        self.gamma = gamma
        self.batch_size = 32
        self.max_treward = 0
        # 이건 뭐에 대한 평균인가?
        self.averages = list()
        # 내용 기록용인가?
        self.memory = deque(maxlen=2000)
        # 이 env 때문에 이 클래스 코드가 env 인스턴스 만든 뒤로 내려가야 되나?
        self.osn = env.observation_space.shape[0]
        # 학습 모델은 완전연결 층 3개로 된 모델...
        self.model = self._build_model(hu, opt, lr)

    def _build_model(self, hu, opt, lr):
        model = Sequential()
        model.add(Dense(hu, input_dim=self.osn, activation="relu"))
        model.add(Dense(hu, activation="relu"))
        # 마지막 출력은 2, (0,1)
        model.add(Dense(env.action_space.n, activation="linear"))
        # 손실 값은 MSE로..
        model.compile(loss="mse", optimizer=opt(learning_rate=lr))
        return model

    def act(self, state):
        # 탐색 비율에 따라 무작위 탐색
        if random.random() <= self.epsilon:
            return env.action_space.sample()
        # 또는 현재까지 학습된 경험으로 선택
        action = self.model.predict(state)[0]
        return np.argmax(action)

    def learn(self, episodes):
        trewards = []
        # 에피소드 반복 횟수만큼 돌면서
        for e in range(1, episodes + 1):
            state = env.reset()[0]
            # 이건 학습을 위해서 (1,4) shape로 만드는 모양...
            state = np.reshape(state, [1, self.osn])
            # 5천번 이내로 학습?
            for _ in range(5000):
                # 행동 선택하고 그에 따른 상태, 보상 등 확인...근데 info는 쓰는데가 없네?
                action = self.act(state)
                next_state, reward, done, trunc, info = env.step(action)
                next_state = np.reshape(next_state, [1, self.osn])
                # 현재 상태, 선택 행동, 보상, 다음 상태 등을 경험으로 기록
                self.memory.append([state, action, reward, next_state, done])
                # 다음 반복 위해서 다음 상태를 현재 상태로...
                state = next_state
                # 끝났으면
                if trunc:
                    # 이건 for 문의 _ 를 변수로 사용하는 거 같은데...안 좋은 코드라고 봐...
                    # 그렇다면 안 죽고 몇 번이나 버텼나를 총 보상 점수로 본다는 얘기...
                    treward = _ + 1
                    trewards.append(treward)
                    # 마지막 25개만 평균내서 그걸 평균 보상이라고 하는 모양...
                    av = sum(trewards[-25:]) / 25
                    self.averages.append(av)
                    # 최대 총 보상
                    self.max_treward = max(self.max_treward, treward)
                    templ = "episode: {:4d}/{} | treward: {:4d} | "
                    templ += "av: {:6.1f} | max: {:4d}"
                    print(
                        templ.format(e, episodes, treward, av, self.max_treward),
                        end="\r",
                    )
                    break
            # 한 에피소드 내에서 5천번 시도해서, 마지막 25개 평균이 195점 넘으면 아예 게임을 끝낸다?
            if av > 195 and self.finish:
                print("평균 195점 넘음...학습 종료")
                break
            # 5천번 시도했던 경험들이 배치 크기 이상으로 쌓이면 리플레이 하면서 학습...
            if len(self.memory) > self.batch_size:
                self.replay()

    def replay(self):
        # 리플레이 하면서 학습할 때는 무작위로 배치 크기만큼만 뽑아서 학습...
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            if not done:
                # Q 값 계산, np.amax는 최대값 반환
                reward += self.gamma * np.amax(self.model.predict(next_state)[0])
            # 도무지 이해가...target은 모델이 내놓는 상승/하락 배팅 점수인데...
            # 그 중에 행동으로 선택했던 쪽에 Q값을 넣어서, 4시간 단위 로그 수익률과 그걸 맞춰?
            target = self.model.predict(state)
            target[0, action] = reward
            # 새로운 상태-행위로 신경망 학습
            self.model.fit(state, target, epochs=1, verbose=False)
        if self.epsilon > self.epsilon_min:
            # 탐색 비율이 최소값보다 높은 상태면 감쇠 비율로 조정
            self.epsilon *= self.epsilon_decay

    def test(self, episodes):
        trewards = []
        # 에피소드만큼 돌면서
        for e in range(1, episodes + 1):
            state = env.reset()[0]
            # 다시 5천번 돌면서
            for _ in range(5001):
                # 행동 취하고 다음 상태, 보상 등 받고
                state = np.reshape(state, [1, self.osn])
                action = np.argmax(self.model.predict(state)[0])
                next_state, reward, done, trunc, info = env.step(action)
                # 다음 상태로 받은 걸 다시 현재 상태로 반복하는데...
                state = next_state
                # 끝났으면
                if trunc:
                    # 이것도 for 문의 _ 를 변수로 사용하는 거 같은데...안 좋은 코드라고 봐...
                    # 그렇다면 안 죽고 몇 번이나 버텼나를 총 보상 점수로 본다는 얘기
                    treward = _ + 1
                    trewards.append(treward)
                    print(
                        "episode: {:4d}/{} | treward: {:4d}".format(
                            e, episodes, treward
                        ),
                        end="\r",
                    )
                    break
        # 모든 에피소드 다 끝내고 에피소드마다 받았던 총 보상 리스트 반환
        return trewards


def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # env.seed(seed)
    env.action_space.seed(seed)


class observation_space:
    def __init__(self, n):
        # 단순히 shape 변수만 설정
        self.shape = (n,)


class action_space:
    def __init__(self, n):
        self.n = n

    def seed(self, seed):
        pass

    def sample(self):
        # 0~n 사이의 난수 반환
        return random.randint(0, self.n - 1)


class Finance:
    url = "data/aiif_eikon_eod_data.csv"

    def __init__(self, symbol, features):
        self.symbol = symbol
        self.features = features
        # 관측 공간의 shape가 (4,), osn은 스칼라 값으로 4
        self.observation_space = observation_space(4)
        self.osn = self.observation_space.shape[0]
        # 행동은 0 또는 1
        self.action_space = action_space(2)
        # 이 정확도를 넘지 못하면 실패라...책은 50%라는데?
        self.min_accuracy = 0.475
        # 데이터 읽어두고 로그 수익률, 정규화, 방향 설정
        self._get_data()
        self._prepare_data()

    def _get_data(self):
        self.raw = pd.read_csv(self.url, index_col=0, parse_dates=True).dropna()

    def _prepare_data(self):
        self.data = pd.DataFrame(self.raw[self.symbol])
        # 로그 수익률
        self.data["r"] = np.log(self.data / self.data.shift(1))
        self.data.dropna(inplace=True)
        # 가우시안 정규화
        self.data = (self.data - self.data.mean()) / self.data.std()
        # 방향성, 상승은 1 하락은 0
        self.data["d"] = np.where(self.data["r"] > 0, 1, 0)

    def _get_state(self):
        # 처음엔 0:4, 다음엔 1:5, 그 다음엔 2:6으로 bar 크기만큼 데이터 읽는 구조...
        return self.data[self.features].iloc[self.bar - self.osn : self.bar].values

    def seed(self, seed=None):
        pass

    def reset(self):
        # 총보상, 정확도, 데이터 인덱스, 상태 데이터 초기화...
        self.treward = 0
        self.accuracy = 0
        self.bar = self.osn
        # 처음엔 0:4, 다음엔 1:5, 그 다음엔 2:6으로 bar 크기만큼 데이터 읽는 구조...
        state = self.data[self.features].iloc[self.bar - self.osn : self.bar]
        # 4 시간 단위 동안의 가격 변화와 빈 딕셔너리 반환...
        return state.values, {}

    def step(self, action):
        trunc = False
        # 행동(0 또는 1)이 bar 인덱스로 선택된 데이터의 방향성(0 또는 1)과 같으면 성공
        correct = action == self.data["d"].iloc[self.bar]
        # 성공하면 보상 1
        reward = 1 if correct else 0
        self.treward += reward
        # 다음 상태를 위해 bar 하나 증가
        self.bar += 1
        # bar - osn은 현재까지 데이터 읽은 횟수...
        self.accuracy = self.treward / (self.bar - self.osn)
        # 데이 다 읽었으면 종료
        if self.bar >= len(self.data):
            done = True
            trunc = True
        # ...니까, 보상이 있다면 종료 아니고...이건 필요 없는거 같은데...
        elif reward == 1:
            done = False
        # 10회 이상 찍었는데 정확도가 min_accuracy 아래면 종료
        elif self.accuracy < self.min_accuracy and self.bar > self.osn + 10:
            done = True
            trunc = True
        else:
            done = False
        # 다음 상태 데이터 읽고
        state = self._get_state()
        # 근데 info는 왜 비워서 반환하지?
        info = {}
        return state, reward, done, trunc, info


env = Finance("EUR=", "EUR=")
print("환경 초기화", env.reset())
a = env.action_space.sample()
print("행위 선택", a)
print("행위 결과", env.step(a))
set_seeds(100)
agent = DQLAgent(gamma=0.5, opt=RMSprop)
episodes = 1000
with Timer():
    agent.learn(episodes)
print("3회 테스트", agent.test(3))
plt.figure(figsize=(10, 6))
x = range(len(agent.averages))
y = np.polyval(np.polyfit(x, agent.averages, deg=3), x)
plt.plot(agent.averages, label="moving average")
plt.plot(x, y, "r--", label="regression")
plt.xlabel("episodes")
plt.ylabel("total reward")
plt.legend()
plt.show()
