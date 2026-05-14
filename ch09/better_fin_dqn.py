import os
import math
import random
import numpy as np
import pandas as pd
from pylab import plt, mpl
from collections import deque

plt.style.use("seaborn-v0_8")
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "serif"
np.set_printoptions(precision=4, suppress=True)
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.metrics import accuracy_score
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

import sys

sys.path.append(".")
from util import Timer


def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # 이건 저 아래에 생기는 클래스를 글로벌 변수로 쓰는데...이거 참...
    env.seed(seed)
    env.action_space.seed(100)


class observation_space:
    def __init__(self, n):
        self.shape = (n,)


class action_space:
    def __init__(self, n):
        self.n = n

    def seed(self, seed):
        pass

    def sample(self):
        return random.randint(0, self.n - 1)


class Finance:
    url = "data/aiif_eikon_eod_data.csv"

    def __init__(
        self,
        symbol,
        features,
        window,
        lags,
        leverage=1,
        min_performance=0.85,
        start=0,
        end=None,
        mu=None,
        std=None,
    ):
        self.symbol = symbol
        # 상태를 정의할 특징 데이터라...
        self.features = features
        self.n_features = len(features)
        self.window = window
        # 쉬프트 몇 칸이나 할지...
        self.lags = lags
        self.leverage = leverage
        # 지켜야 할 최저 누적 수익률
        self.min_performance = min_performance
        # 시작 인덱스...훈련과 테스트 구분
        self.start = start
        self.end = end
        self.mu = mu
        self.std = std
        # 관측 영역, 즉 볼 시간 단위는 lags로 지정된 숫자만큼...
        self.observation_space = observation_space(self.lags)
        # 가능한 행동은 여전히 상승/하락
        self.action_space = action_space(2)
        self._get_data()
        self._prepare_data()

    def _get_data(self):
        # 일단 데이터 읽고
        self.raw = pd.read_csv(self.url, index_col=0, parse_dates=True).dropna()

    def _prepare_data(self):
        self.data = pd.DataFrame(self.raw[self.symbol])
        # 시작 인덱스로 데이터 자르고
        self.data = self.data.iloc[self.start :]
        # 로그 수익률
        self.data["r"] = np.log(self.data / self.data.shift(1))
        self.data.dropna(inplace=True)
        # 이동 평균, 최대, 최소 - 추가적인 금융 특징 데이터라고...
        self.data["s"] = self.data[self.symbol].rolling(self.window).mean()
        self.data["m"] = self.data["r"].rolling(self.window).mean()
        self.data["v"] = self.data["r"].rolling(self.window).std()
        self.data.dropna(inplace=True)
        # 가우스 정규화 - 여기부터는 별도의 data_ 변수로 만든다...
        if self.mu is None:
            self.mu = self.data.mean()
            self.std = self.data.std()
        self.data_ = (self.data - self.mu) / self.std
        # 상승과 하락 방향성...그냥 만들면 실수가 되나? 아무튼 정수형으로 고정...
        self.data_["d"] = np.where(self.data["r"] > 0, 1, 0)
        self.data_["d"] = self.data_["d"].astype(int)
        # 데이터 시작 시간은 맨 앞에서 자르고 끝은 마지막에 자른다?
        if self.end is not None:
            self.data = self.data.iloc[: self.end - self.start]
            self.data_ = self.data_.iloc[: self.end - self.start]

    def _get_state(self):
        # 0:5 -> 1:6 -> 2:7 -> ... 방식으로 차트 읽어 반환...bar가 마지막 봉...
        return self.data_[self.features].iloc[self.bar - self.lags : self.bar]

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        # 성공 횟수 0, 정확도 0, 수익률은 100%로 초기화....
        self.treward = 0
        self.accuracy = 0
        self.performance = 1
        # 쉬프트 갯수가 bar의 처음 값이고...
        self.bar = self.lags
        # 처음엔 0~lags 사이 데이터를 읽어서 상태 값으로 반환
        state = self.data_[self.features].iloc[self.bar - self.lags : self.bar]
        return state.values

    def step(self, action):
        # 마지막 봉의 방향성과 현재 선택 행동 방향이 같으면 정답...
        correct = action == self.data_["d"].iloc[self.bar]
        # 레버리지 곱한 현재 봉의 수익률
        ret = self.data["r"].iloc[self.bar] * self.leverage
        # 그걸 이용해서 기반 보상 처리...성공하면 수익률 만큼 이익, 실패하면 그만큼 손해
        reward_1 = 1 if correct else 0
        reward_2 = abs(ret) if correct else -abs(ret)
        # 이건 뭔지 모르겠고...안 쓰는데?
        factor = 1 if correct else -1
        # 이건 보상이라기 보다는 성공 횟수 정도네...성공했으면 1 추가...
        self.treward += reward_1
        # 한 스텝에서 bar가 1칸씩 이동하네?
        self.bar += 1
        self.accuracy = self.treward / (self.bar - self.lags)
        # 성능은 수익률이었다...현재 수익률을 계속 누적곱하면 누적 수익률이 된다...
        self.performance *= math.exp(reward_2)
        # 데이터 다 읽었으면 종료..
        if self.bar >= len(self.data):
            done = True
        # 수익이 있다는 건 끝난게 아니다? 수익이 없으면? 아래 else 때문에 이건 필요 없는거 같은데...
        elif reward_1 == 1:
            done = False
        # 시작하고 5번 이후에, 최저 누적 수익률 기준보다 떨어졌으면 끝내기...
        elif self.performance < self.min_performance and self.bar > self.lags + 5:
            done = True
        else:
            # 나머진 계속...
            done = False
        # 현재 인덱스가 2:5 상태였으면 bar는 5...
        # 위에서 각종 지표들은 bar 인덱스로 읽었으니까 5 기준으로 계산된다...
        # 계산 끝내고 bar를 1 증가시켰으니 bar=6 되고,
        # 그럼 상태 인덱스가 3:6, 즉 3~5번까지만 읽게 되니, 여기서 했던 계산들과 짝이 된다...
        # 그걸 현재 상태, 현재 보상, 끝내기 정보 등으로 묶어서 반환...
        state = self._get_state()
        info = {}
        return state.values, reward_1 + reward_2 * 5, done, info


# 환율 데이터로 트레이딩 환경 만들고...
env = Finance("EUR=", ["EUR=", "r"], 10, 5)
start_idx = env.action_space.sample()
print("행위 선택", start_idx)
print("환경 초기화", env.reset())

symbol = "EUR="
features = [symbol, "r", "s", "m", "v"]
# 첫 번째 데이터에서 시작해서 2천 개를 학습용
start_idx = 0
end_idx = 2000
# 그 뒤 500개를 검증용 환경으로 선택
c = 500
learn_env = Finance(
    symbol,
    features,
    # 이동 평균 등은 10 시간 단위로 계산
    window=10,
    # 시간 쉬프트는 6 단위까지
    lags=6,
    # 레버리지는 사용 안하고...
    leverage=1,
    # 최저 자산은 85% 유지
    min_performance=0.85,
    start=start_idx,
    end=start_idx + end_idx,
    # 가우스 정규화는 데이터에 기반해서...
    mu=None,
    std=None,
)
learn_env.data.info()
# 검증 환경은 학습 환경 매개변수를 이용해서 초기화...
valid_env = Finance(
    symbol,
    features,
    window=learn_env.window,
    lags=learn_env.lags,
    leverage=learn_env.leverage,
    min_performance=learn_env.min_performance,
    start=start_idx + end_idx,
    end=start_idx + end_idx + c,
    # 근데 평균/분산도 학습 환경거를 갖다 쓰네?
    mu=learn_env.mu,
    std=learn_env.std,
)
valid_env.data.info()


class FQLAgent:
    def __init__(self, hidden_units, learning_rate, learn_env, valid_env):
        # 이번에는 학습 환경과 검증 환경 구분
        self.learn_env = learn_env
        self.valid_env = valid_env
        # 탐색 비율은 100%에서 시작해서 10%까지 -2%로 감쇠
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.98
        self.learning_rate = learning_rate
        # Q 값 계산에서 할인율
        self.gamma = 0.95
        self.batch_size = 128
        # 최대/평균 성공 횟수 기록
        self.max_treward = 0
        self.trewards = list()
        self.averages = list()
        # 이건 수익률일텐데 그 밑엔 뭐지?
        self.performances = list()
        self.aperformances = list()
        self.vperformances = list()
        # 경험 기록
        self.memory = deque(maxlen=2000)
        # 모델은 완전연결 3층에 드롭아웃 끼워서 설정...
        self.model = self._build_model(hidden_units, learning_rate)

    def _build_model(self, hu, lr):
        model = Sequential()
        model.add(
            Dense(
                hu,
                input_shape=(self.learn_env.lags, self.learn_env.n_features),
                activation="relu",
            )
        )
        model.add(Dropout(0.3, seed=100))
        model.add(Dense(hu, activation="relu"))
        model.add(Dropout(0.3, seed=100))
        model.add(Dense(2, activation="linear"))
        model.compile(loss="mse", optimizer=RMSprop(learning_rate=lr))
        return model

    def act(self, state):
        # 탐색 비율에 따라서 무작위 탐색 또는 학습된 모델로 행동 선택
        if random.random() <= self.epsilon:
            return self.learn_env.action_space.sample()
        action = self.model.predict(state)[0, 0]
        return np.argmax(action)

    def replay(self):
        # 에피소드 끝내고나면 경험 재생해서 학습...학습용 배치 데이터는 무작위 선택
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            if not done:
                # Q 값 계산, np.amax는 최대값 반환
                reward += self.gamma * np.amax(self.model.predict(next_state)[0, 0])
            # 이건 여전히 전혀 의미를 모르겠고...
            target = self.model.predict(state)
            target[0, 0, action] = reward
            self.model.fit(state, target, epochs=1, verbose=False)
        # 다음 에피소드를 위해서 탐색 비율 감쇠
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, episodes):
        # 에피소드 수만큼 돌면서
        for e in range(1, episodes + 1):
            # 환경 초기화하고 게임 시작 상태값 받고
            state = self.learn_env.reset()
            state = np.reshape(
                state, [1, self.learn_env.lags, self.learn_env.n_features]
            )
            # 다시 만 번 시도해보는데...
            for _ in range(10000):
                # 행동 선택하고, 그 행동에 따른 다음 상태, 보상, 종료 여부 등 받고...
                action = self.act(state)
                next_state, reward, done, info = self.learn_env.step(action)
                # 다음 상태 벡터를 (1, 쉬프트 수, 특성 수) 형태로 변환해서 경험으로 저장
                next_state = np.reshape(
                    next_state, [1, self.learn_env.lags, self.learn_env.n_features]
                )
                self.memory.append([state, action, reward, next_state, done])
                # 다음 반복을 위해 다음 상태가 현재 상태로...
                state = next_state
                # 자산 하한선 아래이거나 데이터 다 읽었으면 done....
                if done:
                    # 이건 for 문의 _ 를 변수로 사용하는 거 같은데...안 좋은 코드라고 봐...
                    # 그렇다면 안 죽고 몇 번이나 버텼나를 총 보상 점수로 본다는 얘기
                    # 환경에서는 같은 변수가 몇 번 맞췄나이고, 에이전트에서는 이게 몇 번 살아 남았나...
                    treward = _ + 1
                    self.trewards.append(treward)
                    # averages는 마지막 25번의 평균적으로 몇 번째까지 살아남았나....
                    av = sum(self.trewards[-25:]) / 25
                    self.averages.append(av)
                    # 이건 환경에서 받은 수익률...근데 수익률을 왜 환경에서 받지?
                    perf = self.learn_env.performance
                    self.performances.append(perf)
                    # a는 마지막 25번의 수익률 평균
                    self.aperformances.append(sum(self.performances[-25:]) / 25)
                    self.max_treward = max(self.max_treward, treward)
                    templ = "episode: {:2d}/{} | treward: {:4d} | "
                    templ += "perf: {:5.3f} | av: {:5.1f} | max: {:4d}"
                    print(
                        templ.format(e, episodes, treward, perf, av, self.max_treward),
                        # 이거 덕분에 커서가 안내려가고 그 상태에서 숫자가 업데이트 되듯이 보이는 효과인 듯...
                        end="\r",
                    )
                    break
            # 이것도 이전 만 번 게임으로 학습된 모델로 위에 학습환경의 정확도를 보고하고,
            # 같은 모델로 검증 환경의 정확도를 보고하려고 이런 순서로 진행...
            self.validate(e, episodes)
            # 그래서 검증까지 한 후에 리플레이해서 다음 에피소드용 모델 학습...
            if len(self.memory) > self.batch_size:
                self.replay()

    def validate(self, e, episodes):
        # 검증은 에피소드 크기만큼 도는게 아니라, 각 에피소드에 한 번씩...
        # 검증도 환경 초기화하고 초기 상태 정보 받고...
        state = self.valid_env.reset()
        state = np.reshape(state, [1, self.valid_env.lags, self.valid_env.n_features])
        # 시도 만 번은 같고...
        for _ in range(10000):
            # 행동 결정하고, 그에 따른 다음 상태, 보상, 종료 여부 등 받고...
            action = np.argmax(self.model.predict(state)[0, 0])
            next_state, reward, done, info = self.valid_env.step(action)
            state = np.reshape(
                next_state, [1, self.valid_env.lags, self.valid_env.n_features]
            )
            # 데이터 다 읽었거나 최소 자산 아래로 떨어졌으면 done...
            if done:
                # 몇 번 살아남았고, 수익률은 얼마인지...
                treward = _ + 1
                perf = self.valid_env.performance
                self.vperformances.append(perf)
                if e % 20 == 0:
                    templ = 71 * "="
                    templ += "\nepisode: {:2d}/{} | VALIDATION | "
                    templ += "treward: {:4d} | perf: {:5.3f} | "
                    templ += "eps: {:.2f}\n"
                    templ += 71 * "="
                    print(templ.format(e, episodes, treward, perf, self.epsilon))
                break


set_seeds(100)
# 히든 층은 24개 노드로, 학습률...
agent = FQLAgent(24, 0.0001, learn_env, valid_env)
# 이번엔 에피소드를 60번만?
episodes = 200
with Timer():
    agent.learn(episodes)
print("학습 후 탐색 비율", agent.epsilon)

plt.figure(figsize=(10, 6))
x = range(1, len(agent.averages) + 1)
y = np.polyval(np.polyfit(x, agent.averages, deg=3), x)
plt.plot(agent.averages, label="moving average")
plt.plot(x, y, "r--", label="regression")
plt.xlabel("episodes")
plt.ylabel("total reward")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
x = range(1, len(agent.performances) + 1)
y = np.polyval(np.polyfit(x, agent.performances, deg=3), x)
y_ = np.polyval(np.polyfit(x, agent.vperformances, deg=3), x)
plt.plot(agent.performances[:], label="training")
plt.plot(agent.vperformances[:], label="validation")
plt.plot(x, y, "r--", label="regression (train)")
plt.plot(x, y_, "r-.", label="regression (valid)")
plt.xlabel("episodes")
plt.ylabel("gross performance")
plt.legend()
plt.show()
