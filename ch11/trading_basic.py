import os
import numpy as np
import pandas as pd
from pylab import plt, mpl
import sys

sys.path.append(".")
from util import Timer
import finance
import tradingbot

# 이 설정들은 한 군데서 하면 될거 같은데...모듈들 죄다 하고 있네...
plt.style.use("seaborn-v0_8")
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "serif"
pd.set_option("mode.chained_assignment", None)
pd.set_option("display.float_format", "{:.4f}".format)
np.set_printoptions(suppress=True, precision=4)
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"


symbol = "EUR="
features = [symbol, "r", "s", "m", "v"]
a = 0
b = 1750
c = 250

learn_env = finance.Finance(
    symbol,
    features,
    window=20,
    lags=3,
    leverage=1,
    min_performance=0.9,
    min_accuracy=0.475,
    start=a,
    end=a + b,
    mu=None,
    std=None,
)
print("훈련 환경 정보")
print(learn_env.data.info())

valid_env = finance.Finance(
    symbol,
    features=learn_env.features,
    window=learn_env.window,
    lags=learn_env.lags,
    leverage=learn_env.leverage,
    min_performance=0.0,
    min_accuracy=0.0,
    start=a + b,
    end=a + b + c,
    mu=learn_env.mu,
    std=learn_env.std,
)
print("시험 환경 정보")
print(valid_env.data.info())

tradingbot.set_seeds(100)
agent = tradingbot.TradingBot(24, 0.001, learn_env, valid_env)
episodes = 61
with Timer():
    agent.learn(episodes)
tradingbot.plot_treward(agent)
tradingbot.plot_performance(agent)
