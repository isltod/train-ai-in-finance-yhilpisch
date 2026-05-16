import os
import numpy as np
import pandas as pd
from pylab import plt, mpl
import sys

sys.path.append(".")
from util import Timer
import finance
import tradingbot
import tbbacktesterrm as tbbrm

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

data = pd.DataFrame(learn_env.data[symbol])
print(data.head())

# 이 지표들은 14일 기준으로 계산...
window = 14
# 결국 ATR 계산인거 같은데...
data["min"] = data[symbol].rolling(window).min()
data["max"] = data[symbol].rolling(window).max()
data["mami"] = data["max"] - data["min"]
# 이동 최대/최소와 전날 가격차 절대값이라..
data["mac"] = abs(data["max"] - data[symbol].shift(1))
data["mic"] = abs(data["min"] - data[symbol].shift(1))
# 이동 최대 최소 차이, 최대와 전날 가격 차이, 최소와 전날 가격 차이 중 제일 큰게 ATR
data["atr"] = np.maximum(data["mami"], data["mac"])
data["atr"] = np.maximum(data["atr"], data["mic"])
# ATR%는 가격대비 ATR
data["atr%"] = data["atr"] / data[symbol]

# 이 값으로 손절 수준을 정한다고...
print("ATR 절대값과 상대값\n", data[["atr", "atr%"]].tail())
leverage = 10
print(
    "레버리지 10배에서 ATR 절대값과 상대값\n", data[["atr", "atr%"]].tail() * leverage
)
print("레버리지 10배에서 중앙값\n", data[["atr", "atr%"]].median() * leverage)

data[["atr", "atr%"]].plot(subplots=True, figsize=(10, 6))
plt.show()


# test_env = finance.Finance(
#     symbol,
#     features=learn_env.features,
#     window=learn_env.window,
#     lags=learn_env.lags,
#     leverage=learn_env.leverage,
#     min_performance=0.0,
#     min_accuracy=0.0,
#     start=a + b + c,
#     end=None,
#     mu=learn_env.mu,
#     std=learn_env.std,
# )
# env = test_env

# agent = tradingbot.TradingBot(24, 0.001, learn_env, valid_env, load_model=True)

# tb = tbbrm.TBBacktesterRM(env, agent.model, 10000, 0.0, 0, verbose=False)
