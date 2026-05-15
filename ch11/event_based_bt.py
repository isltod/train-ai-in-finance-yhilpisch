import os
import numpy as np
import pandas as pd
from pylab import plt, mpl
import sys

sys.path.append(".")
from util import Timer
import finance
import tradingbot
import backtesting as bt

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
tradingbot.set_seeds(100)
agent = tradingbot.TradingBot(24, 0.001, learn_env, valid_env)
# 당연히 에이전트를 훈련 시켜야 뭔가 되지...참내...
episodes = 61
with Timer():
    agent.learn(episodes)

bb = bt.BacktestingBase(
    env=agent.learn_env,
    model=agent.model,
    amount=10000,
    ptc=0.0001,
    ftc=1.0,
    verbose=True,
)
print("초기 자산", bb.initial_amount)
bar = 100
print("100번째 날짜와 가격", bb.get_date_price(bar))
print("100번째 봉의 환경 상태\n", bb.env.get_state(bar))
# 5000유료? 달러? 100번째 봉에서 매수해보기...
bb.place_buy_order(bar, amount=5000)
# 200번째 봉에서의 평가 자산
bb.print_net_wealth(2 * bar)
# 200번째 봉에서 매도해보기...
bb.place_sell_order(2 * bar, units=1000)
# 300번째 봉에서 모든 포지션 종료...
bb.close_out(3 * bar)

# 올인 전략 백테스트...수수료 등 없는 경우...
tb = bt.TBBacktester(learn_env, agent.model, 10000, 0.0, 0, verbose=False)
tb.backtest_strategy()
# 올인 전략 백테스트...이건 수수료 등 있는 경우...
tb_ = bt.TBBacktester(learn_env, agent.model, 10000, 0.00012, 0.0, verbose=False)
tb_.backtest_strategy()
ax = tb.net_wealths.plot(figsize=(10, 6))
tb_.net_wealths.columns = ["net_wealth (after tc)"]
tb_.net_wealths.plot(ax=ax)
plt.show()
