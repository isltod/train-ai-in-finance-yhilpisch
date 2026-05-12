import numpy as np
import pandas as pd
from pylab import plt, mpl
from scipy.optimize import minimize

plt.style.use("seaborn-v0_8")
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "serif"
np.set_printoptions(
    precision=5, suppress=True, formatter={"float": lambda x: f"{x:6.3f}"}
)
# 이게 역사적 일간 데이터
url = "data/aiif_eikon_eod_data.csv"
raw = pd.read_csv(url, index_col=0, parse_dates=True).dropna()
raw.info()
symbols = ["AAPL.O", "MSFT.O", "INTC.O", "AMZN.O", "GLD"]
# 나누기와 로그 연산을 이렇게 한 번에 할 수 있구나...로그 수익률
rets = np.log(raw[symbols] / raw[symbols].shift(1)).dropna()
# 선택한 주가 테이블
print(raw[symbols[:]])
# 그 각각의 첫 번째 값
print(raw[symbols[:]].iloc[0])
# 초기값에 사서 그냥 들고 있었다면 수익률 - 이게 정규화된 시계열이라고...
# (raw[symbols[:]] / raw[symbols[:]].iloc[0]).plot(figsize=(10, 6))
# plt.show()

# 동일 가중치 포트폴리오를 기준 값으로 보겠다는 가정...
# 2 종목이라면 [1/2] * 2 -> [1/2, 1/2] 동일 가중치 포트폴리오...
weights = len(rets.columns) * [1 / len(rets.columns)]
print("포트폴리오", weights)


# 연률화 수익률
def port_return(rets, weights):
    # 이게 일간 로그 수익률일텐데...그냥 처음부터 마지막까지 더해서 exp 해야 되는거 아닌가?
    return np.dot(rets.mean(), weights) * 252  # annualized


port_return(rets, weights)


# 변동성
def port_volatility(rets, weights):
    return np.dot(weights, np.dot(rets.cov() * 252, weights)) ** 0.5  # annualized


port_volatility(rets, weights)


# 샤프지수
def port_sharpe(rets, weights):
    return port_return(rets, weights) / port_volatility(rets, weights)


port_sharpe(rets, weights)

# 여러가지 랜덤 포트폴리오를 몬테카를로 시뮬레이션...
# (1000, n)이면 1000개의 포트폴리오
w = np.random.random((1000, len(symbols)))
w = (w.T / w.sum(axis=1)).T
print("5번째까지 포트폴리오", w[:5])
print("5번째까지 포트폴리오의 합", w[:5].sum(axis=1))
# 각 포트폴리오에 대해서 변동성과 수익률 계산해서 리스트 -> 넘파이 배열로
pvr = [
    (port_volatility(rets[symbols], weights), port_return(rets[symbols], weights))
    for weights in w
]
pvr = np.array(pvr)
# 배열1과 0인 port_return / port_volatility -> 샤프 비율 계산
psr = pvr[:, 1] / pvr[:, 0]
# plt.figure(figsize=(10, 6))
# fig = plt.scatter(pvr[:, 0], pvr[:, 1], c=psr, cmap="coolwarm")
# cb = plt.colorbar(fig)
# cb.set_label("Sharpe ratio")
# plt.xlabel("expected volatility")
# plt.ylabel("expected return")
# plt.title(" | ".join(symbols))
# plt.show()

# 2010년 샤프 지수가 최대가 되는 포트폴리오로 2011년 수익을 내고, 2011 -> 2012...이런 식으로...
# 각 주식들의 가중치 한계를 정한다고...
bnds = len(symbols) * [
    (0, 1),
]
# 가중치 합은 1이 되게하는 조건이라고...
cons = {"type": "eq", "fun": lambda weights: weights.sum() - 1}
opt_weights = {}
# 연도별로 돌면서
for year in range(2010, 2019):
    # 각 연도 데이터를 뽑아서
    rets_ = rets[symbols].loc[f"{year}-01-01":f"{year}-12-31"]
    # 동일 가중치에서 출발해서 샤프지수를 최대화하는 가중치 구하기라...
    ow = minimize(
        lambda weights: -port_sharpe(rets_, weights),
        len(symbols) * [1 / len(symbols)],
        bounds=bnds,
        constraints=cons,
    )["x"]
    # 각 연도별 최대 샤프지수 가중치 기록
    opt_weights[year] = ow
print("연도별 최대 샤프지수 가중치", opt_weights)
# 전년도 데이터로 계산한 포트폴리오와 기대 수익률과 실제 수익률을 비교하는데...
res = pd.DataFrame()
for year in range(2010, 2019):
    rets_ = rets[symbols].loc[f"{year}-01-01":f"{year}-12-31"]
    # 이게 기대 수익률 통계이고 - 전년도 가중치로 전년도 수익률, 변동성, 샤프지수 구하고
    epv = port_volatility(rets_, opt_weights[year])
    epr = port_return(rets_, opt_weights[year])
    esr = epr / epv
    # 이게 실제 수익률 통계라고...전년도 가중치로 다음년도 통계 구하고...
    rets_ = rets[symbols].loc[f"{year + 1}-01-01" :f"{year + 1}-12-31"]
    rpv = port_volatility(rets_, opt_weights[year])
    rpr = port_return(rets_, opt_weights[year])
    rsr = rpr / rpv
    # 다음 년도 기준으로 이어 붙인다...
    res = pd.concat(
        (
            res,
            pd.DataFrame(
                {
                    "epv": epv,
                    "epr": epr,
                    "esr": esr,
                    "rpv": rpv,
                    "rpr": rpr,
                    "rsr": rsr,
                },
                index=[year + 1],
            ),
        )
    )
# 평균과 상관 보고 그린다...변동성은 잘 맞추는데, 수익률은 못 맞춘다...
print("평균?", res.mean())
print("기대 변동성과 실제 변동성 상관계수\n", res[["epv", "rpv"]].corr())
res[["epv", "rpv"]].plot(
    kind="bar", figsize=(10, 6), title="Expected vs. Realized Portfolio Volatility"
)
plt.show()
print("기대 수익률과 실제 수익률 상관계수\n", res[["epr", "rpr"]].corr())
res[["epr", "rpr"]].plot(
    kind="bar", figsize=(10, 6), title="Expected vs. Realized Portfolio Return"
)
plt.show()
print("기대 샤프지수와 실제 샤프지수 상관계수\n", res[["esr", "rsr"]].corr())
res[["esr", "rsr"]].plot(
    kind="bar", figsize=(10, 6), title="Expected vs. Realized Sharpe Ratio"
)
plt.show()
