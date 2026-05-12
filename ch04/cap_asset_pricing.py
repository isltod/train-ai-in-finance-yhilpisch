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
rets = np.log(raw / raw.shift(1)).dropna()
# 선택한 주가 테이블
print(raw[symbols[:]])
# 그 각각의 첫 번째 값
print(raw[symbols[:]].iloc[0])
# 초기값에 사서 그냥 들고 있었다면 수익률 - 이게 정규화된 시계열이라고...
# (raw[symbols[:]] / raw[symbols[:]].iloc[0]).plot(figsize=(10, 6))
# plt.show()

# 무위험 단기 이자율...예금? 채권?
r = 0.005
# 시장 포트폴리오 설정...S&P 500 지수인가?
market = ".SPX"
res = pd.DataFrame()
for sym in rets.columns[:4]:
    # 각 주식별로 돌면서
    print("\n" + sym)
    print(54 * "=")
    # 다시 연도별로 돌면서...
    for year in range(2010, 2019):
        # 전년도 수익률
        rets_ = rets.loc[f"{year}-01-01":f"{year}-12-31"]
        # 전년도 시장 수익률을 S&P 500 지수 수익률로 구하는데...이것도 평균으로 구하네...
        # 근데 뭐지? 구해만 놓고 안 쓰네?
        # muM = rets_[market].mean() * 252
        # 개별 주식의 베타를 계산한다...
        cov = rets_.cov().loc[sym, market]
        var = rets_[market].var()
        beta = cov / var
        # 현재 연도 수익률...
        rets_ = rets.loc[f"{year + 1}-01-01" :f"{year + 1}-12-31"]
        muM = rets_[market].mean() * 252
        # 전년도 베타와 현재 연도의 시장 기대 수익률에서 개별 주식의 수익률 예상
        mu_capm = r + beta * (muM - r)
        # 현재 연도의 실현 수익률 계산...이게 포트폴리오 성능인가?
        mu_real = rets_[sym].mean() * 252
        # 결과 기록
        res = pd.concat(
            (
                res,
                pd.DataFrame(
                    {"symbol": sym, "mu_capm": mu_capm, "mu_real": mu_real},
                    index=[year + 1],
                ),
            ),
            sort=True,
        )
        print(
            "{} | beta: {:.3f} | mu_capm: {:6.3f} | mu_real: {:6.3f}".format(
                year + 1, beta, mu_capm, mu_real
            )
        )
# 아마존을 예로...
sym = "AMZN.O"
print(
    "예상 수익률과 실제 수익률 상관계수\n",
    res[res["symbol"] == sym][["mu_capm", "mu_real"]].corr(),
)
res[res["symbol"] == sym].plot(kind="bar", figsize=(10, 6), title=sym)
plt.show()
#
grouped = res.groupby("symbol").mean()
print(grouped)
grouped.plot(kind="bar", figsize=(10, 6), title="Average Values")
plt.show()
#
