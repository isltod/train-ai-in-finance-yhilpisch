import numpy as np
import pandas as pd
from pylab import plt, mpl
from sklearn.metrics import r2_score

plt.style.use("seaborn-v0_8")
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "serif"
np.set_printoptions(
    precision=5, suppress=True, formatter={"float": lambda x: f"{x:6.3f}"}
)

# 이게 역사적 일간 데이터
url = "data/aiif_eikon_eod_data.csv"
raw = pd.read_csv(url, index_col=0, parse_dates=True).dropna()
# 이번에는 모든 항목들에 대해서 로그 수익률을 구해놓고...
rets = np.log(raw / raw.shift(1)).dropna()

# cap_asset_pricing에서 했던 베타, 자본자산 가격결정 모형의 기대 수익률 등을 다시...
r = 0.005
market = ".SPX"
res = pd.DataFrame()
for sym in rets.columns[:4]:
    for year in range(2010, 2019):
        rets_ = rets.loc[f"{year}-01-01":f"{year}-12-31"]
        muM = rets_[market].mean() * 252
        cov = rets_.cov().loc[sym, market]
        var = rets_[market].var()
        beta = cov / var
        rets_ = rets.loc[f"{year + 1}-01-01" :f"{year + 1}-12-31"]
        muM = rets_[market].mean() * 252
        mu_capm = r + beta * (muM - r)
        mu_real = rets_[sym].mean() * 252
        # 이번에는 베타를 포함해서 결과 기록...
        res = pd.concat(
            (
                res,
                pd.DataFrame(
                    {
                        "symbol": sym,
                        "beta": beta,
                        "mu_capm": mu_capm,
                        "mu_real": mu_real,
                    },
                    index=[year + 1],
                ),
            ),
            sort=True,
        )

# 베타를 독립 변수로, 기대 수익률을 종속 변수로 회귀식 구해서
reg = np.polyfit(res["beta"], res["mu_capm"], deg=1)
# 회귀식으로 기대 수익률의 예측값 구하고...
res["mu_capm_ols"] = np.polyval(reg, res["beta"])
# R2 값 - 0.1...
print("베타-기대수익률 R^2", r2_score(res["mu_capm"], res["mu_capm_ols"]))
# 그려보면 관계가 거의 없다...
res.plot(kind="scatter", x="beta", y="mu_capm", figsize=(10, 6))
x = np.linspace(res["beta"].min(), res["beta"].max())
plt.plot(x, np.polyval(reg, x), "g--", label="regression")
plt.legend()
plt.show()

# 이번에는 베타와 실현 수익률 관계 - 이건 더 상관없다...
reg = np.polyfit(res["beta"], res["mu_real"], deg=1)
res["mu_real_ols"] = np.polyval(reg, res["beta"])
print("베타-실현수익률 R^2", r2_score(res["mu_real"], res["mu_real_ols"]))
res.plot(kind="scatter", x="beta", y="mu_real", figsize=(10, 6))
x = np.linspace(res["beta"].min(), res["beta"].max())
plt.plot(x, np.polyval(reg, x), "g--", label="regression")
plt.legend()
plt.show()
