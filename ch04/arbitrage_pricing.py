import numpy as np
import pandas as pd
from pylab import plt, mpl

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

# VIX 지수 시장 변동성, EUR/USD 환율, XAU 금으로 대표하는 원자재 가격
factors = [".SPX", ".VIX", "EUR=", "XAU="]
res = pd.DataFrame()
np.set_printoptions(formatter={"float": lambda x: f"{x:5.2f}"})
# 각 주식별로 돌면서
for sym in rets.columns[:4]:
    print("\n" + sym)
    print(71 * "=")
    # 다시 연도별로...
    for year in range(2010, 2019):
        # 전년도 수익률을 위 4가지 요인과 선형 회귀
        rets_ = rets.loc[f"{year}-01-01":f"{year}-12-31"]
        reg = np.linalg.lstsq(rets_[factors], rets_[sym], rcond=-1)[0]
        # 현재 연도 수익률
        rets_ = rets.loc[f"{year + 1}-01-01" :f"{year + 1}-12-31"]
        # 평균값을 넣어서 회귀식으로 예측한 주식 수익률 - 이게 차익거래 가격결정 모형이라고?
        mu_apt = np.dot(rets_[factors].mean() * 252, reg)
        # 실제 수익률
        mu_real = rets_[sym].mean() * 252
        res = pd.concat(
            (
                res,
                pd.DataFrame(
                    {"symbol": sym, "mu_apt": mu_apt, "mu_real": mu_real},
                    index=[year + 1],
                ),
            )
        )
        print(
            "{} | fl: {} | mu_apt: {:6.3f} | mu_real: {:6.3f}".format(
                year + 1, reg.round(2), mu_apt, mu_real
            )
        )
# 이렇게 선형 회귀를 해도 별반...
sym = "AMZN.O"
print(
    "예상 수익률과 실제 수익률 상관계수\n",
    res[res["symbol"] == sym][["mu_apt", "mu_real"]].corr(),
)
# res[res["symbol"] == sym].plot(kind="bar", figsize=(10, 6), title=sym)
# plt.show()
#
grouped = res.groupby("symbol").mean()
print(grouped)
# grouped.plot(kind="bar", figsize=(10, 6), title="Average Values")
# plt.show()
#
# 차익거래 가격결정 모형을 위한 리스크 요인이라고 갑자기 보여주는데...
csv = "data/aiif_eikon_eod_factors.csv"
factors = pd.read_csv(csv, index_col=0, parse_dates=True)
# (factors / factors.iloc[0]).plot(figsize=(10, 6))
# plt.show()

# 연도를 한정하고
start = "2017-01-01"
end = "2020-01-01"
# d와 f 테이블이 있는데....d는 데이터, f는 바로 위에서 추가된 시트의 요인들이고...
retsd = rets.loc[start:end].copy()
retsd.dropna(inplace=True)
# 이건 요인들의 로그 변화율인가...
retsf = np.log(factors / factors.shift(1))
# 연도 한정하고...
retsf = retsf.loc[start:end]
retsf.dropna(inplace=True)
# retsd 테이블에 있는 날짜만 추리고 dropna 하면 양쪽 다 있는 레코드만 추리나...
retsf = retsf.loc[retsd.index].dropna()
print("추가 요인들과 수익률 상관 관계", retsf.corr())
res = pd.DataFrame()
np.set_printoptions(formatter={"float": lambda x: f"{x:5.2f}"})
# 데이터 절반 정도로 나누는데...
split = int(len(retsf) * 0.5)
for sym in rets.columns[:4]:
    print("\n" + sym)
    print(74 * "=")
    # 앞 쪽 절반 가지고 선형회귀...
    retsf_, retsd_ = retsf.iloc[:split], retsd.iloc[:split]
    reg = np.linalg.lstsq(retsf_, retsd_[sym], rcond=-1)[0]
    # 뒤쪽 절반 가지고 회귀식에 넣어 예상 수익률과, 뒤쪽 절반에서 구한 실제 수익률?
    retsf_, retsd_ = retsf.iloc[split:], retsd.iloc[split:]
    mu_apt = np.dot(retsf_.mean() * 252, reg)
    mu_real = retsd_[sym].mean() * 252
    res = pd.concat(
        (
            res,
            pd.DataFrame(
                {"mu_apt": mu_apt, "mu_real": mu_real},
                index=[
                    sym,
                ],
            ),
        ),
        sort=True,
    )
    print("fl: {} | apt: {:.3f} | real: {:.3f}".format(reg.round(1), mu_apt, mu_real))
res.plot(kind="bar", figsize=(10, 6))
plt.show()
#
# 종목은 여전히 아마존
print("종목", sym)
# 회귀식으로 예상 수익률, 실제 수익률 구하고 데이터프레임으로...
rets_sym = np.dot(retsf_, reg)
rets_sym = pd.DataFrame(rets_sym, columns=[sym + "_apt"], index=retsf_.index)
rets_sym[sym + "_real"] = retsd_[sym]
# 수익률 연률, 표준편차, 상관관계...
print("연률화된 수익률", rets_sym.mean() * 252)
print("연률화된 표준편차", rets_sym.std() * 252**0.5)
print("상관관계", rets_sym.corr())
rets_sym.cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()
# 방향성만 맞춰보면...
rets_sym["same"] = np.sign(rets_sym[sym + "_apt"]) == np.sign(rets_sym[sym + "_real"])
print("전체", rets_sym["same"].value_counts())
print("맞은 비율", rets_sym["same"].value_counts()[True] / len(rets_sym))
