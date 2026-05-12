import math
import numpy as np
import pandas as pd
import scipy.stats as scs
import statsmodels.api as sm
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
# 이번에는 모든 항목들에 대해서 로그 수익률을 구해놓고...
rets = np.log(raw / raw.shift(1)).dropna()


def dN(x, mu, sigma):
    """Probability density function of a normal random variable x."""
    z = (x - mu) / sigma
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi * sigma**2)
    return pdf


def return_histogram(rets, title=""):
    """Plots a histogram of the returns."""
    plt.figure(figsize=(10, 6))
    x = np.linspace(min(rets), max(rets), 100)
    plt.hist(np.array(rets), bins=50, density=True, label="frequency")
    y = dN(x, np.mean(rets), np.std(rets))
    plt.plot(x, y, linewidth=2, label="PDF")
    plt.xlabel("log returns")
    plt.ylabel("frequency/probability")
    plt.title(title)
    plt.legend()
    plt.show()


def return_qqplot(rets, title=""):
    """Generates a Q-Q plot of the returns."""
    fig = sm.qqplot(rets, line="s", alpha=0.5)
    fig.set_size_inches(10, 6)
    plt.title(title)
    plt.xlabel("theoretical quantiles")
    plt.ylabel("sample quantiles")
    plt.show()


def print_statistics(rets):
    print("RETURN SAMPLE STATISTICS")
    print("---------------------------------------------")
    print("Skew of Sample Log Returns {:9.6f}".format(scs.skew(rets)))
    print("Skew Normal Test p-value   {:9.6f}".format(scs.skewtest(rets)[1]))
    print("---------------------------------------------")
    print("Kurt of Sample Log Returns {:9.6f}".format(scs.kurtosis(rets)))
    print("Kurt Normal Test p-value   {:9.6f}".format(scs.kurtosistest(rets)[1]))
    print("---------------------------------------------")
    print("Normal Test p-value        {:9.6f}".format(scs.normaltest(rets)[1]))
    print("---------------------------------------------")


# 우선 S&P 500으로 정규분표인지 그려보고...아니란다...
symbol = ".SPX"
return_histogram(rets[symbol].values, symbol)
return_qqplot(rets[symbol].values, symbol)
# S&P 500, 아마존, 유료달러 환율, 금? 가격을 대표로 정규성 테스트...
symbols = [".SPX", "AMZN.O", "EUR=", "GLD"]
for sym in symbols:
    print("\n{}".format(sym))
    print(45 * "=")
    print_statistics(rets[sym].values)
