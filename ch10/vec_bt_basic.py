import os
import math
import numpy as np
import pandas as pd
from pylab import plt, mpl

plt.style.use("seaborn-v0_8")
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "serif"
pd.set_option("mode.chained_assignment", None)
pd.set_option("display.float_format", "{:.4f}".format)
np.set_printoptions(suppress=True, precision=4)
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"

# 데이터는 유로/달러 환율
url = "data/aiif_eikon_eod_data.csv"
symbol = "EUR="
data = pd.DataFrame(pd.read_csv(url, index_col=0, parse_dates=True).dropna()[symbol])
print(data.info())

# 전략은 42/258 이동평균
data["SMA1"] = data[symbol].rolling(42).mean()
data["SMA2"] = data[symbol].rolling(258).mean()
# data.plot(figsize=(10, 6))
# plt.show()

# 단기가 위면 롱, 아래면 숏
data.dropna(inplace=True)
data["p"] = np.where(data["SMA1"] > data["SMA2"], 1, -1)
# 종가가 나와야 이동평균을 계산할 수 있으니, 그 날의 포지션 전략은 다음 날 사용 가능한 것...
data["p"] = data["p"].shift(1)
data.dropna(inplace=True)
# data.plot(figsize=(10, 6), secondary_y="p")
# plt.show()

# 로그 수익률
data["r"] = np.log(data[symbol] / data[symbol].shift(1))
data.dropna(inplace=True)
# 로그 수익률을 전략에 곱하면 그대로 수익률? 숏 포지션은?
data["s"] = data["p"] * data["r"]
# 누적 수익률은 로그니까 더해서(곱이되고) exp 하면 log 풀리고
data[["r", "s"]].sum().apply(np.exp)  # gross performance
# 원래 1에서 시작했으니 그걸 빼면 남는 돈 비율...
data[["r", "s"]].sum().apply(np.exp) - 1  # net performance
# data[["r", "s"]].cumsum().apply(np.exp).plot(figsize=(10, 6))
# plt.show()
# 남긴 하는데...이건 제대로 된 비교라기 보다는 그냥 롱만 하는 경우와 롱/숏 하는 경우 비교인데...
# 그것도 10년동안 40% 수익률이면 영...

print("매매 횟수", sum(data["p"].diff() != 0) + 1)
# 수수료를 약 0.5%로 하면 남는 비율은 0.995이고, 여기에 로그 취하면 ln(0.995) = -0.005 정도 된다...
print("log(0.995)", np.log(0.995))
pc = 0.005
# 그래서 남는 비율을 0.995로 조정하기 위해서 그냥 숫자 -0.005가 들어간다...
data["s_"] = np.where(data["p"].diff() != 0, data["s"] - pc, data["s"])
# 이 코드는 경고 부른다...그래서 바꿈
# data["s_"].iloc[-1] -= pc
# 진입 매매 조정? 맨 마지막 값을 0.005 빼주는게 왜 진입매매 조정이지?
data.loc[data.index[-1], "s_"] -= pc
print("포지션 변경 시기", data[["r", "s", "s_"]][data["p"].diff() != 0])
print(
    "기본 수익, 전략 수익, 거래비용 고려 수익\n",
    data[["r", "s", "s_"]].sum().apply(np.exp),
)
print(
    "기본 수익, 전략 수익, 거래비용 고려 수익(NET)\n",
    data[["r", "s", "s_"]].sum().apply(np.exp) - 1,
)
print("손익의 변동\n", data[["r", "s", "s_"]].std())
print("연률화된 손익 변동\n", data[["r", "s", "s_"]].std() * math.sqrt(252))
data[["r", "s", "s_"]].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()

# 일단 백테스팅 코드가 간단하고, 속도도 엄청 빠르다...
# 근데 수수료 반영이 진입과 청산이 아니고 바뀐 시점에 한 번 이뤄진다는 문제...
# 그래서 수수료를 2배로 잡아야 할 것 같고...
# 익절과 손절 전략을 쓸 수가 없단다...
