import numpy as np
import pandas as pd
from pylab import plt, mpl
from sklearn.metrics import r2_score

np.random.seed(100)
plt.style.use("seaborn-v0_8")
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "serif"


# 성능 척도는 평균 제곱 오차
def MSE(l, p):
    return np.mean((l - p) ** 2)


def evaluate(reg, f, l):
    p = np.polyval(reg, f)
    # 절대값 오차 평균을 편향으로...
    bias = np.abs(l - p).mean()
    # 예측값의 분산을 분산으로? 원래 정답이 분산이 크면 문제 없는거 아닌가?
    var = p.var()
    msg = f"MSE={MSE(l, p):.4f} | R2={r2_score(l, p):9.4f} | "
    msg += f"bias={bias:.4f} | var={var:.4f}"
    print(msg)


url = "data/aiif_eikon_eod_data.csv"
raw = pd.read_csv(url, index_col=0, parse_dates=True)["EUR="]
# 월말 값으로 리샘플링
l = raw.resample("1ME").last()
# y값은 평균을 제거해주고
l = l.values
l -= l.mean()
# 별 의미없는 x축을 만들고...
f = np.linspace(-2, 2, len(l))

# 훈련 데이터는 20까지만 짝수 인덱스...
f_tr = f[:20:2]
l_tr = l[:20:2]
# 검증 데이터는 20까지 홀수 인덱스
f_va = f[1:20:2]
l_va = l[1:20:2]
# 1차와 9차 회귀?
reg_b = np.polyfit(f_tr, l_tr, deg=1)
reg_v = np.polyfit(f_tr, l_tr, deg=9, full=True)[0]
# 편향과 분산...1차와 9차 회귀에 대해서, 훈련과 검증 데이터에 대해서...
# 1차 회귀는 R2낮고 편향 크지만 훈련과 검증 데이터에 차이가 없다...
evaluate(reg_b, f_tr, l_tr)
evaluate(reg_b, f_va, l_va)
# 9차 회귀는 훈련 데이터에 R2가 100%가 나온다..근데 검증 데이터는 -150...분산도 크다...
# 근데 9차 이상을 써보면 잘 맞을거 같기도 한데?
evaluate(reg_v, f_tr, l_tr)
evaluate(reg_v, f_va, l_va)
# 회귀 예측을 그리기 위한 x값...
f_ = np.linspace(f_tr.min(), f_va.max(), 75)
plt.figure(figsize=(10, 6))
plt.plot(f_tr, l_tr, "ro", label="training data")
plt.plot(f_va, l_va, "go", label="validation data")
plt.plot(f_, np.polyval(reg_b, f_), "--", label="high bias")
plt.plot(f_, np.polyval(reg_v, f_), "--", label="high variance")
plt.ylim(-0.2)
plt.legend(loc=2)
plt.show()
