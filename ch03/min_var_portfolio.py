import numpy as np

# 오늘의 주식, 채권 가격
S0 = 10
B0 = 10
# 내일의 가능한 주식과 채권 가격 집합 - 이걸 페이오프라고 하는 모양...
S1 = np.array((20, 5))
B1 = np.array((11, 11))
# 시장 가격 벡터라고...현재 가격 배열이...
M0 = np.array((S0, B0))
print("M0", M0)
# 시장 페이오프 행렬
M1 = np.array((S1, B1)).T
print("M1", M1)

# 먼저 포트폴리오의 기대수익률을 구하는데...
print("기대수익률 구하기...")
# 위험 자산 주식의 수익률 벡터
rS = S1 / S0 - 1
print("rS", rS)
# 무위험 자산 채권의 수익률 벡터
rB = B1 / B0 - 1
print("rB", rB)
# 확률인데 반반...
P = np.array((0.5, 0.5))


# 기대수익률...수익률에 확률 곱하면 기대수익률
def mu(rX):
    return np.dot(P, rX)


# 주식과 채권 각각의 기대수익률
print("mu(rS)", mu(rS))
print("mu(rB)", mu(rB))
# 수익률을 행렬로
rM = M1 / M0 - 1
print("rM", rM)
# 이걸 따로따로 하지말고 그냥 행렬로 처리하면...포트폴리오의 기대수익률 벡터
print("mu(rM)", mu(rM))


# 다음으로 분산 구하기...
print("분산 구하기...")


# 분산 함수인데...수익률의 분산...
def var(rX):
    return ((rX - mu(rX)) ** 2).mean()


print("var(rS)", var(rS))
print("var(rB)", var(rB))


# 변동성 함수라는데...사실은 표준편차 정도...
def sigma(rX):
    return np.sqrt(var(rX))


print("sigma(rS)", sigma(rS))
print("sigma(rB)", sigma(rB))
# 공분산
print("공분산", np.cov(rM.T, aweights=P, ddof=0))

# 모든 가중치가 같은 포트폴리오의 수익률, 분산, 변동성 구하기...
print("모든 가중치가 같은 포트폴리오의 수익률, 분산, 변동성 구하기...")
# 이게 포트폴리오...결국 포트폴리오란 자산 벡터에 대한 가중치 벡터라는 얘기...
phi = np.array((0.5, 0.5))


# 이건 포트폴리오 받는 경우의 기대수익률 함수고...
def mu_phi(phi):
    return np.dot(phi, mu(rM))


print("mu_phi(phi)", mu_phi(phi))


# 이건 포트폴리오 받는 경우의 분산...
def var_phi(phi):
    # 공분산 구해서 그걸 가중치랑 두 번 닷곱하면 분산이 되는 모양...
    cv = np.cov(rM.T, aweights=P, ddof=0)
    return np.dot(phi, np.dot(cv, phi))


print("var_phi(phi)", var_phi(phi))


# 포트폴리오 받는 경우의 변동성...
def sigma_phi(phi):
    return var_phi(phi) ** 0.5


print("sigma_phi(phi)", sigma_phi(phi))

from pylab import plt, mpl

plt.style.use("seaborn-v0_8")
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "serif"

# 투자 기회 집합이라는 걸 만들어 보는데...
print("투자 기회 집합 만들기...")
# 랜덤 가중치 200개, 각 가중치는 주식/채권 2개로, 그 합은 1이 되야 하니까...
phi_mcs = np.random.random((2, 200))
phi_mcs = (phi_mcs / phi_mcs.sum(axis=0)).T
print("phi_mcs", phi_mcs.shape)
# 랜덤 포트폴리오들에 대한 변동성과 기대수익률 벡터
mcs = np.array([(sigma_phi(phi), mu_phi(phi)) for phi in phi_mcs])
# 2개 자산에 대한 투자 기회는 직선으로...
plt.figure(figsize=(10, 6))
plt.plot(mcs[:, 0], mcs[:, 1], "ro")
plt.xlabel("expected volatility")
plt.ylabel("expected return")
plt.show()

# 상승, 보합, 하락 세 가지를 가지는 정적 경제모델...
# 확률은 마찬가지로 같게...
P = np.ones(3) / 3
print("P", P)
# 거래 가능 자산은 위험자산만 2
S1 = np.array((20, 10, 5))
T0 = 10
T1 = np.array((1, 12, 13))
M0 = np.array((S0, T0))
print("M0", M0)
M1 = np.array((S1, T1)).T
print("M1", M1)
rM = M1 / M0 - 1
print("rM", rM)
# mcs는 몬테카를로 시뮬레이션?
mcs = np.array([(sigma_phi(phi), mu_phi(phi)) for phi in phi_mcs])
# 2개 자산인데 시나리오가 3개면 총알 모양이 된다...
plt.figure(figsize=(10, 6))
plt.plot(mcs[:, 0], mcs[:, 1], "ro")
plt.xlabel("expected volatility")
plt.ylabel("expected return")
plt.show()

# 최소 변동성 및 최대 샤프 비율 포트폴리오 도출하기...
print("최소 변동성 및 최대 샤프 비율 포트폴리오 도출하기...")
cons = {"type": "eq", "fun": lambda phi: np.sum(phi) - 1}
bnds = ((0, 1), (0, 1))

from scipy.optimize import minimize

# 포트폴리오 변동성 최소화...
min_var = minimize(sigma_phi, (0.5, 0.5), constraints=cons, bounds=bnds)
print("min_var", min_var)


# 샤프 비율 함수
def sharpe(phi):
    return mu_phi(phi) / sigma_phi(phi)


# 음의 샤프를 최소화하면 샤프를 최대화...
max_sharpe = minimize(
    lambda phi: -sharpe(phi), (0.5, 0.5), constraints=cons, bounds=bnds
)
print("max_sharpe", max_sharpe)

plt.figure(figsize=(10, 6))
plt.plot(mcs[:, 0], mcs[:, 1], "ro", ms=5)
plt.plot(
    sigma_phi(min_var["x"]),
    mu_phi(min_var["x"]),
    "^",
    ms=12.5,
    label="minimum volatility",
)
plt.plot(
    sigma_phi(max_sharpe["x"]),
    mu_phi(max_sharpe["x"]),
    "v",
    ms=12.5,
    label="maximum Sharpe ratio",
)
plt.xlabel("expected volatility")
plt.ylabel("expected return")
plt.legend()
plt.show()

# 리스크 대비 최대 기대수익 포트폴리오 -
# 위에서 나오는 그림에서 왼쪽 최소 위험 포트폴리오보다 낮으면 비효율이라고...
print("리스크 대비 최대 기대수익 포트폴리오...................")
# 제한 조건을 기대수익률 목표값으로 설정한다? 이거 뭘 어떻게 넣는지는 minimize를 공부해봐야 알겠네...
cons = [
    {"type": "eq", "fun": lambda phi: np.sum(phi) - 1},
    {"type": "eq", "fun": lambda phi: mu_phi(phi) - target},
]
bnds = ((0, 1), (0, 1))
# 이게 목표 수익률 집합이라고...최소 위험 기대 수익률 이상...0.16까지...
targets = np.linspace(mu_phi(min_var["x"]), 0.16)
frontier = []
# 목표 수익률에 대해서 돌면서
for target in targets:
    # 변동성이 가장 낮은 포트폴리오를 찾고
    phi_eff = minimize(sigma_phi, (0.5, 0.5), constraints=cons, bounds=bnds)["x"]
    # 그 때 변동성과 기대수익률 기록
    frontier.append((sigma_phi(phi_eff), mu_phi(phi_eff)))
frontier = np.array(frontier)
plt.figure(figsize=(10, 6))
plt.plot(frontier[:, 0], frontier[:, 1], "mo", ms=5, label="efficient frontier")
plt.plot(
    sigma_phi(min_var["x"]),
    mu_phi(min_var["x"]),
    "^",
    ms=12.5,
    label="minimum volatility",
)
plt.plot(
    sigma_phi(max_sharpe["x"]),
    mu_phi(max_sharpe["x"]),
    "v",
    ms=12.5,
    label="maximum Sharpe ratio",
)
plt.xlabel("expected volatility")
plt.ylabel("expected return")
plt.legend()
plt.show()
