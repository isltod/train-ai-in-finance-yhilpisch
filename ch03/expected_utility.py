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


# 리스크 회피 베르누이 함수
def u(x):
    return np.sqrt(x)


# 서로 다른 가중치를 가진 두 포트폴리오...복제 포트폴리오...왜 복제인가...
phi_A = np.array((0.75, 0.25))
phi_D = np.array((0.25, 0.75))
# 이게 포트폴리오를 만드는 비용이라...암튼 같다...
print("Ca == Cd", np.dot(M0, phi_A) == np.dot(M0, phi_D))
# 이게 옵션의 불확실성 페이오프였지...A1
A1 = np.dot(M1, phi_A)
print("A1", A1)
D1 = np.dot(M1, phi_D)
print("D1", D1)
# 확률인데 반반...
P = np.array((0.5, 0.5))


# 이게 기대효용함수라...베르누이 효용 함수 값에 확률 곱하면 기대효용...
def EUT(x):
    return np.dot(P, u(x))


# 두 페이오프의 기대효용
print("EUT(A1)", EUT(A1))
print("EUT(D1)", EUT(D1))

from scipy.optimize import minimize

# 이게 예산
w = 10
# 밑에 minimize 함수에 넣을 제한조건...
cons = {"type": "eq", "fun": lambda phi: np.dot(M0, phi) - w}


# 위에서는 옵션의 불확실성 페이오프를 받는데, 여기선 복제 포트폴리오를 받는 식으로...기대효용함수
def EUT_(phi):
    x = np.dot(M1, phi)
    return EUT(x)


# 기대효용함수 값을 음수로...최소화란 기대효용값을 최대화한다는 얘기...
# x0는 초깃값, constraints는 제한조건...
opt = minimize(lambda phi: -EUT_(phi), x0=phi_A, constraints=cons)
print("opt", opt)
print("예산 배분", opt["x"])
print("EUT(opt[x])", EUT_(opt["x"]))
print(np.dot(M0, opt["x"]))
