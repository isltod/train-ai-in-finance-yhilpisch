import numpy as np

# 오늘의 주식, 채권 가격
S0 = 10

# 거래 가능 자산은 위험자산만 2
S1 = np.array((20, 10, 5))
T0 = 10
T1 = np.array((1, 12, 13))
M0 = np.array((S0, T0))
print("M0", M0)
M1 = np.array((S1, T1)).T

V1 = np.array((12, 15, 7))
# 최적 회귀 인수값을 요인 적재량으로 해석?
reg = np.linalg.lstsq(M1, V1, rcond=-1)[0]
print("reg", reg)

print("dot 2", np.dot(M1, reg))
# 두 개의 요인만으로는 페이오프 V1을 완전하게 설명(복제)할 수 없다...
print(np.dot(M1, reg) - V1)
# 증가한 시장가격 벡터?
V0 = np.dot(M0, reg)
print("V0", V0)

# 세 번째 리스크 요인
U0 = 10
U1 = np.array((12, 5, 11))
T0 = 10
M0_ = np.array((S0, T0, U0))
M1_ = np.concatenate(
    (
        M1.T,
        np.array(
            [
                U1,
            ]
        ),
    )
).T
print("M1_", M1_)

np.linalg.matrix_rank(M1_)
# 풀 랭크로 증가한 시장 페이오프 행렬?
reg = np.linalg.lstsq(M1_, V1, rcond=-1)[0]
print("reg", reg)
# V1의 정확한 복제...잔차는 0
np.allclose(np.dot(M1_, reg), V1)
# 위험자산 V에 대한 유일한 무차익거래?
V0_ = np.dot(M0_, reg)
print("V0_", V0_)
