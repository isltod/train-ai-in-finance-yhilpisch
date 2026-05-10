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
# 옵션의 행사가
K = 14.5
# 옵션의 불확실 페이오프라고...
C1 = np.maximum(S1 - K, 0)
print("C1", C1)
# c1 = M1 * x 방정식의 해...대충 c1/M1 정도... - 복제 포트폴리오...
phi = np.linalg.solve(M1, C1)
print("phi", phi)
# 해가 맞는지 검산
print(np.allclose(C1, np.dot(M1, phi)))
C0 = np.dot(M0, phi)
print("C0", C0)
