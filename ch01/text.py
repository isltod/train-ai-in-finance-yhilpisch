import numpy as np
import pandas as pd
from pylab import plt, mpl
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 환경 설정들
plt.style.use("seaborn-v0_8")
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "serif"
np.set_printoptions(precision=4, suppress=True)

# 비지도 학습 - K-Means 클러스터링
# 가우시안 정규분포로 클러스터링 가상 데이터 생성, random_state는 seed 같은 거 같네..
# x는 x1, x2, y는 그룹
x, y = make_blobs(n_samples=100, centers=4, random_state=500, cluster_std=1.25)
model = KMeans(n_clusters=4, random_state=0, n_init="auto")
model.fit(x)
y_ = model.predict(x)
print(y_)
# plt.figure(figsize=(10, 6))
# plt.scatter(x[:, 0], x[:, 1], c=y_, cmap="coolwarm")
# plt.show()

# 강화학습
# 이게 그냥 무작위로 찍을 때이고...
# 이게 상태 공간이고
ssp = [1, 1, 1, 1, 0]
# 이건 행동 공간인 모양...
asp = [1, 0]


def epoch():
    tr = 0
    for _ in range(100):
        a = np.random.choice(asp)
        s = np.random.choice(ssp)
        if a == s:
            tr += 1
    return tr


rl = np.array([epoch() for _ in range(15)])
print(rl)
print(rl.mean())


# 이게 학습이 되는 거라는데...
def epoch():
    tr = 0
    asp = [0, 1]
    for _ in range(100):
        a = np.random.choice(asp)
        s = np.random.choice(ssp)
        if a == s:
            tr += 1
            # 여기서 학습이 된다는 거 같은데...논리상 if 문으로 들어가야 할 거 같은데...
            # 그래야 성공하면 그게 더해져서 성공했던 숫자를 뽑을 확률이 높아지는거 아닌가?
            asp.append(s)
    return tr


rl = np.array([epoch() for _ in range(15)])
print(rl)
print(rl.mean())
