import pandas as pd
import numpy as np
from pylab import plt
import sys

sys.path.append(".")
from util import Timer
from data import get_data, cw, set_seeds, create_model

data, cols, train, test = get_data()
mu, std = train.mean(), train.std()
train_ = (train - mu) / std
test_ = (test - mu) / std

set_seeds()

# 여기선 이렇게 규제를 사용하는데...
from keras.regularizers import l1, l2

# 규제만, 그것도 L1 규제만 사용하니 과적합 문제가 별로...
model = create_model(hl=1, hu=128, cols=cols, regularize=True, reg=l1(0.0005))

with Timer():
    hist = model.fit(
        train_[cols],
        train["d"],
        epochs=50,
        verbose=False,
        validation_split=0.2,
        shuffle=False,
        class_weight=cw(train),
    )
print("훈련 데이터 결과 평가\n", model.evaluate(train_[cols], train["d"]))
print("시험 데이터 결과 평가\n", model.evaluate(test_[cols], test["d"]))
res = pd.DataFrame(hist.history)
res[["accuracy", "val_accuracy"]].plot(figsize=(10, 6), style="--")
plt.show()

set_seeds()
# 그런데 정규화, 드롭아웃, L2 규제를 사용하면 책은 과적합이 없어지는데, 소스코드는 아니고...
# 나는 약간 줄어드는 것 같다...훈련 75, 시험 65% 정도로...
model = create_model(
    hl=2,
    hu=128,
    cols=cols,
    dropout=True,
    rate=0.3,
    regularize=True,
    reg=l2(0.001),
)


with Timer():
    hist = model.fit(
        train_[cols],
        train["d"],
        epochs=50,
        verbose=False,
        validation_split=0.2,
        shuffle=False,
        class_weight=cw(train),
    )
print("훈련 데이터 결과 평가\n", model.evaluate(train_[cols], train["d"]))
print("시험 데이터 결과 평가\n", model.evaluate(test_[cols], test["d"]))
res = pd.DataFrame(hist.history)
res[["accuracy", "val_accuracy"]].plot(figsize=(10, 6), style="--")
plt.show()
