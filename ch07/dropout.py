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
# 여기서 드롭아웃을 30%주면 훈련과 시험 데이터에 대한 성능이 다 떨어진다...
# 그리고 과적합도 그리 좋아지질 않는데...
model = create_model(hl=1, hu=128, cols=cols, dropout=True, rate=0.3)
with Timer():
    hist = model.fit(
        train_[cols],
        train["d"],
        epochs=50,
        verbose=False,
        validation_split=0.15,
        shuffle=False,
        class_weight=cw(train),
    )
print("훈련 데이터 결과 평가\n", model.evaluate(train_[cols], train["d"]))
print("시험 데이터 결과 평가\n", model.evaluate(test_[cols], test["d"]))
res = pd.DataFrame(hist.history)
res[["accuracy", "val_accuracy"]].plot(figsize=(10, 6), style="--")
plt.show()
