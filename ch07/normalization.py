import pandas as pd
import numpy as np
from pylab import plt
import sys

sys.path.append(".")
from util import Timer
from data import get_data, cw, set_seeds, create_model

data, cols, train, test = get_data()

# 이렇게 간단하게 정규화라는 걸 하면 훈련 데이터 정확도는 많이 올라간다...시험 데이터도 65% 정도로...
mu, std = train.mean(), train.std()
train_ = (train - mu) / std

set_seeds()
model = create_model(hl=2, hu=128, cols=cols)
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
test_ = (test - mu) / std
print("시험 데이터 결과 평가\n", model.evaluate(test_[cols], test["d"]))
test["p"] = np.where(model.predict(test_[cols]) > 0.5, 1, 0)
print("시험 데이터 상승과 하락 예측\n", test["p"].value_counts())
res = pd.DataFrame(hist.history)
# 근데 훈련 데이터 정확도가 90을 넘어가며 올라가는 동안 시험 데이터가 정체되는 과적합 문제...
res[["accuracy", "val_accuracy"]].plot(figsize=(10, 6), style="--")
plt.show()
