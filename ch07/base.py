import os
import numpy as np
import pandas as pd
from pylab import plt
import sys

sys.path.append(".")
from util import Timer
from data import get_data, cw, create_model, set_seeds

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"

data, cols, train, test = get_data()

set_seeds()
model = create_model(hl=1, hu=128, cols=cols)

with Timer():
    model.fit(data[cols], data["d"], epochs=50, verbose=False, class_weight=cw(data))

print("훈련 데이터 결과 평가\n", model.evaluate(data[cols], data["d"]))
# p는 0.5 이상으로 예측되면 상승 1, 아니면 하락 0
data["p"] = np.where(model.predict(data[cols]) > 0.5, 1, 0)
print("예측 결과 상승과 하락 수\n", data["p"].value_counts())

# 이번에는 훈련/시험 데이터 나눠서...
set_seeds()
model = create_model(hl=1, hu=128, cols=cols)
with Timer():
    hist = model.fit(
        train[cols],
        train["d"],
        epochs=50,
        verbose=False,
        validation_split=0.2,
        shuffle=False,
        class_weight=cw(train),
    )

print("훈련 데이터 결과 평가\n", model.evaluate(train[cols], train["d"]))
print("시험 데이터 결과 평가\n", model.evaluate(test[cols], test["d"]))
test["p"] = np.where(model.predict(test[cols]) > 0.5, 1, 0)
print("시험 데이터 상승과 하락 예측\n", test["p"].value_counts())
res = pd.DataFrame(hist.history)
res[["accuracy", "val_accuracy"]].plot(figsize=(10, 6), style="--")
plt.show()
