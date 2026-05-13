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

from sklearn.ensemble import BaggingClassifier

# 이건 scikeras 호환 문제로 포기
from keras.wrappers.scikit_learn import KerasClassifier

# from scikeras.wrappers import KerasClassifier

len(cols)
max_features = 0.75
set_seeds()
base_estimator = KerasClassifier(
    model=create_model,
    verbose=False,
    epochs=20,
    hl=1,
    hu=128,
    dropout=True,
    regularize=False,
    input_dim=int(len(cols) * max_features),
)
model_bag = BaggingClassifier(
    estimator=base_estimator,
    n_estimators=15,
    max_samples=0.75,
    max_features=max_features,
    bootstrap=True,
    bootstrap_features=True,
    n_jobs=1,
    random_state=100,
)
with Timer():
    model_bag.fit(train_[cols], train["d"])
print("훈련 데이터 결과 평가\n", model_bag.score(train_[cols], train["d"]))
print("시험 데이터 결과 평가\n", model_bag.score(test_[cols], test["d"]))
test["p"] = model_bag.predict(test_[cols])
print("시험 데이터 상승과 하락 예측\n", test["p"].value_counts())
