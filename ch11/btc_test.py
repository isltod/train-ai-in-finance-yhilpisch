import os
import numpy as np
import pandas as pd
from pylab import plt, mpl
import sys

sys.path.append(".")
from util import Timer
import market
import tradingbot
import tbbacktesterrm as tbbrm


def clean_btc_csv(csv):
    df = pd.read_csv(csv)
    print(df.columns)

    # # 밀리초(ms) 단위 타임스탬프를 datetime으로 변환
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms")
    colums = ["dt", "open", "high", "low", "close", "volume"]
    df = df[colums]
    df.set_index("dt", inplace=True)
    df.to_csv(csv)


if __name__ == "__main__":
    btc_csvs = {"1h": "data/btc_usdt_1h_cache.csv", "1m": "data/btc_usdt_1m_cache.csv"}
    # btc 데이터를 이 책 예제 구조와 비슷하게 변경
    # for key in btc_csvs:
    #     clean_btc_csv(btc_csvs[key])

    symbol = "close"
    # 이것도 Market 클래스 내부의 로직과 하드코딩으로 연결되어 있어서, 바꾸려면 같이 수정해야 함...
    features = [symbol, "r", "s", "m", "v"]
    csv = btc_csvs["1h"]
    a = 0.0
    b = 0.8
    c = 0.9
    learn_env = market.Market(
        csv,
        symbol,
        features,
        window=20,
        # 시간 쉬프트를 3칸밖에 안 했던가...
        lags=3,
        leverage=1,
        min_performance=0.9,
        min_accuracy=0.475,
        start=a,
        end=a + b,
        mu=None,
        std=None,
    )
    env = learn_env
    print("환경 초기화", env.reset())
    a = env.action_space.sample()
    print("행위 선택", a)
    print("행위 결과", env.step(a))
