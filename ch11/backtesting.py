#
# Event-Based Backtesting
# --Base Class (1)
#
# (c) Dr. Yves J. Hilpisch
# Artificial Intelligence in Finance
#
import numpy as np
import pandas as pd


class BacktestingBase:
    def __init__(self, env, model, amount, ptc, ftc, verbose=False):
        self.env = env
        self.model = model
        self.initial_amount = amount
        self.current_balance = amount
        self.ptc = ptc
        self.ftc = ftc
        self.verbose = verbose
        # 이게 보유 수량인 모양...
        self.units = 0
        self.trades = 0

    # 몇 번째 봉인지 받아서, 날짜와 가격을 반환..
    def get_date_price(self, bar):
        # 날짜 까지만 추리는데...이건 시간이나 분봉으로 하려면 다 써야 할텐데...
        date = str(self.env.data.index[bar])[:10]
        # 가격은 iloc 인덱스로 찾으니 상관없고...
        price = self.env.data[self.env.symbol].iloc[bar]
        return date, price

    # 현재 잔고를 출력...
    def print_balance(self, bar):
        date, _ = self.get_date_price(bar)
        print(f"{date} | current balance = {self.current_balance:.2f}")

    # 잔고 + 보유수량 * 가격
    def calculate_net_wealth(self, price):
        return self.current_balance + self.units * price

    # 몇 번째 봉인지 받아서 평가 총 자산 출력
    def print_net_wealth(self, bar):
        date, price = self.get_date_price(bar)
        net_wealth = self.calculate_net_wealth(price)
        print(f"{date} | net wealth = {net_wealth:.2f}")

    # 매수 주문이겠지...몇 번째 봉인지 받아서...
    def place_buy_order(self, bar, amount=None, units=None):
        date, price = self.get_date_price(bar)
        # amount는 총 금액? 수량 또는 총금액 중 하나는 있어야겠고...아무튼 수량을 만들어서...
        if units is None:
            units = int(amount / price)
            # units = amount / price  # alternative handling
        # ptc와 ftc는 뭐냐? 현재 잔고에서 빼고
        self.current_balance -= (1 + self.ptc) * units * price + self.ftc
        # 보유 수량은 늘리고
        self.units += units
        # 이건 거래 횟수인가...
        self.trades += 1
        if self.verbose:
            print(f"{date} | buy {units} units for {price:.4f}")
            self.print_balance(bar)

    # 이건 매도겠지....
    def place_sell_order(self, bar, amount=None, units=None):
        date, price = self.get_date_price(bar)
        if units is None:
            units = int(amount / price)
            # units = amount / price  # altermative handling
        self.current_balance += (1 - self.ptc) * units * price - self.ftc
        self.units -= units
        # 거래 횟수를 팔 때도 기록하네...선물 방식인듯...
        self.trades += 1
        if self.verbose:
            print(f"{date} | sell {units} units for {price:.4f}")
            self.print_balance(bar)

    # 모든 포지션 종료하는 모양...
    def close_out(self, bar):
        date, _ = self.get_date_price(bar)
        print(50 * "=")
        print(f"{date} | *** CLOSING OUT ***")
        # 보유 수량이 음수가 되는걸 보니 선물 방식인듯...
        if self.units < 0:
            self.place_buy_order(bar, units=-self.units)
        else:
            self.place_sell_order(bar, units=self.units)
        # 잉? verbose여야 출력해야 맞는거 아닌가?
        if not self.verbose:
            print(f"{date} | current balance = {self.current_balance:.2f}")
        # 순 수익률
        perf = (self.current_balance / self.initial_amount - 1) * 100
        print(f"{date} | net performance [%] = {perf:.4f}")
        print(f"{date} | number of trades [#] = {self.trades}")
        print(50 * "=")


class TBBacktester(BacktestingBase):
    def _reshape(self, state):
        # 상태 벡터를 (배치, 쉬프트 수, 컬럼 수) 벡터로 변환...
        return np.reshape(state, [1, self.env.lags, self.env.n_features])

    def backtest_strategy(self):
        # 일단 속성들을 초기화하고...
        self.units = 0
        self.position = 0
        self.trades = 0
        self.current_balance = self.initial_amount
        self.net_wealths = list()

        # 쉬프트 시간만큼은 평가할 수 없으니, 첫 번째 봉은 쉬프트 수
        bar1st = self.env.lags

        # 거래 시작 알리고
        date1st, _ = self.get_date_price(bar1st)
        print(50 * "=")
        print(f"{date1st} | *** START BACKTEST ***")
        self.print_balance(bar1st)
        print(50 * "=")

        # 쉬프트 수 이후부터 데이터 있는 동안 반복해서...
        for bar in range(bar1st, len(self.env.data)):
            # 날짜와 가격 받고
            date, price = self.get_date_price(bar)
            # 해당 봉 이전의 쉬프트 수 만큼의 가격 등 정보 받고,
            state = self.env.get_state(bar)
            # 모델에서 행동 예측 받아서
            action = np.argmax(self.model.predict(self._reshape(state.values))[0, 0])
            # 예측 포지션을 행동에 따라 정의하고
            position = 1 if action == 1 else -1
            # 현재 보유 포지션과 예측 포지션이 다르면 거래...
            if self.position in [0, -1] and position == 1:
                # 이건 롱 예측 경우인데...
                if self.verbose:
                    print(50 * "-")
                    print(f"{date} | *** GOING LONG ***")
                # 현재 포지션이 숏이었으면, 다 팔고
                if self.position == -1:
                    self.place_buy_order(bar - 1, units=-self.units)
                # 현재 잔고 올인해서 롱 진입...
                self.place_buy_order(bar - 1, amount=self.current_balance)
                if self.verbose:
                    self.print_net_wealth(bar)
                # 현재 포지션 업데이트...
                self.position = 1
            elif self.position in [0, 1] and position == -1:
                if self.verbose:
                    print(50 * "-")
                    print(f"{date} | *** GOING SHORT ***")
                if self.position == 1:
                    self.place_sell_order(bar - 1, units=self.units)
                self.place_sell_order(bar - 1, amount=self.current_balance)
                if self.verbose:
                    self.print_net_wealth(bar)
                self.position = -1
            # 잔고 변화를 기록...
            self.net_wealths.append((date, self.calculate_net_wealth(price)))
        # 잔고 변화를 데이터프레임으로 만들어서 저장...
        self.net_wealths = pd.DataFrame(
            self.net_wealths, columns=["date", "net_wealth"]
        )
        self.net_wealths.set_index("date", inplace=True)
        self.net_wealths.index = pd.DatetimeIndex(self.net_wealths.index)
        # 남아있는 포지션 있으면 다 정리...
        self.close_out(bar)
