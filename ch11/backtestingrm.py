#
# Event-Based Backtesting
# --Base Class (2)
#
# (c) Dr. Yves J. Hilpisch
#
from backtesting import *


class BacktestingBaseRM(BacktestingBase):

    def set_prices(self, price):
        # 진입 가격, 진입 후 최저와 최대 가격 기록...
        self.entry_price = price
        self.min_price = price
        self.max_price = price

    def place_buy_order(self, bar, amount=None, units=None, gprice=None):
        # 매수 주문인데, 원래 클래스와 다른건 gprice와 진입 가격 기록...
        date, price = self.get_date_price(bar)
        # 이게 뭐냐?
        if gprice is not None:
            price = gprice
        if units is None:
            units = int(amount / price)
        self.current_balance -= (1 + self.ptc) * units * price + self.ftc
        self.units += units
        self.trades += 1
        # 진입 가격 기록...
        self.set_prices(price)
        if self.verbose:
            print(f"{date} | buy {units} units for {price:.4f}")
            self.print_balance(bar)

    def place_sell_order(self, bar, amount=None, units=None, gprice=None):
        # 매도 주문, 원래와 다른건 gprice와 진입 가격 기록...
        date, price = self.get_date_price(bar)
        if gprice is not None:
            price = gprice
        if units is None:
            units = int(amount / price)
        self.current_balance += (1 - self.ptc) * units * price - self.ftc
        self.units -= units
        self.trades += 1
        self.set_prices(price)
        if self.verbose:
            print(f"{date} | sell {units} units for {price:.4f}")
            self.print_balance(bar)
