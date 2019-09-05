"""Derivatives library."""

import abc
import enum
import math
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scsp
import scipy.stats as scs


def e(value: float) -> float:
    return math.exp(value)


class BuyType(enum.Enum):
    LONG = 0
    SHORT = 1

    DEFAULT = LONG


class Investment(abc.ABC):
    def __init__(
        self,
        cost: float = 0,
        buy_type: BuyType = BuyType.DEFAULT,
    ) -> None:
        self.buy_type = buy_type
        self.cost = cost

        self.type = 'Investment'

    def _short_sell(self) -> None:
        if self.cost > 0 and self.buy_type is BuyType.SHORT:
            self.cost = -self.cost

    def position_type(self) -> str:
        return 'long' if self.buy_type is BuyType.LONG else 'short'

    def name(self) -> str:
        return f'{self.type} ({self.position_type()})'

    @abc.abstractmethod
    def raw_payoff(self, value: float) -> float:
        raise NotImplementedError()

    def payoff(self, value: float) -> float:
        return (
            self.raw_payoff(value)
            if self.buy_type is BuyType.LONG else
            -self.raw_payoff(value))

    def profit(self, value: float) -> float:
        return self.payoff(value) - self.cost

    def key_values(self) -> List[float]:
        return []


class Stock(Investment):
    def __init__(
        self,
        cost: float,
        interest: float = 0,
        buy_type: BuyType = BuyType.DEFAULT,
    ) -> None:
        super().__init__(cost, buy_type)

        self.type = 'Stock'
        self.interest = interest

        if cost > 0 and self.buy_type is BuyType.SHORT:
            self.cost = -cost

    def raw_payoff(self, value: float) -> float:
        return value * (1 + self.interest)

    def key_values(self) -> List[float]:
        return []


class Put(Investment):
    def __init__(
        self,
        strike: float,
        cost: float = 0,
        buy_type: BuyType = BuyType.DEFAULT,
    ) -> None:
        super().__init__(cost, buy_type)

        self.strike = strike
        self.type = 'Put'
        self._short_sell()

    def raw_payoff(self, value: float) -> float:
        if value < self.strike:
            return self.strike - value
        else:
            return 0.0

    def key_values(self) -> List[float]:
        return [self.strike]


class Call(Investment):
    def __init__(
        self,
        strike: float,
        cost: float = 0,
        buy_type: BuyType = BuyType.DEFAULT,
    ) -> None:
        super().__init__(cost, buy_type)

        self.strike = strike
        self.type = 'Call'
        self._short_sell()

    def raw_payoff(self, value: float) -> float:
        if value > self.strike:
            return value - self.strike
        else:
            return 0.0

    def key_values(self) -> List[float]:
        return [self.strike]


class Pure(Investment):
    def __init__(
        self, cost: float = 0, buy_type: BuyType = BuyType.DEFAULT,
    ) -> None:
        super().__init__(cost, buy_type)

        self.type = 'Pure'

    def raw_payoff(self, value: float) -> float:
        return self.cost

    def key_values(self) -> List[float]:
        return []


class Portfolio:
    def __init__(self) -> None:
        self.invs: List[Tuple(Investment, int)] = []

    def add(self, inv: Investment, count: int = 1) -> None:
        self.invs.append((inv, count))

    def payoff(self, value: float) -> float:
        return sum(inv.payoff(value) * count for inv, count in self.invs)

    def profit(self, value: float) -> float:
        return sum(inv.profit(value) * count for inv, count in self.invs)

    def plot(
        self,
        x_range: Optional[Tuple[float, float]] = None,
        n: int = 100,
        **kwargs,
    ) -> Any:

        offset_ratio = 0.5
        alpha = 0.7
        min_offset = 0.5

        if x_range is None:
            # Determine key values to find upper and lower bound.
            key_values = sum((inv.key_values() for inv, _ in self.invs), [])
            if not key_values:
                key_values = [0, 100]
            offset = max(
                offset_ratio * (max(key_values) - min(key_values)), min_offset)
            xs = np.linspace(
                min(key_values) - offset, max(key_values) + offset, n)
        else:
            xs = np.linspace(*x_range, n)

        # Plot each individual item.
        if bool(kwargs.get('lines', True)):
            for inv, count in self.invs:
                ys = [inv.payoff(x) * count for x in xs]
                label = inv.name() if count == 1 else f'{inv.name()} x{count}'
                plt.plot(xs, ys, '--', label=label, alpha=alpha)

        # Plot payoff.
        if bool(kwargs.get('payoff', True)):
            ys_payoff = [self.payoff(x) for x in xs]
            plt.plot(xs, ys_payoff, 'g', label='Payoff')

        # Plot profit.
        if bool(kwargs.get('profit', True)):
            ys_profit = [self.profit(x) for x in xs]
            plt.plot(xs, ys_profit, 'b', label='Profit')

        plt.title('Payoff and profit chart')
        plt.ylabel('Payoff/profit ($)')
        plt.xlabel('Value ($)')
        plt.legend()
        plt.grid(True)

        return plt.show()


def pv(
    n: int,
    coupon: float,
    rate: float,
    face: float,
    continuous: bool = False,
) -> float:
    if continuous:
        apply_rate = lambda rate, t: math.exp(rate * t)
    else:
        apply_rate = lambda rate, t: (1 + rate) ** t
    return sum(
        coupon * face / apply_rate(rate, t) for t in range(1, n + 1)
    ) + face / apply_rate(rate, n)


def black_scholes(
    price: float,
    strike: float,
    rate: float,
    vol: float,
    time: float,
    dividend: float,
) -> float:
    n = scs.norm()
    d1 = (
        (math.log(price / strike) + (rate + vol ** 2 / 2) * time) /
        (vol * math.sqrt(time)))
    d2 = d1 - vol * math.sqrt(time)

    return {
        'call':
            price * math.exp(-dividend * time) * n.cdf(d1) -
            strike * math.exp(-rate * time) * n.cdf(d2),
        'put':
            strike * math.exp(-rate * time) * n.cdf(-d2) -
            price * math.exp(-dividend * time) * n.cdf(-d1),
    }


def find_zero(
    func: Callable[[float], float],
    guess_low: float,
    guess_high: float,
    epsilon: float = 0.005,
) -> float:
    low = guess_low
    high = guess_high
    guess = (high + low) / 2

    if np.sign(func(low)) == np.sign(func(high)):
        raise ValueError('Intersection not guaranteed.')

    while abs(func(guess)) > epsilon:
        low_sign = np.sign(func(low))
        guess_sign = np.sign(func(guess))
        high_sign = np.sign(func(high))

        # Search right side first.
        if low_sign != guess_sign:
            high = guess
        elif guess_sign != high_sign:
            low = guess
        else:
            raise ValueError('Failed to find solution.')

        guess = (high + low) / 2

    return guess


def binom_price_call(
    u: float, d: float, R: float, S: float, K: float, n: int, T: float = 1.0,
) -> float:
    p = (R / T - d) / (u - d)
    return sum(
        scsp.comb(n, j) * p ** j * (1 - p) ** (n - j) *
        max(0, u ** j * d ** (n - j) * S - K)
        for j in range(0, n + 1)
    ) / R ** n


def binom_price_put(
    u: float, d: float, R: float, S: float, K: float, n: int, T: float = 1.0,
) -> float:
    p = (R / T - d) / (u - d)
    return sum(
        scsp.comb(n, j) * p ** j * (1 - p) ** (n - j) *
        max(0, K - u ** j * d ** (n - j) * S)
        for j in range(0, n + 1)
    ) / R ** n


def binom_delta_b(
    c_u: float, c_d: float, u: float, d: float, R: float, S: float,
) -> float:
    return {
        'Delta': (c_u - c_d) / (S * (u - d)),
        'B': (u * c_d - d * c_u) / (R * (u - d)),
    }
