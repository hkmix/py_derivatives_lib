# Python derivatives library

Use at your own risk. Read function arguments for additional use cases.

Requires numpy, matplotlib, scipy.

## Examples

Assumes you have imported `derivatives` as `dv`.

Plot a payoff/profit graph for a collar:

```py
>>> collar = dv.Portfolio()
>>> collar.add(dv.Stock(20))
>>> collar.add(dv.Put(20, cost=5))
>>> collar.add(dv.Call(40, cost=4, buy_type=dv.BuyType.SHORT))
>>> collar.plot()
<<< # matplotlib.pyplot graph shows up here.
```

Calculate Black-Scholes:

```py
>>> dv.black_scholes(
...     price=100,
...     strike=120,
...     rate=0.03,
...     vol=0.20,
...     time=3,
...     dividend=0.02,
... )
<<< {'call': 7.388206108075657, 'put': 22.883494982198187}
```

Binary search for a zero:

```py
>>> dv.find_zero(
...     lambda x: (x - 3) * (x - 2),
...     guess_low=0,
...     guess_high = 2.5,
...     epsilon=0.001,
... )
<<< 1.99951171875
```
