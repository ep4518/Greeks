from __future__ import annotations
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import plotly.express as px
# import matplotlib.pyplot as plt
import BS as bs
from typing import TypeAlias, Callable, Any
from dataclasses import dataclass
from functools import partial
from abc import ABC, abstractmethod

# type bsParam = NDArray | float
bsParam: TypeAlias = NDArray | float


class FunctionalFunction(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __add__(self, other):
        def summed(*args, **kwargs):
            return self(*args, **kwargs) + other(*args, **kwargs)
        return summed

    def __mul__(self, other):
        def composed(*args, **kwargs):
            return self(other(*args, **kwargs))
        return composed

    def __iadd__(self, other):
        return self.__add__(other)

    def __imul__(self, other):
        return self.__mul__(other)


K = 1.0
T = 0.01
r = 0.0
q = 0.0
sigma = 0.2

Ku = 0.73
Kl = 0.70

S = np.linspace(20, 300, 281) / 100


@dataclass
class Equity:
    q: float
    sigma: float


call, put, put_spread = [pd.DataFrame(index=S) for _ in range(3)]
call['V'] = bs.C(S, K, T, r, q, sigma)
call['delta'] = bs.deltaC(S, K, T, r, q, sigma)
call['gamma'] = bs.gamma(S, K, T, r, q, sigma)
call['vega'] = bs.vega(S, K, T, r, q, sigma)
call['theta'] = bs.thetaC(S, K, T, r, q, sigma)
call['rho'] = bs.rhoC(S, K, T, r, q, sigma)


put['P'] = bs.P(S, K, T, r, q, sigma)
put['delta'] = bs.deltaP(S, K, T, r, q, sigma)
put['gamma'] = bs.gamma(S, K, T, r, q, sigma)
put['vega'] = bs.vega(S, K, T, r, q, sigma)
put['theta'] = bs.thetaP(S, K, T, r, q, sigma)
put['rho'] = bs.rhoP(S, K, T, r, q, sigma)


put_spread['V'] = 10 * (bs.P(S, Kl * K, T, r, q, sigma) -
                        bs.P(S, Ku * K, T, r, q, sigma))
put_spread['delta'] = 10 * (bs.deltaC(S, Kl * K, T, r, q, sigma) -
                            bs.deltaC(S, Ku * K, T, r, q, sigma))
put_spread['gamma'] = 10 * (bs.gamma(S, Kl * K, T, r, q, sigma) -
                            bs.gamma(S, Ku * K, T, r, q, sigma))
put_spread['vega'] = 10 * (bs.vega(S, Kl * K, T, r, q, sigma) -
                           bs.vega(S, Ku * K, T, r, q, sigma))
put_spread['theta'] = 10 * (bs.thetaC(S, Kl * K, T, r, q, sigma) -
                            bs.thetaC(S, Ku * K, T, r, q, sigma))
put_spread['rho'] = 10 * (bs.rhoC(S, Kl * K, T, r, q, sigma) -
                          bs.rhoC(S, Ku * K, T, r, q, sigma))

px.line(call, title=f'Call K: {K}, T: {T}, r: {
        r}, q: {q}, sigma: {sigma}').show()
px.line(put, title=f'Put K: {K}, T: {T}, r: {
        r}, q: {q}, sigma: {sigma}').show()
px.line(put_spread,
        title=f'Put Spread 73-70, T: {T}, r: {r}, q: {q}, sigma: {sigma}').show()


# x = S
# y = np.linspace(0, 1, 101)
# xx, yy = np.meshgrid(x, y, sparse=True)
#
# test = bs.C(xx, K, T, r, q, sigma)
