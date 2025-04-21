from numpy import log, sqrt, exp
from scipy.stats import norm
from numpy.typing import NDArray
from typing import TypeAlias

bsParam: TypeAlias = NDArray | float

N = norm.cdf
psi = norm.pdf


def d1(S: bsParam, K: float, T: bsParam, r: float, q: float, sigma: float):
    return (log(S / K) + (r - q + .5 * sigma ** 2) * T) / (sigma * sqrt(T))


def d2(S: bsParam, K: float, T: bsParam, r: float, q: float, sigma: float):
    return d1(S, K, T, r, q, sigma) - sigma * sqrt(T)


def C(S: bsParam, K: float, T: bsParam, r: float, q: float, sigma: float):
    return S * exp(-q * T) * N(d1(S, K, T, r, q, sigma)) - \
        K * exp(-r * T) * N(d2(S, K, T, r, q, sigma))


def P(S: bsParam, K: float, T: bsParam, r: float, q: float, sigma: float):
    return K * exp(-r * T) * N(-d2(S, K, T, r, q, sigma)) - \
        S * exp(-q * T) * N(-d1(S, K, T, r, q, sigma))


def deltaC(S: bsParam, K: float, T: bsParam, r: float, q: float, sigma: float):
    return exp(-q * T) * N(d1(S, K, T, r, q, sigma))


def deltaP(S: bsParam, K: float, T: bsParam, r: float, q: float, sigma: float):
    return -exp(-q * T) * N(-d1(S, K, T, r, q, sigma))


def gamma(S: bsParam, K: float, T: bsParam, r: float, q: float, sigma: float):
    return exp(-q * T) * psi(d1(S, K, T, r, q, sigma)) / (S * sigma * sqrt(T))


def vega(S: bsParam, K: float, T: bsParam, r: float, q: float, sigma: float):
    return -S * exp(-q * T) * psi(d1(S, K, T, r, q, sigma)) * sqrt(T)


def rhoC(S: bsParam, K: float, T: bsParam, r: float, q: float, sigma: float):
    return K * T * exp(-r * T) * N(d2(S, K, T, r, q, sigma))


def rhoP(S: bsParam, K: float, T: bsParam, r: float, q: float, sigma: float):
    return -K * T * exp(-r * T) * N(-d2(S, K, T, r, q, sigma))


def thetaC(S: bsParam, K: float, T: bsParam, r: float, q: float, sigma: float):
    d_1 = d1(S, K, T, r, q, sigma)
    return -S * sigma * exp(-q * T) * psi(d_1) / (2 * sqrt(T)) + \
        q * S * exp(-q * T) * N(d_1) - r * K * \
        exp(-r * T) * N(d2(S, K, T, r, q, sigma))


def thetaP(S: bsParam, K: float, T: bsParam, r: float, q: float, sigma: float):
    d_1 = d1(S, K, T, r, q, sigma)
    return -S * sigma * exp(-q * T) * psi(d_1) / (2 * sqrt(T)) - \
        q * S * exp(-q * T) * N(-d_1) + r * K * \
        exp(-r * T) * N(-d2(S, K, T, r, q, sigma))
