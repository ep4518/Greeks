from numpy import sqrt, exp, log, pow
from scipy.stats import norm
from numpy.typing import NDArray
from typing import TypeAlias
from enum import Enum

bsParam: TypeAlias = NDArray | float

N = norm.cdf
phi = norm.pdf


"""
Reiner & Rubinstein (1991)* - Barrier Option Payoffs — Summary Table

This table describes the payoffs for various types of barrier options
(Down/Up, In/Out, Call/Put) depending on whether the strike (K) is less than or
greater than the barrier (H).

Notation:
- A: Plain vanilla call payoff = max(S - K, 0)
- B: Plain vanilla put payoff = max(K - S, 0)
- C: Rebate if barrier is hit
- D: Vanilla option activated when barrier is hit
- E: Immediate payoff upon hitting barrier (if any)
- F: Rebate if barrier is never hit

+--------------------------------------------------------------------+
| Type           | K ≤ B             | K ≥ B             | phi | eta |
+----------------+-------------------+-------------------+-----+-----+
| Down In Call   | A − B + D + E     | C + E             |  1  |  1  |
| Up In Call     | B − C + D + E     | A + E             |  1  | -1  |
| Down In Put    | A + E             | B − C + D + E     | -1  |  1  |
| Up In Put      | C + E             | A − B + D + E     | -1  | -1  |
| Down Out Call  | B − D + F         | A − C + F         |  1  |  1  |
| Up Out Call    | A − B + C − D + F | F                 |  1  | -1  |
| Down Out Put   | F                 | A − B + C − D + F | -1  |  1  |
| Up Out Put     | A − C + F         | B − D + F         | -1  | -1  |
+--------------------------------------------------------+-----------+

*E.G Haug and J.Haug. Resetting strikes, barriers and time.
Wilmott Magazine, 2001
"""


class indicCall(Enum):
    PUT = -1
    CALL = 1


class indicDown(Enum):
    UP = -1
    DOWN = 1


class indicBarrier(Enum):
    IN = -1
    OUT = 1


def CDI(
    S: bsParam,
    K: float,
    H: float,
    R: float,
    T: bsParam,
    r: float,
    q: float,
    sigma: float,
    phi: indicCall = indicCall.CALL,
    eta: indicDown = indicDown.DOWN
):
    if K < H:
        return A(S, K, H, T, r, q, sigma, phi) - \
            B(S, K, H, T, r, q, sigma, phi) + \
            D(S, K, H, T, r, q, sigma, phi, eta) + \
            E(S, K, H, R, T, r, q, sigma, eta)
    else:
        return C(S, K, H, T, r, q, sigma, phi, eta) + \
            E(S, K, H, R, T, r, q, sigma, eta)


def CUI(
    S: bsParam,
    K: float,
    H: float,
    R: float,
    T: bsParam,
    r: float,
    q: float,
    sigma: float,
    phi: indicCall = indicCall.CALL,
    eta: indicDown = indicDown.UP
):
    if K < H:
        return B(S, K, H, T, r, q, sigma, phi) - \
            C(S, K, H, T, r, q, sigma, phi, eta) + \
            D(S, K, H, T, r, q, sigma, phi, eta) + \
            E(S, K, H, R, T, r, q, sigma, eta)
    else:
        return A(S, K, H, T, r, q, sigma, phi) + \
            E(S, K, H, R, T, r, q, sigma, eta)


def PDI(
    S: bsParam,
    K: float,
    H: float,
    R: float,
    T: bsParam,
    r: float,
    q: float,
    sigma: float,
    phi: indicCall = indicCall.PUT,
    eta: indicDown = indicDown.DOWN
):
    if K < H:
        return A(S, K, H, T, r, q, sigma, phi) + \
            E(S, K, H, R, T, r, q, sigma, eta)
    else:
        return B(S, K, H, T, r, q, sigma, phi) - \
            C(S, K, H, T, r, q, sigma, phi, eta) + \
            D(S, K, H, T, r, q, sigma, phi, eta) + \
            E(S, K, H, R, T, r, q, sigma, eta)


def PUI(
    S: bsParam,
    K: float,
    H: float,
    R: float,
    T: bsParam,
    r: float,
    q: float,
    sigma: float,
    phi: indicCall = indicCall.PUT,
    eta: indicDown = indicDown.UP
):
    if K < H:
        return C(S, K, H, T, r, q, sigma, phi, eta) + \
            E(S, K, H, R, T, r, q, sigma, eta)
    else:
        return A(S, K, H, T, r, q, sigma, phi) - \
            B(S, K, H, T, r, q, sigma, phi) + \
            D(S, K, H, T, r, q, sigma, phi, eta) + \
            E(S, K, H, R, T, r, q, sigma, eta)


def CDO(
    S: bsParam,
    K: float,
    H: float,
    R: float,
    T: bsParam,
    r: float,
    q: float,
    sigma: float,
    phi: indicCall = indicCall.CALL,
    eta: indicDown = indicDown.DOWN
):
    if K < H:
        return B(S, K, H, T, r, q, sigma, phi) - \
            D(S, K, H, T, r, q, sigma, phi, eta) + \
            F(S, K, H, R, T, r, q, sigma, eta)
    else:
        return A(S, K, H, T, r, q, sigma, phi) - \
            C(S, K, H, T, r, q, sigma, phi, eta) + \
            F(S, K, H, R, T, r, q, sigma, eta)


def CUO(
    S: bsParam,
    K: float,
    H: float,
    R: float,
    T: bsParam,
    r: float,
    q: float,
    sigma: float,
    phi: indicCall = indicCall.CALL,
    eta: indicDown = indicDown.UP
):
    if K < H:
        return A(S, K, H, T, r, q, sigma, phi) - \
            B(S, K, H, T, r, q, sigma, phi) + \
            C(S, K, H, T, r, q, sigma, phi, eta) -\
            D(S, K, H, T, r, q, sigma, phi, eta) +\
            F(S, K, H, R, T, r, q, sigma, eta)
    else:
        return F(S, K, H, R, T, r, q, sigma, eta)


def PDO(
    S: bsParam,
    K: float,
    H: float,
    R: float,
    T: bsParam,
    r: float,
    q: float,
    sigma: float,
    phi: indicCall = indicCall.PUT,
    eta: indicDown = indicDown.DOWN
):
    if K < H:
        return F(S, K, H, R, T, r, q, sigma, eta)
    else:
        return A(S, K, H, T, r, q, sigma, phi) - \
            B(S, K, H, T, r, q, sigma, phi) + \
            C(S, K, H, T, r, q, sigma, phi, eta) -\
            D(S, K, H, T, r, q, sigma, phi, eta) +\
            F(S, K, H, R, T, r, q, sigma, eta)


def PUO(
    S: bsParam,
    K: float,
    H: float,
    R: float,
    T: bsParam,
    r: float,
    q: float,
    sigma: float,
    phi: indicCall = indicCall.PUT,
    eta: indicDown = indicDown.UP
):
    if K < H:
        return A(S, K, H, T, r, q, sigma, phi) - \
            C(S, K, H, T, r, q, sigma, phi, eta) + \
            F(S, K, H, R, T, r, q, sigma, eta)
    else:
        return B(S, K, H, T, r, q, sigma, phi) - \
            D(S, K, H, T, r, q, sigma, phi, eta) + \
            F(S, K, H, R, T, r, q, sigma, eta)


def A(
    S: bsParam,
    K: float,
    H: float,
    T: bsParam,
    r: float,
    q: float,
    sigma: float,
    phi: indicCall
) -> bsParam:
    x1_ = x1(S, K, T, q, sigma)
    return phi.value * S * exp((q - r) * T) * N(phi.value * x1_) - \
        phi.value * K * exp(-r * T) * N(phi.value * (x1_ - sigma * sqrt(T)))


def B(
    S: bsParam,
    K: float,
    H: float,
    T: bsParam,
    r: float,
    q: float,
    sigma: float,
    phi: indicCall
) -> bsParam:
    x2_ = x2(S, H, T, q, sigma)
    return phi.value * S * exp((q - r) * T) * N(phi.value * x2_) - \
        phi.value * K * exp(-r * T) * N(phi.value * (x2_ - sigma * sqrt(T)))


def C(
    S: bsParam,
    K: float,
    H: float,
    T: bsParam,
    r: float,
    q: float,
    sigma: float,
    phi: indicCall,
    eta: indicDown
) -> bsParam:
    y1_ = y1(S, K, H, T, q, sigma)
    mu_ = mu(q, sigma)
    return phi.value * S * exp((q - r) * T) * pow(H / S, 2 * (mu_ + 1)) * \
        N(eta.value * y1_) - phi.value * K * exp(-r * T) * \
        pow(H / S, 2 * mu_) * N(eta.value * (y1_ - sigma * sqrt(T)))


def D(
    S: bsParam,
    K: float,
    H: float,
    T: bsParam,
    r: float,
    q: float,
    sigma: float,
    phi: indicCall,
    eta: indicDown
) -> bsParam:
    y2_ = y2(S, H, T, q, sigma)
    mu_ = mu(q, sigma)
    return phi.value * S * exp((q - r) * T) * pow(H / S, 2 * (mu_ + 1)) * \
        N(eta.value * y2_) - phi.value * K * exp(-r * T) * \
        pow(H / S, 2 * mu_) * N(eta.value * (y2_ - sigma * sqrt(T)))


def E(
    S: bsParam,
    K: float,
    H: float,
    R: float,
    T: bsParam,
    r: float,
    q: float,
    sigma: float,
    eta: indicDown
) -> bsParam:
    return R * exp(-r * T) * (
        N(eta.value * (x2(S, H, T, q, sigma) - sigma * sqrt(T))) -
        pow(H / S, 2 * mu(q, sigma)) *
        N(eta.value * (y2(S, H, T, q, sigma) - sigma * sqrt(T))))


def F(
    S: bsParam,
    K: float,
    H: float,
    R: float,
    T: bsParam,
    r: float,
    q: float,
    sigma: float,
    eta: indicDown
) -> bsParam:
    mu_ = mu(q, sigma)
    lambda__ = lambda_(r, q, sigma)
    z_ = z(S, H, T, r, q, sigma)
    return R * ((H / S) ** (mu_ + lambda__) * N(eta.value * z_) +
                (H / S) ** (mu_ - lambda__) *
                N(eta.value * (z_ - 2 * lambda__ * sigma * sqrt(T)))
                )


def z(
    S: bsParam,
    H: float,
    T: bsParam,
    r: float,
    q: float,
    sigma: float
) -> bsParam:
    return log(H / S) / (sigma * sqrt(T)) + \
        lambda_(r, q, sigma) * sigma * sqrt(T)


def y2(
    S: bsParam,
    H: float,
    T: bsParam,
    q: float,
    sigma: float
) -> bsParam:
    return log(H / S) / (sigma * sqrt(T)) + \
        (1 + mu(q, sigma)) * sigma * sqrt(T)


def y1(
    S: bsParam,
    K: float,
    H: float,
    T: bsParam,
    q: float,
    sigma: float
) -> bsParam:
    return log(H ** 2 / (S * K)) / (sigma * sqrt(T)) + \
        (1 + mu(q, sigma)) * sigma * sqrt(T)


def x2(
    S: bsParam,
    H: float,
    T: bsParam,
    q: float,
    sigma: float
) -> bsParam:
    return log(S / H) / (sigma * sqrt(T)) + \
        (1 + mu(q, sigma)) * sigma * sqrt(T)


def x1(
    S: bsParam,
    K: float,
    T: bsParam,
    q: float,
    sigma: float
) -> bsParam:
    return log(S / K) / (sigma * sqrt(T)) + \
        (1 + mu(q, sigma)) * sigma * sqrt(T)


def lambda_(
        r: float,
        q: float,
        sigma: float
) -> float:
    return sqrt(mu(q, sigma) ** 2 + 2 * r / sigma ** 2)


def mu(
    q: float,
    sigma: float
) -> float:
    return (q - sigma ** 2 / 2) / sigma ** 2


if __name__ == '__main__':

    import RR as rr

    S = 100
    K = 90
    R = 3
    H = 95
    T = 0.5
    r = 0.08
    q = 0.04
    sigma = 0.25

    Ks = [90, 100, 110] * 3
    Hs = [95] * 3 + [100] * 3 + [105] * 3

    tmp = [None, None]
    for f in [rr.CDI, rr.CUI, rr.PDI, rr.PUI, rr.CDO, rr.CUO, rr.PDO, rr.PUO]:
        for K, H in zip(Ks, Hs):
            for i, sigma in enumerate([0.25, 0.3]):
                tmp[i] = f(S, K, H, R, T, r, q, sigma)
            print(f'{f.__name__} \
            X: {str(K).ljust(5)} \
            H: {str(H).ljust(5)} \
            sigma: {'0.25'.ljust(5)} = {str(tmp[0])[:6].ljust(5)} \
            sigma: {'0.30'.ljust(5)} = {str(tmp[1])[:6].ljust(5)}')
