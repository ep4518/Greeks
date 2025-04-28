from __future__ import annotations
__author__ = "epeterson"
__version__ = "0.0"

import abc
from dataclasses import dataclass
from functools import partial

import BS as bs
from numpy.typing import NDArray
import numpy as np
from scipy.stats import norm


R: float = 0.03


@dataclass
class Equity:
    q: float
    sigma: float


nvda = Equity(0.0, 0.4)


# Base payoff class
class Greek(abc.ABC):
    @abc.abstractmethod
    def __call__(self, ST):
        pass

    def __add__(self, other):
        return CompositeGreek(self, other, op='+')

    def __sub__(self, other):
        return CompositeGreek(self, other, op='-')


class CompositeGreek(Greek):
    def __init__(self, left, right, op='+'):
        self.left = left
        self.right = right
        self.op = op

    def __call__(self, ST):
        if self.op == '+':
            return self.left(ST) + self.right(ST)
        elif self.op == '-':
            return self.left(ST) - self.right(ST)


class Payoff(Greek):

    def __call__(self, S):
        return bs.C(S, K, T, r, q, sigma)

# class VanillaPayoff(Greek):
#     def __init__(self, K, option_type):
#         self.K = K
#         self.option_type = option_type.lower()
#
#     def __call__(self, ST):
#         if self.option_type == 'call':
#             return max(ST - self.K, 0)
#         elif self.option_type == 'put':
#             return max(self.K - ST, 0)


class Vanilla:
    def __init__(self, option_type, K, T, r=R, udly: Equity = nvda):
        self.K = K
        self.T = T
        self.r = r
        self.option_type = option_type.lower()
        # the callable payoff object
        self.payoff = Greek()


# Example usage
if __name__ == "__main__":
    call = Vanilla(K=100, T=1, r=0.05, udly=nvda, option_type='call')
    put = Vanilla(K=100, T=1, r=0.05, udly=nvda, option_type='put')

    combo_payoff = call.payoff + put.payoff  # sum of call and put payoffs

    for ST in [90, 100, 110]:
        print(f"Combined payoff at ST={ST}: {combo_payoff(ST)}")
