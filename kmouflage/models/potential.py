from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Potential:
    name:   str
    f:      Callable[[float], float]
    f_phi:  Callable[[float], float]
    params: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Potential({self.name})"


def make_constant_potential() -> Potential:
    return Potential(
        name   = "constant (f=1)",
        f      = lambda phi: 1.0,
        f_phi  = lambda phi: 0.0,
        params = {},
    )


def make_exponential_potential(lam: float = 1.0) -> Potential:
    def f(phi):     return np.exp(-lam * phi)
    def f_phi(phi): return -lam * f(phi)
    return Potential(
        name   = f"exponential (lam={lam})",
        f      = f,
        f_phi  = f_phi,
        params = {"lam": lam},
    )
