from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ConformalCoupling:
    name:      str
    A:         Callable[[float], float]
    alpha:     Callable[[float], float]
    alpha_phi: Callable[[float], float]
    params:    dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"ConformalCoupling({self.name})"


def make_exponential_coupling(beta: float) -> ConformalCoupling:
    return ConformalCoupling(
        name      = f"exponential (β={beta})",
        A         = lambda phi: np.exp(beta * phi),
        alpha     = lambda phi: float(beta),
        alpha_phi = lambda phi: 0.0,
        params    = {"beta": beta},
    )


def make_gaussian_coupling(beta: float, gamma: float) -> ConformalCoupling:
    return ConformalCoupling(
        name      = f"gaussian (β={beta}, γ={gamma})",
        A         = lambda phi: np.exp(beta * phi + 0.5 * gamma * phi**2),
        alpha     = lambda phi: float(beta + gamma * phi),
        alpha_phi = lambda phi: float(gamma),
        params    = {"beta": beta, "gamma": gamma},
    )


def make_LambdaCDM_coupling() -> ConformalCoupling:
    return ConformalCoupling(
        name      = "ΛCDM (A=1, no coupling)",
        A         = lambda phi: 1.0,
        alpha     = lambda phi: 0.0,
        alpha_phi = lambda phi: 0.0,
        params    = {},
    )
