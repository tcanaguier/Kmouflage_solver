"""
Kinetic functions K(X), K'(X), K''(X).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class KModel:
    """
    Container for a kinetic model.

    Attributes
    ----------
    name : str
    K    : Callable[[float], float]   K(X)
    Kp   : Callable[[float], float]   dK/dX  (K'(X))
    Kpp  : Callable[[float], float]   d²K/dX²
    params : dict
        Free parameters for display / comparison.
    """
    name:   str
    K:      Callable[[float], float]
    Kp:     Callable[[float], float]
    Kpp:    Callable[[float], float]
    params: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"KModel({self.name})"


def make_powerlaw_K(K0: float, m: int) -> KModel:
    """
    Build a power-law K-model:

        K(X)   = X + K0 · X^m,               K(0) = 0
        K'(X)  = 1  + m · K0 · X^(m-1)
        K''(X) = m(m-1) · K0 · X^(m-2)   [0 if m < 2]

    Pure kinetic term (K-essence). The vacuum-energy offset lives in the
    potential f(φ), see models.potential.

    Parameters
    ----------
    K0 : float   Coefficient of the X^m term.
    m  : int     Exponent (integer ≥ 1; m=3 recommended).
    """
    if m < 1:
        raise ValueError(f"Exponent m must be ≥ 1, got {m}.")

    def K(X: float) -> float:
        return X + K0 * X**m

    def Kp(X):
        return 1.0 + m * K0 * X ** (m - 1)

    def Kpp(X):
        if m < 2:
            return 0.0

        return m * (m - 1) * K0 * X ** (m - 2)

    return KModel(
        name   = f"power-law (K0={K0}, m={m})",
        K      = K,
        Kp     = Kp,
        Kpp    = Kpp,
        params = {"K0": K0, "m": m},
    )


def make_LambdaCDM_K() -> KModel:
    """
    Canonical kinetic term K(X) = X, to be used with make_LambdaCDM_coupling()
    and a constant potential to recover ΛCDM with Λ = M4_tilde.

    With alpha = f_phi = 0 the demixed Klein-Gordon source vanishes, pi_ini = 0
    and phi' ≡ 0, so K never enters the background. K ≡ 0 would not work: with
    K' ≡ 0 the Klein-Gordon equation degenerates (phi' undetermined) and the
    solver cannot invert K'(X) phi' = A^2 pi.
    """
    def K(X):   return X
    def Kp(X):  return np.ones_like(X, dtype=float)
    def Kpp(X): return np.zeros_like(X, dtype=float)

    return KModel(
        name   = "ΛCDM (K=X, frozen field)",
        K      = K,
        Kp     = Kp,
        Kpp    = Kpp,
        params = {},
    )


def make_arctan_K(K_star, X_star):
    def K(X):   return X + K_star * (X - X_star * np.arctan(X / X_star))
    def Kp(X):  return 1.0 + K_star * (1.0 - 1.0 / (1.0 + (X / X_star)**2))
    def Kpp(X): return K_star * (2.0 * X / X_star**2) / (1.0 + (X / X_star)**2)**2

    return KModel(
        name   = f'arctan (K_star={K_star}, X_star={X_star})',
        K      = K,
        Kp     = Kp,
        Kpp    = Kpp,
        params = {'K_star': K_star, 'X_star': X_star},
    )