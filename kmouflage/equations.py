"""
K-mouflage equations of motion (background level), with the optional
Kinessence potential f(phi).

Every function here is stateless: it takes a ``Physics`` context (model +
coupling + potential + cosmological constants) plus the current field state
(phi, phi_prime, a, H_conf) and returns a physical quantity.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

if TYPE_CHECKING:
    from .models.couplings import ConformalCoupling
    from .models.k_functions import KModel
    from .models.potential import Potential


@dataclass(frozen=True)
class Physics:
    """Bundles the model, coupling, potential and cosmological constants
    needed to evaluate the K-mouflage (or Kinessence) equations of motion."""
    model:        "KModel"
    coupling:     "ConformalCoupling"
    potential:    "Potential"
    M4_tilde:     float
    rho_m0_tilde: float
    rho_r0_tilde: float

    def A(self, phi):         return self.coupling.A(phi)
    def alpha(self, phi):     return self.coupling.alpha(phi)
    def alpha_phi(self, phi): return self.coupling.alpha_phi(phi)
    def K(self, X):            return self.model.K(X)
    def Kp(self, X):           return self.model.Kp(X)
    def Kpp(self, X):          return self.model.Kpp(X)
    def f_pot(self, phi):      return self.potential.f(phi)
    def f_pot_phi(self, phi):  return self.potential.f_phi(phi)


def H(E_conf, a) -> float:
    return E_conf / a


def F(phys: Physics, phi) -> float:
    return phys.A(phi) ** (-2)


def F_phi(phys: Physics, phi) -> float:
    return -2.0 * phys.alpha(phi) * F(phys, phi)


def F_phi2(phys: Physics, phi) -> float:
    return 2.0 * F(phys, phi) * (2.0 * phys.alpha(phi) ** 2 - phys.alpha_phi(phi))


def chi(phys: Physics, phi_prime, a) -> float:
    return phi_prime ** 2 / (2.0 * phys.M4_tilde * a ** 2)


def X(phys: Physics, phi, phi_prime, a) -> float:
    return phys.A(phi) ** 2 * chi(phys, phi_prime, a)


def rho_m(phys: Physics, a) -> float:
    return phys.rho_m0_tilde * a ** (-3)


def rho_r(phys: Physics, a) -> float:
    return phys.rho_r0_tilde * a ** (-4)


def p_r(phys: Physics, a) -> float:
    return rho_r(phys, a) / 3.0


def Z(phys: Physics, phi, phi_prime, a) -> float:
    X_ = X(phys, phi, phi_prime, a)
    return phys.A(phi) ** (-2) * phys.Kp(X_) - 6.0 * F(phys, phi) * phys.alpha(phi) ** 2


def Z_eff(phys: Physics, phi, phi_prime, a) -> float:
    X_ = X(phys, phi, phi_prime, a)
    return (phys.A(phi) ** (-2) * (phys.Kp(X_) + 2.0 * X_ * phys.Kpp(X_))
            - 6.0 * F(phys, phi) * phys.alpha(phi) ** 2)


def Z_phi(phys: Physics, phi, phi_prime, a) -> float:
    X_ = X(phys, phi, phi_prime, a)
    av = phys.alpha(phi)
    ap = phys.alpha_phi(phi)
    return (phys.A(phi) ** (-2) * (-2.0 * av * phys.Kp(X_) + 2.0 * av * X_ * phys.Kpp(X_))
            + 12.0 * F(phys, phi) * av * (av ** 2 - ap))


def rho_phi(phys: Physics, phi, phi_prime, a) -> float:
    X_ = X(phys, phi, phi_prime, a)
    Z_ = Z(phys, phi, phi_prime, a)
    F_ = F(phys, phi)
    av = phys.alpha(phi)
    A4 = phys.A(phi) ** (-4)
    return ((Z_ + 3.0 * F_ * av ** 2) * phi_prime ** 2 / a ** 2
            - A4 * phys.M4_tilde * (phys.K(X_) - phys.f_pot(phi)))


def p_phi(phys: Physics, phi, phi_prime, a) -> float:
    X_ = X(phys, phi, phi_prime, a)
    F_ = F(phys, phi)
    av = phys.alpha(phi)
    A4 = phys.A(phi) ** (-4)
    return (- 3.0 * F_ * av ** 2 * phi_prime ** 2 / a ** 2
            + A4 * phys.M4_tilde * (phys.K(X_) - phys.f_pot(phi)))


def E_conf_from_F1(phys: Physics, phi, phi_prime, a) -> float:
    F_       = F(phys, phi)
    F_prime  = F_phi(phys, phi) * phi_prime
    rho_tot  = rho_m(phys, a) + rho_r(phys, a) + rho_phi(phys, phi, phi_prime, a)
    disc     = 9.0 * F_prime ** 2 + 12.0 * F_ * a ** 2 * rho_tot
    if disc < 0:
        raise ValueError(f"Discriminant négatif: disc={disc:.3e}")
    H_conf = (-3.0 * F_prime + np.sqrt(disc)) / (6.0 * F_)
    if H_conf <= 0:
        raise ValueError(f"H_conf={H_conf:.3e} <= 0")
    return H_conf


def linear_system(phys: Physics, phi, phi_prime, a, H_conf):
    F_    = F(phys, phi)
    av    = phys.alpha(phi)
    ap    = phys.alpha_phi(phi)
    X_    = X(phys, phi, phi_prime, a)
    Z_    = Z(phys, phi, phi_prime, a)
    Zphi_ = Z_phi(phys, phi, phi_prime, a)
    Kp    = phys.Kp(X_)
    Kpp   = phys.Kpp(X_)
    A2    = phys.A(phi) ** (-2)
    A4    = phys.A(phi) ** (-4)

    M = np.array([
        [Z_eff(phys, phi, phi_prime, a), 6.0 * av * F_],
        [F_phi(phys, phi),              2.0 * F_     ],
    ])

    f_  = phys.f_pot(phi)
    fp_ = phys.f_pot_phi(phi)

    d1 = (
        - 2.0 * H_conf * (Z_ - A2 * X_ * Kpp) * phi_prime

        - (Zphi_ + 2.0 * av * A2 * Kpp * X_) * phi_prime ** 2

        + 6.0 * F_ * av * (ap - av ** 2) * phi_prime ** 2

        + av * A2 * Kp * phi_prime ** 2

        - 4.0 * av * A4 * phys.M4_tilde * phys.K(X_) * a ** 2

        - 6.0 * av * F_ * H_conf ** 2

        + a ** 2 * A4 * phys.M4_tilde * (4.0 * av * f_ - fp_)
    )

    d2 = (- 2.0 * F_ * (2.0 * av ** 2 - ap) * phi_prime ** 2
          + 2.0 * av * F_ * H_conf * phi_prime
          - F_ * H_conf ** 2
          - a ** 2 * (p_r(phys, a) + p_phi(phys, phi, phi_prime, a)))

    sol_norm     = np.linalg.solve(M, np.array([d1, d2]))
    phi_prime2   = sol_norm[0]
    H_conf_prime = sol_norm[1]

    return M, phi_prime2, H_conf_prime


# ── Demixed form (notes Sec. 8): no R, no H_conf', no linear system ─────────

def H_conf_from_constraint(phys: Physics, phi, phi_prime, a) -> float:
    """Eq. (113): H_conf = alpha phi' + H_E, F1 with the 3 F alpha^2 phi'^2 terms completed into a square."""
    A_  = phys.A(phi)
    X_  = X(phys, phi, phi_prime, a)
    rad = (A_ ** 2 * a ** 2 * (rho_m(phys, a) + rho_r(phys, a))
           + phys.Kp(X_) * phi_prime ** 2
           - a ** 2 * A_ ** (-2) * phys.M4_tilde * (phys.K(X_) - phys.f_pot(phi))) / 3.0
    if rad < 0:
        raise ValueError(f"H_E^2 < 0: rad={rad:.3e}")
    H_conf = phys.alpha(phi) * phi_prime + np.sqrt(rad)
    if H_conf <= 0:
        raise ValueError(f"H_conf={H_conf:.3e} <= 0")
    return H_conf


def phi_prime2_demixed(phys: Physics, phi, phi_prime, a, H_conf) -> float:
    """Eq. (114): phi'' from the demixed Klein-Gordon equation (radiation drops out, p_m = 0)."""
    A_  = phys.A(phi)
    av  = phys.alpha(phi)
    X_  = X(phys, phi, phi_prime, a)
    Kp  = phys.Kp(X_)
    Kpp = phys.Kpp(X_)
    num = (2.0 * (Kp - X_ * Kpp) * (H_conf - av * phi_prime) * phi_prime
           + a ** 2 * A_ ** (-2) * (phys.M4_tilde * phys.f_pot_phi(phi)
                                    + av * A_ ** 4 * rho_m(phys, a)))
    den = Kp + 2.0 * X_ * Kpp
    if not np.isfinite(den) or abs(den) < 1e-30:
        raise RuntimeError(f"K' + 2X K'' = {den:.3e} (singular demixed KG)")
    return -num / den


def H_conf_prime_from_F2(phys: Physics, phi, phi_prime, phi_prime2, a, H_conf) -> float:
    """Eq. (120): H_conf' from the second Friedmann equation, once phi'' is known."""
    F_       = F(phys, phi)
    F_prime  = F_phi(phys, phi) * phi_prime
    F_prime2 = F_phi2(phys, phi) * phi_prime ** 2 + F_phi(phys, phi) * phi_prime2
    return (- 0.5 * H_conf ** 2
            - (a ** 2 * (p_r(phys, a) + p_phi(phys, phi, phi_prime, a))
               + F_prime2 + H_conf * F_prime) / (2.0 * F_))


# ── Conjugate momentum (notes eq. 116), rescaled: pi_t = pi / a^2 = A^-2 K'(X) phi' ──
# pi_t is ~constant in radiation domination, whereas pi ~ a^2 spans many decades.

def pi_tilde(phys: Physics, phi, phi_prime, a) -> float:
    return phys.A(phi) ** (-2) * phys.Kp(X(phys, phi, phi_prime, a)) * phi_prime


def phi_prime_from_pi_tilde(phys: Physics, phi, pi_t, a, guess=None) -> float:
    """Root of K'(X) phi' = A^2 pi_t, X = A^2 phi'^2 / (2 M4 a^2).
    g' = K' + 2X K'' > 0 (denominator of (114)) so the root is unique; Newton, brentq as fallback."""
    c = phys.A(phi) ** 2 * pi_t
    if c == 0.0:
        return 0.0

    def g(pp):
        return phys.Kp(X(phys, phi, pp, a)) * pp - c

    pp = guess if (guess is not None and guess * c > 0.0) else c / max(phys.Kp(0.0), 1.0)
    for _ in range(50):
        X_ = X(phys, phi, pp, a)
        dg = phys.Kp(X_) + 2.0 * X_ * phys.Kpp(X_)
        if not np.isfinite(dg) or dg <= 0.0:
            break
        step = (phys.Kp(X_) * pp - c) / dg
        new = pp - step
        if new * c <= 0.0:          # stay on the sign of the root
            new = 0.5 * pp
        if abs(new - pp) <= 1e-14 * abs(new):
            return new
        pp = new

    lo, hi = sorted((0.0, c))
    for _ in range(200):
        if g(lo) * g(hi) <= 0.0:
            break
        lo, hi = (2.0 * lo, hi) if c < 0 else (lo, 2.0 * hi)
    else:
        raise RuntimeError(f"phi_prime_from_pi_tilde: no bracket (pi_t={pi_t:.3e}, a={a:.3e})")
    return brentq(g, lo, hi, xtol=1e-300, rtol=1e-15)


def dpi_tilde_dN(phys: Physics, phi, pi_t, a, H_conf) -> float:
    """Demixed KG (50) for pi_t = pi/a^2: dpi_t/dN = -2 pi_t - (a^2/H)(A^-4 M4 f_phi + alpha rho_m), p_m = 0."""
    return (- 2.0 * pi_t
            - a ** 2 / H_conf * (phys.A(phi) ** (-4) * phys.M4_tilde * phys.f_pot_phi(phi)
                                 + phys.alpha(phi) * rho_m(phys, a)))


def pi_tilde_ini(phys: Physics, phi0, a_ini, omega_m, omega_r) -> float:
    """Eq. (116)-(117) integrated from a = 0 with H_conf^2 = omega_r (1 + a/a_eq) / a^2, alpha const, f_phi neglected.
    With t = sqrt(1 + a/a_eq): pi_t = -2 alpha omega_m (t + 2) / (sqrt(omega_r) (t + 1)^2)
    (cancellation-free form; -> -3 alpha omega_m / (2 sqrt(omega_r)) deep in RD, i.e. (117))."""
    t = np.sqrt(1.0 + a_ini * omega_m / omega_r)
    return -2.0 * phys.alpha(phi0) * omega_m * (t + 2.0) / (np.sqrt(omega_r) * (t + 1.0) ** 2)
