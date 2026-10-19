"""
Checks on a K-mouflage (or Kinessence) run.

verify(bg) runs everything on a background that has already run:
  numerical accuracy (numerical derivatives vs the equations, Friedmann, closure),
  stability along the trajectory (no ghost, demixed KG regular, c_s^2 > 0),
  Solar System screening (Cassini, Moon periastron, m_eff criterion).

verify_screening(model, coupling, M4_tilde) runs only the Solar System part.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from .solver import KMouflageBackground
from .models.k_functions import KModel
from .models.couplings import ConformalCoupling


# Physical constants (SI), notes Sec. 6.4
GM_SUN   = 1.32712e20          # [m^3 s^-2]
AU       = 1.49598e11          # [m]
R_SUN    = 6.96e8              # [m]
RHO_SUN  = 1.5e5               # solar core density [kg m^-3]
G_NEWTON = 6.6743e-11          # [SI]
C_LIGHT  = 2.99792458e8        # [m s^-1]
H100_SI  = 1e5 / 3.0857e22     # 100 km/s/Mpc [s^-1], the solver's unit

# Bounds
CASSINI_BOUND   = 1e-5    # beta^2/K', eq. (52)/(103)
# TODO: |gamma - 1| < 2.3e-5 with gamma - 1 = -4 beta^2/K' would give 5.75e-6 (relation to double check)
MOON_BOUND      = 8e-13   # periastron, eq. (105)
MEFF_BOUND      = 1e-2    # m_eff R_sun / sqrt(Z) << 1
NUMERICAL_BOUND = 1e-3    # numerical accuracy checks


@dataclass
class Check:
    name:   str
    value:  float
    bound:  Optional[float]           # None: information only
    passed: Optional[bool]
    note:   str = ""

    def line(self) -> str:
        status = "" if self.passed is None else ("PASS" if self.passed else "FAIL")
        bound = "" if self.bound is None else f"{self.bound:.1e}"
        return f"  {self.name:<30} {self.value:>11.3e}  {bound:>8}  {status:<4}  {self.note}"


@dataclass
class Report:
    title:  str
    checks: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed is not False for c in self.checks)

    def __getitem__(self, name: str) -> Check:
        return next(c for c in self.checks if c.name == name)

    def add(self, name, value, bound=None, ok=None, note=""):
        value = float(value)
        if ok is None and bound is not None:
            ok = abs(value) <= bound
        self.checks.append(Check(name, value, bound, None if ok is None else bool(ok), note))

    def __repr__(self) -> str:
        head = f"{self.title}\n  {'check':<30} {'value':>11}  {'bound':>8}  status"
        lines = [head, "  " + "─" * 70, *(c.line() for c in self.checks), "  " + "─" * 70,
                 f"  {'ALL PASS' if self.passed else 'SOME CHECKS FAIL'}"]
        return "\n".join(lines)


# Solar System screening

@dataclass
class ScreeningReport:
    model_name:    str
    coupling_name: str
    alpha:         float
    M4_tilde:      float

    X_ss:          float   # kinetic variable at 1 AU (< 0)
    Kp_ss:         float
    Kpp_ss:        float

    cassini_value: float   # beta^2 / K'
    cassini_pass:  bool
    moon_value:    float   # (beta^2/K') |X K''| / (K' + 2X K'')
    moon_pass:     bool

    meffR:           Optional[float] = None
    meff_ratio:      Optional[float] = None   # m_eff R_sun / sqrt(K')
    meff_negligible: Optional[bool]  = None

    def to_report(self) -> Report:
        r = Report(f"Screening: {self.model_name}, {self.coupling_name}, M4_tilde={self.M4_tilde:g}")
        r.add("X at 1 AU", self.X_ss, note=f"K'={self.Kp_ss:.3e}")
        r.add("Cassini beta^2/K'", self.cassini_value, CASSINI_BOUND, self.cassini_pass)
        r.add("Moon periastron", self.moon_value, MOON_BOUND, self.moon_pass)
        if self.meff_ratio is not None:
            r.add("m_eff R_sun / sqrt(Z)", self.meff_ratio, MEFF_BOUND, self.meff_negligible,
                  f"m_eff R_sun = {self.meffR:.2e}")
        return r

    def __repr__(self) -> str:
        return repr(self.to_report())


def _solve_X_ss(Kp, rhs: float) -> float:
    """Eq. (102): sqrt(-X) K'(X) = rhs, X < 0."""
    if rhs == 0.0:
        return 0.0
    g = lambda y: np.sqrt(y) * Kp(-y) - rhs      # y = -X > 0
    hi = 1.0
    while g(hi) < 0.0:
        hi *= 10.0
        if hi > 1e40:
            raise RuntimeError("no screened solution: sqrt(-X) K'(X) never reaches the Solar System value")
    return -brentq(g, 0.0, hi, xtol=1e-300, rtol=1e-14)


def verify_screening(
    model:      KModel,
    coupling:   ConformalCoupling,
    M4_tilde:   float = 1.0,
    phi0:       float = 0.0,
    check_meff: bool  = True,
) -> ScreeningReport:
    """
    Solar System tests at 1 AU (notes Sec. 6), for the coupling alpha = alpha(phi0).

    In solver units the right-hand side of eq. (102) reads
        sqrt(-X) K'(X) = sqrt(2) alpha g_sun / (sqrt(M4_tilde) c H100),   g_sun = GM_sun / (1 AU)^2,
    i.e. the notes' sqrt(2) beta g / (M_num^2 c H0) with M_num^4 H0^2 = M4_tilde H100^2.
    """
    alpha = float(coupling.alpha(phi0))
    g_sun = GM_SUN / AU**2
    rhs = np.sqrt(2.0) * abs(alpha) * g_sun / (np.sqrt(M4_tilde) * C_LIGHT * H100_SI)

    X_ss   = _solve_X_ss(model.Kp, rhs)
    Kp_ss  = float(model.Kp(X_ss))
    Kpp_ss = float(model.Kpp(X_ss))

    cassini = alpha**2 / Kp_ss
    moon = cassini * abs(X_ss * Kpp_ss) / (Kp_ss + 2.0 * X_ss * Kpp_ss)

    report = ScreeningReport(
        model_name=model.name, coupling_name=coupling.name, alpha=alpha, M4_tilde=M4_tilde,
        X_ss=X_ss, Kp_ss=Kp_ss, Kpp_ss=Kpp_ss,
        cassini_value=cassini, cassini_pass=cassini <= CASSINI_BOUND,
        moon_value=moon, moon_pass=moon <= MOON_BOUND,
    )

    if check_meff:
        # eq. (106): m_eff = alpha sqrt(8 pi G rho) / c, inside the Sun
        # TODO: also m_eff in voids, and a check of the quasi-static approximation
        m_eff = abs(alpha) * np.sqrt(8.0 * np.pi * G_NEWTON * RHO_SUN) / C_LIGHT
        report.meffR = m_eff * R_SUN
        report.meff_ratio = report.meffR / np.sqrt(Kp_ss)
        report.meff_negligible = report.meff_ratio < MEFF_BOUND

    return report


# Full check of a run 

def verify(bg: KMouflageBackground) -> Report:
    """Numerical accuracy, stability and Solar System checks on a background that has run."""
    if not hasattr(bg, "_N"):
        raise RuntimeError("Call bg.run() before verify(bg).")

    r = Report(f"verify: {bg.model.name}, {bg.coupling.name}, {bg.potential.name}, M4_tilde={bg.M4_tilde:g}")

    # numerical accuracy, on the solver's own grid (edges skipped for np.gradient)
    N = bg._N
    H, phi, pp, pp2 = bg.E_conf(N), bg.phi(N), bg.phi_prime(N), bg.phi_prime2(N)
    inner = slice(2, -2)

    def rel(num, ana):   # max |num - ana| normalised by max |ana| (ana may cross zero)
        scale = np.max(np.abs(ana))
        return np.max(np.abs(num - ana)[inner]) / scale if scale > 0 else np.max(np.abs(num)[inner])

    r.add("dphi/dN  num vs phi'/H", rel(np.gradient(phi, N), pp / H), NUMERICAL_BOUND)
    r.add("dphi'/dN num vs phi''/H", rel(np.gradient(pp, N), pp2 / H), NUMERICAL_BOUND)
    r.add("dlnH/dN  num vs H'/H^2", np.max(np.abs(np.gradient(np.log(H), N) - bg.H_conf_prime(N) / H**2)[inner]),
          NUMERICAL_BOUND, note="Friedmann 2 vs derivative of Friedmann 1")
    r.add("Friedmann 1 residual", np.max(bg.residual_F1(N)), NUMERICAL_BOUND)
    r.add("Omega_m+Omega_r+Omega_DE-1", np.max(np.abs(bg.Omega_m(N) + bg.Omega_r(N) + bg.Omega_de_def(N) - 1.0)),
          NUMERICAL_BOUND)
    r.add("Omega_DE at z_ini", float(bg.Omega_de_def(N[0])), NUMERICAL_BOUND,
          note="initial conditions assume no dark energy")

    # stability along the trajectory
    X = bg.X(N)
    Kp, Kpp = np.asarray(bg.model.Kp(X)), np.asarray(bg.model.Kpp(X))
    cs2 = bg.Z(N) / bg.Z_eff(N)
    r.add("min K'", np.min(Kp), ok=np.min(Kp) > 0, note="no ghost")
    r.add("min K' + 2X K''", np.min(Kp + 2.0 * X * Kpp), ok=np.min(Kp + 2.0 * X * Kpp) > 0,
          note="demixed KG regular")
    r.add("min c_s^2", np.min(cs2), ok=np.min(cs2) > 0, note=f"max c_s^2 = {np.max(cs2):.3f}")
    r.add("epsilon_F today", float(bg.epsilon_F(0.0)), note="M_Pl^2/F - 1")

    # Solar System, with the coupling of today
    s = verify_screening(bg.model, bg.coupling, M4_tilde=bg.M4_tilde, phi0=float(bg.phi(0.0)))
    r.checks += s.to_report().checks
    return r
