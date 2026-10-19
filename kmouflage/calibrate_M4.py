"""
>>> from kmouflage.solver import KMouflageBackground, CosmologicalParams
>>> from kmouflage.calibrate_M4 import calibrate_M4_tilde
>>>
>>> bg = KMouflageBackground(model, coupling, cosmo=CosmologicalParams(M4_tilde=0.75))
>>> M4, H0_final, niter = calibrate_M4_tilde(bg, target_H0=67.36)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .solver import KMouflageBackground


def calibrate_M4_tilde(
    bg: KMouflageBackground,
    target_H0: float,
    tol: float = 1e-5,
    max_iter: int = 100,
    verbose: bool = True,
    bracket: tuple[float, float] = (0.01, 15.0),
) -> tuple[float, float, int]:
    def _run_and_read(m4: float) -> float:
        if m4 <= 0.0:
            raise ValueError(f"M4_tilde = {m4:.3e} <= 0")
        bg.M4_tilde = m4
        bg.run(verbose=False)
        return float(bg.H0_predicted)

    if verbose:
        header = f"{'it':>4}  {'M4_tilde':>16}  {'H0_predicted':>14}  {'err':>13}"
        sep = "─" * len(header)
        print(f"\n[calibrate_M4_tilde]  target_H0={target_H0}  tol={tol:.1e}")
        print(sep)
        print(header)
        print(sep)

    it_count = [0]

    def residual(m4: float) -> float:
        H0 = _run_and_read(m4)
        err = H0 - target_H0
        if verbose:
            print(f"{it_count[0]:>4}  {m4:>16.10f}  {H0:>14.6f}  {err:>+13.4e}")
        it_count[0] += 1
        return err

    m4_sol = brentq(residual, bracket[0], bracket[1], xtol=tol * 1e-3, rtol=1e-12,
                    maxiter=max_iter, full_output=False)
    H0_final = _run_and_read(m4_sol)
    n_iter = it_count[0]
    if verbose:
        print(sep)
        print(f"[brentq converged]  it={n_iter}  M4_tilde={m4_sol:.10f}  "
              f"H0_predicted={H0_final:.6f}  err={H0_final - target_H0:+.4e}\n")
    return bg.M4_tilde, H0_final, n_iter
