from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class GrowthSolver:
    """
    Shared ODE core resolution for f/D+.
    """

    def __init__(
        self,
        N_ini:    float,
        N_end:    float = 0.0,
        N_points: int   = 5000,
        rtol:     float = 1e-9,
        atol:     float = 1e-11,
    ):
        self.N_ini    = N_ini
        self.N_end    = N_end
        self.N_points = N_points
        self.rtol     = rtol
        self.atol     = atol
        self._results = None
        self._gamma_interp = None
        self._S_interp     = None

    def _ic(self) -> np.ndarray:
        """f(N_ini) = 1, ln D+(N_ini) = N_ini  (matter-domination attractor)."""
        return np.array([1.0, self.N_ini])

    def _rhs(self, N: float, Y: np.ndarray) -> np.ndarray:
        f     = Y[0]
        gamma = float(self._gamma_interp(N))
        S     = float(self._S_interp(N))
        return np.array([-f**2 - gamma * f + S, f])

    def _integrate(self):
        """Solve the f/D+ ODE, normalize D+(z=0)=1, return raw arrays."""
        N_eval = np.linspace(self.N_ini, self.N_end, self.N_points)

        sol = solve_ivp(
            fun    = self._rhs,
            t_span = (self.N_ini, self.N_end),
            y0     = self._ic(),
            method = "DOP853",
            t_eval = N_eval,
            rtol   = self.rtol,
            atol   = self.atol,
        )
        if not sol.success:
            raise RuntimeError(f"Intégration growth échouée : {sol.message}")

        f    = sol.y[0]
        lnDp = sol.y[1]
        Dp   = np.exp(lnDp - lnDp[-1])    # normalisation D+(z=0) = 1

        a_arr = np.exp(N_eval)
        z_arr = 1.0 / a_arr - 1.0
        return N_eval, f, Dp, a_arr, z_arr

    def run(self, verbose: bool = True) -> dict:
        raise NotImplementedError

    def summary(self) -> None:
        raise NotImplementedError


class LCDMGrowth(GrowthSolver):
    """
    ΛCDM growth solver.
    """

    def __init__(
        self,
        Omega_m0: float        = 0.25,
        Omega_r0: float        = 8.4e-5,
        N_ini:    float | None = None,
        N_end:    float        = 0.0,
        N_points: int          = 5000,
        rtol:     float        = 1e-9,
        atol:     float        = 1e-11,
    ):
        N_ini = N_ini if N_ini is not None else np.log(1.0 / 1001.0)  # z ≃ 1000
        super().__init__(N_ini=N_ini, N_end=N_end, N_points=N_points, rtol=rtol, atol=atol)

        self.Omega_m0 = Omega_m0
        self.Omega_r0 = Omega_r0
        self.Omega_L0 = 1.0 - Omega_m0 - Omega_r0

        N_bg  = np.linspace(self.N_ini, self.N_end, 10 * N_points)
        E2_bg = self._E2_vec(N_bg)

        gamma_bg = self._gamma_vec(N_bg, E2_bg)
        S_bg     = 1.5 * self._Omega_m_vec(N_bg, E2_bg)

        kw = dict(kind='linear', bounds_error=False, fill_value='extrapolate')
        self._gamma_interp = interp1d(N_bg, gamma_bg, **kw)
        self._S_interp     = interp1d(N_bg, S_bg,     **kw)

    def _E2_vec(self, N: np.ndarray) -> np.ndarray:
        return np.exp(2*N)*(
            self.Omega_m0 * np.exp(-3.0 * N)
            + self.Omega_r0 * np.exp(-4.0 * N)
            + self.Omega_L0
        )

    def _E_conf(self, N: np.ndarray, E2: np.ndarray) -> np.ndarray:
        return np.sqrt(E2)

    def _gamma_vec(self, N: np.ndarray, E2_conf: np.ndarray) -> np.ndarray:
        dE2_conf_dN = (
            -1.0 * self.Omega_m0 * np.exp(-1.0 * N)
            - 2.0 * self.Omega_r0 * np.exp(-2.0 * N)
            + 2.0 * self.Omega_L0 * np.exp(2.0 * N)
        )
        dlnEconf_dN = 0.5 * dE2_conf_dN / E2_conf
        return 1.0 + dlnEconf_dN

    def _Omega_m_vec(self, N: np.ndarray, E2_conf: np.ndarray) -> np.ndarray:
        return self.Omega_m0 * np.exp(-1.0 * N) / E2_conf

    def run(self, verbose: bool = True) -> dict:
        N_eval, f, Dp, a_arr, z_arr = self._integrate()

        E2_arr    = self._E2_vec(N_eval)
        Om_arr    = self._Omega_m_vec(N_eval, E2_arr)
        gamma_arr = self._gamma_vec(N_eval, E2_arr)
        S_arr     = 1.5 * Om_arr
        E_arr     = np.sqrt(E2_arr)                 # H/H0
        Econf_arr = self._E_conf(N_eval, E2_arr)    # ℋ/H0

        if verbose:
            print(
                f"[ΛCDM] f(z=0)={f[-1]:.5f} | D+(z=0)={Dp[-1]:.5f} | "
                f"Ω_m(z=0)={Om_arr[-1]:.5f} | E(z=0)={E_arr[-1]:.5f} | "
                f"γ(z=0)={gamma_arr[-1]:.5f}"
            )

        self._results = dict(
            N=N_eval, z=z_arr, a=a_arr,
            f=f, Dp=Dp,
            E=E_arr, E_conf=Econf_arr,
            Omega_m=Om_arr, gamma=gamma_arr, S=S_arr,
        )
        return self._results

    def summary(self) -> None:
        if self._results is None:
            print("Lance d'abord le solveur : lcdm.run()"); return
        r  = self._results
        i0 = np.argmin(np.abs(r["z"]))               # z ≃ 0
        i1 = np.argmin(np.abs(r["z"] - 1.0))         # z ≃ 1
        i2 = np.argmin(np.abs(r["z"] - 0.5))         # z ≃ 0.5

        print("\n── ΛCDM Growth ──────────────────────────────────────────────")
        print(f"  Ω_m0={self.Omega_m0:.4f} | Ω_r0={self.Omega_r0:.2e} | Ω_Λ0={self.Omega_L0:.4f}")
        print(f"  {'z':>5}  {'f':>8}  {'D+':>8}  {'Ω_m':>8}  {'γ':>8}")
        print(f"  {'-'*45}")
        for label, idx in [("z=0", i0), ("z=0.5", i2), ("z=1", i1)]:
            print(
                f"  {label:>5}  "
                f"{r['f'][idx]:8.5f}  "
                f"{r['Dp'][idx]:8.5f}  "
                f"{r['Omega_m'][idx]:8.5f}  "
                f"{r['gamma'][idx]:8.5f}"
            )
        print("─────────────────────────────────────────────────────────────\n")


class KMouflageGrowth(GrowthSolver):
    """
    K-mouflage growth solver (also valid with a Kinessence potential).
    The background starts at z_ini ~ 1e5 (RDE); the growth ODE must start
    well into MDE for fgrowth=1 to be a valid initial condition, hence
    N_MDE = N_eq + N_offset (N_offset e-folds after matter-radiation
    equality, default 5).
    """

    def __init__(
        self,
        bg,
        N_offset: float = 5.0,
        N_points: int   = 2000,
        rtol:     float = 1e-9,
        atol:     float = 1e-11,
    ):
        self.bg = bg

        N_eq  = np.log(bg.aeq)
        self.N_eq = N_eq
        N_MDE = N_eq + N_offset

        # Sécurité : N_MDE doit rester dans le domaine du background
        N_MDE = max(N_MDE, bg.N_ini + 0.1)
        N_MDE = min(N_MDE, bg.N_end - 1.0)

        super().__init__(N_ini=N_MDE, N_end=bg.N_end, N_points=N_points, rtol=rtol, atol=atol)

        # Pré-calcul des interpolants γ(N) et S(N) sur grille dense
        N_bg = np.linspace(self.N_ini, self.N_end, 10 * N_points)

        H_conf       = bg.E_conf(N_bg)
        H_conf_prime = bg.H_conf_prime(N_bg)
        mu_K         = bg.mu_K(N_bg)
        Omega_m      = bg.Omega_m(N_bg)

        gamma_bg = 1.0 + H_conf_prime / H_conf**2
        S_bg     = 1.5 * mu_K * Omega_m

        kw = dict(kind='linear', bounds_error=False, fill_value='extrapolate')
        self._gamma_interp = interp1d(N_bg, gamma_bg, **kw)
        self._S_interp     = interp1d(N_bg, S_bg,     **kw)

        
        Om_ini = float(bg.Omega_m(self.N_ini))
        Or_ini = float(bg.Omega_r(self.N_ini))
        mu_ini = float(bg.mu_K(self.N_ini))
        F_ini  = float(bg.F(self.N_ini))
        ratio  = Or_ini / (Om_ini + 1e-30)

        z_MDE = float(1.0 / np.exp(self.N_ini) - 1.0)
        z_eq  = float(1.0 / bg.aeq - 1.0)


    def run(self, verbose: bool = True) -> dict:
        N_eval, f, Dp, a_arr, z_arr = self._integrate()

        H_conf_arr       = self.bg.E_conf(N_eval)
        H_conf_prime_arr = self.bg.H_conf_prime(N_eval)
        mu_K_arr         = self.bg.mu_K(N_eval)
        Omega_m_arr      = self.bg.Omega_m(N_eval)
        gamma_arr        = 1.0 + H_conf_prime_arr / H_conf_arr**2
        S_arr            = 1.5 * mu_K_arr * Omega_m_arr

        if verbose:
            print(
                f"[K-mouflage] f(z=0)={f[-1]:.5f} | D+(z=0)={Dp[-1]:.5f} | "
                f"Ω_m(z=0)={Omega_m_arr[-1]:.5f} | μ_K(z=0)={mu_K_arr[-1]:.5f} | "
                f"γ(z=0)={gamma_arr[-1]:.5f}     | f(N_ini) ={f[0]}"
            )

        self._results = dict(
            N=N_eval, z=z_arr, a=a_arr,
            f=f, Dp=Dp,
            H_conf=H_conf_arr, gamma=gamma_arr, S=S_arr,
            mu_K=mu_K_arr, Omega_m=Omega_m_arr,
        )
        return self._results

    def summary(self) -> None:
        if self._results is None:
            print("Lance d'abord : kg.run()"); return
        r  = self._results
        i0 = np.argmin(np.abs(r["z"]))
        i1 = np.argmin(np.abs(r["z"] - 1.0))
        i2 = np.argmin(np.abs(r["z"] - 0.5))

        print("\n── K-mouflage Growth ────────────────────────────────────────────")
        print(f"  N_ini (MDE) = {self.N_ini:.3f}  "
              f"(z_ini = {1.0/np.exp(self.N_ini)-1.0:.0f}, "
              f"{self.N_ini - self.N_eq:.1f} e-folds après z_eq)")
        print(f"  {'z':>5}  {'f':>8}  {'D+':>8}  {'Ω_m':>8}  {'μ_K':>8}  {'γ':>8}")
        print(f"  {'-'*55}")
        for label, idx in [("z=0", i0), ("z=0.5", i2), ("z=1", i1)]:
            print(
                f"  {label:>5}  "
                f"{r['f'][idx]:8.5f}  "
                f"{r['Dp'][idx]:8.5f}  "
                f"{r['Omega_m'][idx]:8.5f}  "
                f"{r['mu_K'][idx]:8.5f}  "
                f"{r['gamma'][idx]:8.5f}"
            )
        print("──────────────────────────────────────────────────────────────────\n")
