"""Build/run a background, save it to disk, compare it to ΛCDM."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

from .solver import KMouflageBackground, CosmologicalParams

# enough to rebuild every other quantity later, so always saved
MANDATORY_FIELDS = ["a", "phi", "phi_prime"]

DEFAULT_FIELDS = [
    "z", "a", "phi", "phi_prime",
    "t_cosmic", "eta", "t_superconform", "H", "H_conf",
    "Omega_m", "Omega_r", "Omega_de_def", "w_de_def", "w_phi",
]

# <project root>/runs, regardless of cwd
DEFAULT_RUNS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"
)


def run_background(model, coupling, potential=None, cosmo=None, ic=None,
                    verbose: bool = False, **kwargs) -> KMouflageBackground:
    """Build and run a KMouflageBackground."""
    bg = KMouflageBackground(model=model, coupling=coupling, potential=potential,
                              cosmo=cosmo, ic=ic, **kwargs)
    bg.run(verbose=verbose)
    return bg


def _base_name(obj) -> str:
    """"power-law (K0=1, m=3)" -> "power_law"."""
    return re.sub(r"[^0-9A-Za-z]+", "_", obj.name.split("(", 1)[0].strip()).strip("_")


def default_table_name(bg: KMouflageBackground) -> str:
    """Suggests a name from the model's real params, e.g. "..._beta-0.1_..."."""
    def tag(obj):
        return _base_name(obj) + "".join(f"_{k}-{v}" for k, v in obj.params.items())
    return (f"kmouflage_table_{tag(bg.coupling)}_{tag(bg.model)}_{tag(bg.potential)}"
            f"_M4-{bg.M4_tilde:g}")


def save_table(bg: KMouflageBackground, name: str = None, fields=None,
               outdir: str = DEFAULT_RUNS_DIR) -> str:
    """Save an already-run bg as one CSV (run info in a commented header, then the data). Returns the path."""
    if not hasattr(bg, "_N"):
        raise RuntimeError("bg.run() must be called before save_table().")

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name or default_table_name(bg)}.csv")

    fields = list(dict.fromkeys([*MANDATORY_FIELDS, *(fields or DEFAULT_FIELDS)]))
    N_arr = bg._N
    data = np.column_stack([N_arr, *(getattr(bg, f)(N_arr) for f in fields)])

    with open(path, "w") as f:
        f.write(f"# model: {bg.model.name}\n")
        f.write(f"# coupling: {bg.coupling.name}\n")
        f.write(f"# potential: {bg.potential.name}\n")
        f.write(f"# omega_m={bg.cosmo.omega_m} omega_r={bg.cosmo.omega_r} "
                f"H100={bg.cosmo.H100} M4_tilde={bg.M4_tilde}\n")
        f.write(f"# z_ini={bg.ic.z_ini} phi_ini={bg.ic.phi_ini}\n")
        f.write(f"# H0_predicted={bg.H0_predicted} Omega_m0_predicted={bg.Omega_m0_predicted} "
                f"Omega_de0_predicted={bg.Omega_de0_predicted}\n")
        f.write(",".join(["N", *fields]) + "\n")
        np.savetxt(f, data, delimiter=",")

    return path


# column order read by pysco (pysco/cosmotable.py)
PYSCO_COLUMNS = "aexp E mu_K Omega_m Omega_r py_DE"


def save_pysco_table(bg: KMouflageBackground, path: str = None, aexp_min: float = 1e-8,
                     n_points: int = 200_000, header_extra=None, overwrite: bool = False,
                     fmt: str = "%.10e") -> str:
    """Write the table pysco reads for theory = kmouflage. Returns the path.

    Columns: aexp, E = H/H0, mu_K, Omega_m, Omega_r, py_DE = a^2 p_DE / (3 calH^2),
    on n_points log-spaced values of a in [aexp_min, 1] (pysco starts its growth
    functions at a = 1e-8). E = bg.E(N) = H/H0, normalised by h_predicted: the solver works in
    units of H100, not H0.

    header_extra: extra "key=value" lines put first in the header (e.g. run
    parameters, screening tests); the header stays readable by key=value parsing.
    """
    if not hasattr(bg, "_N"):
        raise RuntimeError("bg.run() must be called before save_pysco_table().")
    a_start = float(np.exp(bg.N_ini))
    if a_start > aexp_min * (1.0 + 1e-12):
        raise ValueError(f"background starts at a={a_start:.3e} > aexp_min={aexp_min:.1e}: "
                         f"run it with z_ini >= {1.0 / aexp_min - 1.0:.1e}")

    path = path or os.path.join(DEFAULT_RUNS_DIR, f"{default_table_name(bg)}_pysco.txt")
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} already exists (pass overwrite=True)")

    N = np.linspace(np.log(aexp_min), 0.0, n_points)
    E = bg.E(N)                                       # H/H0, E(a=1) = 1

    header = "\n".join([
        *(header_extra or []),
        f"model={_base_name(bg.model)} coupling={_base_name(bg.coupling)} potential={_base_name(bg.potential)} "
        + " ".join(f"{k}={v}" for obj in (bg.model, bg.coupling, bg.potential) for k, v in obj.params.items()),
        f"omega_m={bg.cosmo.omega_m} omega_r={bg.cosmo.omega_r} M4_tilde={bg.M4_tilde} "
        f"z_ini={bg.ic.z_ini:g} phi_ini={bg.ic.phi_ini}",
        f"H0={bg.H0_predicted:.10f} Om_m0={bg.Omega_m0_predicted:.10f} "
        f"Ode0={bg.Omega_de0_predicted:.10f} mu_K0={float(bg.mu_K(0.0)):.10f}",
        # must stay << 1, otherwise the solver initial conditions (radiation era) are not valid
        f"Ode_zini={float(bg.Omega_de_def(bg._N[0])):.3e}",
        f"columns: {PYSCO_COLUMNS}   (E = H/H0, py_DE = a^2 p_DE / (3 calH^2))",
    ])

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savetxt(
        path,
        np.column_stack([np.exp(N), E, bg.mu_K(N), bg.Omega_m(N), bg.Omega_r(N), bg.py_DE(N)]),
        header=header,
        fmt=fmt,
    )
    return path


@dataclass
class LCDMReference:
    """Exact ΛCDM values, no ODE run, a clean reference for summary_table()."""
    H0_predicted:        float
    Omega_m0_predicted:  float
    Omega_r0_predicted:  float
    Omega_de0_predicted: float

    def w_de_def(self, N):
        return -1.0


def make_lcdm_reference(cosmo: CosmologicalParams, h: float = 0.6736) -> LCDMReference:
    """Builds that reference from cosmo's omega_m/omega_r at a given h (default: Planck 2018)."""
    Om0 = cosmo.omega_m / h**2
    Or0 = cosmo.omega_r / h**2
    return LCDMReference(
        H0_predicted        = h * cosmo.H100,
        Omega_m0_predicted  = Om0,
        Omega_r0_predicted  = Or0,
        Omega_de0_predicted = 1.0 - Om0 - Or0,
    )


def summary_table(models: dict, ref) -> list[dict]:
    """z=0 table of H0/Omega_m0/Omega_de0/w_de0 for each model, with its % deviation from ref."""
    def pct(val, ref_val):
        return (val - ref_val) / ref_val * 100.0

    rows = []
    for name, bg in models.items():
        H0, Om0, Ode0, wde0 = (bg.H0_predicted, bg.Omega_m0_predicted,
                                bg.Omega_de0_predicted, float(bg.w_de_def(0.0)))
        rows.append({
            "model":           name,
            "H0":              H0,
            "H0_dev_%":        pct(H0, ref.H0_predicted),
            "Omega_m0":        Om0,
            "Omega_m0_dev_%":  pct(Om0, ref.Omega_m0_predicted),
            "Omega_de0":       Ode0,
            "Omega_de0_dev_%": pct(Ode0, ref.Omega_de0_predicted),
            "w_de0":           wde0,
            "w_de0_dev_%":     pct(wde0, float(ref.w_de_def(0.0))),
        })
    return rows


def print_summary_table(rows: list[dict]) -> None:
    """Prints summary_table()'s rows nicely."""
    header = (f"{'model':<16} {'H0':>10} {'ΔH0(%)':>9}  "
              f"{'Ω_m0':>9} {'ΔΩ_m0(%)':>10}  "
              f"{'Ω_DE0':>9} {'ΔΩ_DE0(%)':>11}  "
              f"{'w_DE0':>9} {'Δw_DE0(%)':>11}")
    print(header)
    print("─" * len(header))
    for r in rows:
        print(f"{r['model']:<16} {r['H0']:>10.4f} {r['H0_dev_%']:>+9.3f}  "
              f"{r['Omega_m0']:>9.5f} {r['Omega_m0_dev_%']:>+10.3f}  "
              f"{r['Omega_de0']:>9.5f} {r['Omega_de0_dev_%']:>+11.3f}  "
              f"{r['w_de0']:>9.5f} {r['w_de0_dev_%']:>+11.3f}")
