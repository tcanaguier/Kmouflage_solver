# K-mouflage Background

A small Python solver for K-mouflage models, a modified gravity theory built
on a scalar field with a non standard kinetic term K(X) and a coupling A(φ) to
matter, screened by the K-mouflage mechanism. The solver also accepts a
potential f(φ), which extends it to Kinessence models. It computes how the
universe expands (Hubble rate, densities, equations of state) and how
structures grow, and it writes the tables used by the N-body code pysco.

## K-mouflage and its Kinessence extension

The scalar field Lagrangian is

    A⁻⁴ M⁴ [ K(X) − f(φ) ]

K-mouflage is the case f = 1. This is what you get when you don't give a
potential, so by default the solver runs K-mouflage. Giving a potential f(φ)
extends it to a Kinessence model.

Write K(X) with K(0) = 0, for example K(X) = X + K0 X³. The usual K-mouflage
papers write K(X) = −1 + X + ..., and that −1 is exactly our f = 1. Don't add
it to K yourself, or you would count the vacuum energy twice.

## Install

Clone the repo and import `kmouflage` from it. The package needs `numpy` and
`scipy`, the notebooks also use `matplotlib`.

## Quick example

```python
from kmouflage import (
    KMouflageBackground, CosmologicalParams,
    make_powerlaw_K, make_exponential_coupling,
)

model    = make_powerlaw_K(K0=1, m=3)
coupling = make_exponential_coupling(beta=0.1)
cosmo    = CosmologicalParams(omega_m=0.1430, omega_r=4.15e-5, M4_tilde=1.0)

bg = KMouflageBackground(model=model, coupling=coupling, cosmo=cosmo)
bg.run()

N = 0.0          # N = ln(a), N = 0 is today
bg.phi(N)        # scalar field
bg.Omega_m(N)    # matter density parameter
```

`run_background(model, coupling, potential=None, cosmo=None, ic=None)` does the
same thing in one call.

## Inputs and outputs

You choose the model (K, the coupling A(φ), the potential f(φ)) and the
cosmology: `omega_m`, `omega_r` and `M4_tilde`. The potential is optional. H0, Omega_m0 and Omega_de0 then
come out as predictions (`bg.H0_predicted`, `bg.Omega_m0_predicted`,
`bg.Omega_de0_predicted`).

We don't calibrate M4_tilde by default. If you really want a given H0,
`calibrate_M4_tilde` can tune it for you.

All quantities are in units of H100 = 100 km/s/Mpc, with M_Pl = 1. So `bg.H(N)`
is H divided by 100 km/s/Mpc, and today it equals h. For plots, use
`bg.E(N)` for H/H0 (equal to 1 today) and `bg.H_kms(N)` for H in km/s/Mpc.

## ΛCDM

ΛCDM is the model with no coupling and a constant potential. The field then
stays frozen and Λ = M4_tilde:

```python
from kmouflage import run_background, make_LambdaCDM_K, make_LambdaCDM_coupling

bg_lcdm = run_background(make_LambdaCDM_K(), make_LambdaCDM_coupling(), cosmo=cosmo)
```

Using the same `cosmo` as your run gives a ΛCDM with exactly the same
inputs, which is the natural reference.

## How it is solved

The background follows the Klein-Gordon equation written in its demixed form
together with the first Friedmann equation. The state is the field φ and its
conjugate momentum π/a² = A⁻² K'(X) φ', integrated in N = ln a. At each step φ'
comes from π, the Hubble rate comes from the Friedmann constraint, and nothing
has to be inverted. The initial momentum is computed analytically deep in the
radiation era, so the run can start as early as you want (`z_ini`, 1e5 by
default).

`KMouflageGrowth(bg)` then gives the linear growth rate f and the growth
factor D+.

## Saving a run

`save_table(bg)` writes one CSV with a commented header (model, coupling,
potential, cosmology) and the fields you choose as columns. Its default name is
built from the model parameters and M4_tilde.

`save_pysco_table(bg, path)` writes the table pysco reads: aexp, E = H/H0,
mu_K, Omega_m, Omega_r and py_DE, from a = 1e-8 to today. The background must
start early enough for that, so run it with `z_ini = 1e8`.

```python
from kmouflage import InitialConditions, run_background, save_pysco_table

bg = run_background(model, coupling, cosmo=cosmo, ic=InitialConditions(z_ini=1e8))
save_pysco_table(bg, "kmouflage_table.txt")
```

## Papers

The K-mouflage case is the one studied in these papers.

Brax & Valageas 2014, K-mouflage cosmology: the background evolution,
arXiv:1403.5420.

Barreira, Brax, Clesse, Li & Valageas 2015, K-mouflage gravity models that pass
Solar System and cosmological constraints, arXiv:1504.01493.

## Checking a run

`verify(bg)` prints a table of checks with their values and PASS or FAIL: the
numerical accuracy of the run (numerical derivatives of φ, φ' and H against the
equations, Friedmann, sum of the Omegas), the stability of the field (no ghost,
c_s² > 0) and the Solar System tests (Cassini, Moon periastron) computed with
the M4_tilde of the run. `verify_screening(model, coupling, M4_tilde)` runs only
the Solar System part.

## Status

Still a work in progress. There are no automated tests or packaging yet.
