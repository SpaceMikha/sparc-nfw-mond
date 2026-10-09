# sparc-nfw-mond

Code for the paper

> M. Da Silva, *Cusps, cores, and one acceleration: dark-matter haloes versus
> modified dynamics across the full SPARC sample*,
> [arXiv:2610.05871](https://arxiv.org/abs/2610.05871)
> (also [doi:10.5281/zenodo.22342521](https://doi.org/10.5281/zenodo.22342521)).

It fits the rotation curves of the 175 SPARC galaxies with three mass models
under identical priors (a free NFW halo, an NFW halo constrained by the
ΛCDM concentration–mass relation, and MOND with a fixed acceleration
constant), compares them with the Bayesian information criterion, rebuilds
the radial acceleration relation, and profiles the acceleration constant
galaxy by galaxy.

## Requirements

Python 3.10 or later and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

## Running

All commands are run from the root of the repository.

```
python scripts/download_data.py   # SPARC master table and mass models -> data/
python scripts/run_fits.py        # 175 galaxies x 3 models  -> results/fits.csv
python scripts/run_a0.py          # a0 profile per galaxy    -> results/a0.csv
python scripts/analyse.py         # tables and numbers       -> results/
python scripts/make_figures.py    # the eight figures        -> figs/
```

On two cores the fits take about ten minutes and the `a0` profiles a few
minutes more. `run_fits.py` and `run_a0.py` accept `--processes N`.

## What is in `results/`

| File | Content |
| --- | --- |
| `fits.csv` | One row per galaxy and model: χ², BIC, AIC, the maximum-a-posteriori parameters and the 16/50/84 percentiles of every parameter |
| `a0.csv` | Best-fitting log a₀ of each galaxy and its profile uncertainty |
| `table_fits.csv` | The per-galaxy table of the appendix |
| `summary.txt` | Every number quoted in the paper, next to the published value |
| `comparison.csv` | Galaxy-by-galaxy differences from the published table (`paper/table_published.csv`) |

## Layout

```
sparcfit/data.py     reading the SPARC files
sparcfit/models.py   mass models, priors, likelihood (Eqs. 1-7 of the paper)
sparcfit/fit.py      Nelder-Mead maximisation, emcee sampling, a0 profile
sparcfit/rar.py      radial acceleration relation
scripts/             the commands listed above
paper/               the published table, for comparison
```

## Method in brief

- Baryons: `V_bar² = Υ_disk V_disk² + Υ_bul V_bul² + V_gas |V_gas|`, with
  lognormal priors on Υ_disk (0.5) and Υ_bul (0.7) of width 0.1 dex and hard
  limits 0.1 ≤ Υ ≤ 10.
- Distance factor `d` and inclination `i` are sampled with Gaussian priors
  from the SPARC uncertainties, truncated at ±3σ. Distance rescales the radii
  and the baryonic velocities; inclination rescales the observed velocities.
- NFW: flat priors 0 ≤ log c ≤ 2 and 1 ≤ log V₂₀₀ ≤ 2.9, with
  H₀ = 73 km s⁻¹ Mpc⁻¹. The ΛCDM version adds the Dutton & Macciò (2014)
  relation with 0.11 dex scatter and h = 0.671.
- MOND: simple interpolation function, a₀ = 1.2 × 10⁻¹⁰ m s⁻².
- Each posterior is maximised with Nelder–Mead (nine starting points for the
  halo models) and sampled with `emcee` (32 walkers, 1200 steps, 400
  discarded). Random seeds are fixed per galaxy and model, so a run is
  reproducible on a given machine.

## Extension: the CCC model

The repository also contains a fit of the covarying-coupling-constants (CCC)
model of Gupta & Samaras ([arXiv:2608.11575](https://arxiv.org/abs/2608.11575)).
It is not part of the paper and does not change any of its results.

```
python scripts/run_fits.py --models ccc --output fits_ccc.csv
python scripts/analyse_ccc.py     # -> results/summary_ccc.txt, results/table_ccc.csv
```

Gupta & Samaras define the model in the inverse direction: the observed
rotation curve is turned into a density and used to predict the baryons, with
`rho_obs = rho_bar * nu(rho_bar / rho_t)` in the spherical approximation.
The implementation here (`vccc2` in `sparcfit/models.py`) applies the same
relation forwards, so that CCC goes through the same likelihood and priors as
the other models:

- the baryonic mass of each radial shell, `dM = d(V_bar^2 R) / G`, is boosted
  by `nu` evaluated at the mean density of the shell, and the boosted shells
  are summed to give the model velocity;
- `nu` is the "standard" smooth function, `1 / (1 - exp(-sqrt(y)))`;
- the turn-off density `rho_t` is free in every galaxy, with a flat prior on
  `log10(rho_t / g cm^-3)` between -27 and -21, and counts as one parameter;
- no smoothing is applied to the baryonic curves, so shells in which
  `V_bar^2 R` decreases carry negative mass.

The forward and inverse formulations are not equivalent in the presence of
noise and of the spherical approximation, so the numbers in
`results/summary_ccc.txt` are not directly comparable with those of Gupta &
Samaras.

## Reproducibility

The sampler is stochastic, so numbers that depend on posterior medians can
move in the last digit between software versions. `results/summary.txt`
lists the published and recomputed values side by side; the preferred model
is the same for all 175 galaxies.

## Data

The SPARC database is described in Lelli, McGaugh & Schombert (2016, AJ 152,
157) and is available at <https://astroweb.case.edu/SPARC/>. It is downloaded
by `scripts/download_data.py` and is not redistributed here.

## Citation

If you use this code, please cite the paper above.

## License

MIT, see `LICENSE`.
