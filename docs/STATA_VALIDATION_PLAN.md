# Stata validation plan

Stata is an **external oracle only** — never a runtime dependency.

## Status on the development machine (2026-07-15) — EXECUTED

The oracle ran via `C:\Program Files\StataNow19\StataSE-64.exe` in batch
mode (`/e do <script>`): **Stata 19.5 (StataNow, SE), bmaregress 1.0.2
(16jul2023)**. All five `.do` scripts executed cleanly (exit code 0, no
r(###) errors) and exported e(b_bma), e(pip), vecdiag(e(V_bma)) and key e()
scalars at full double precision to `validation/stata/output/`.
`scripts/compare_stata_python.py` then compared Python against those real
exports: **all six designs PASS**, worst absolute difference 1.8e-12 across
264 quantities. Logs are plain text and were scrubbed of serial-number and
license-holder information (batch banner logs at the repository root are
deleted and git-ignored).

Historical note: `C:\Program Files\Stata17/18` contain only renamed
`StataSE-64_old.exe` leftovers that produce no batch output; they are
recorded as not callable by `gpubma doctor`.

## Scripts

| script | data | design |
|---|---|---|
| small_no_fixed_effects.do | panel_8.dta | y ~ x1-x8, always: w1 w2 |
| small_individual_fixed_effects.do | panel_8.dta | + i.individual_id always |
| small_time_fixed_effects.do | panel_8.dta | + i.period always |
| small_two_way_fixed_effects.do | panel_8.dta | + both factor sets always |
| grunfeld_validation.do | grunfeld.dta (original Stata Press bytes) | invest ~ mvalue kstock; + i.company variant |

Each script: loads the frozen `.dta` written by Python; asserts row count,
variable names, and key uniqueness; runs `bmaregress` with **enumeration
explicitly requested**; pins documented fixed priors (`gprior(fixed g)` with
g equal to the Python default for that dataset, `mprior(betabinomial 1 1)`);
records `c(stata_version)` and `about`; dumps `ereturn list`,
`bmastats models/pip/msize`; exports `e(b)` (and `e(pip)` inside a `capture`
block) to CSV via `svmat` + `export delimited`; saves a plain-text log.
No numerical output is ever copied by hand.

## Items resolved by the executed runs

1. Option keywords verified working as written: `enumeration`,
   `gprior(fixed #)`, `mprior(betabinomial 1 1)`, `(varlist, always)`.
2. Stored-result names confirmed: there is **no `e(b)`**; the matrices are
   `e(b_bma)` (posterior means), `e(V_bma)` (posterior covariance; its
   diagonal gives the reported sds), `e(pip)`, plus `e(b_bma_c)`/`e(V_bma_c)`
   (conditional-on-inclusion) and `e(group)`.
3. Factor-variable expansion in the `always` group uses base level = lowest
   value, matching Python's dropped first-sorted-level dummies (coefficient
   name lists in `*_colnames.txt` confirm `1b.` bases).
4. Stata counts always-included variables in model size:
   `e(msize_mean) − e(p_always)` equals Python's optional-only mean size.
5. Stata's always-block prior is JOINT g-shrinkage with df = n − 1
   (`always_prior="shrink"` in gpubma); see
   docs/STATISTICAL_SPECIFICATION.md.

## Execution procedure (reproducible)

From a shell at the repository root (as actually executed):

```powershell
& "C:\Program Files\StataNow19\StataSE-64.exe" /e do "validation\stata\small_no_fixed_effects.do"
# ... likewise for the other four scripts; each exits on its own.
# Delete the banner logs Stata /e drops at the repo root (they contain
# license info and are git-ignored): Remove-Item .\*.log
```

Then run `python scripts/compare_stata_python.py`, which picks up the CSV
exports under `validation/stata/output/` and produces an unrounded
comparison report (quantity, reference, candidate, absolute difference,
relative difference, tolerance, pass/fail). The tolerance is 1e-9 (observed
worst difference: 1.8e-12) and must not be loosened merely to pass
(CLAUDE.md rule 9); a discrepancy beyond it indicates a real
parameterization difference to investigate.
