# Stata validation plan

Stata is an **external oracle only** — never a runtime dependency.

## Status on the development machine (2026-07-15)

`C:\Program Files\Stata17` and `...\Stata18` exist but contain only renamed
`StataSE-64_old.exe` leftovers plus a license file. Batch execution
(`StataSE-64_old.exe /e do file.do`) was actually attempted for both and
produced no log and no process — **treated as not callable**. Consequently
every `.do` script in `validation/stata/` is **prepared, not executed**, and
no number in this repository comes from Stata.

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

## Known limitations (documented, not fabricated)

1. Option keywords (`enumeration`, `gprior(fixed #)`, `mprior(betabinomial …)`)
   follow the [BMA] bmaregress manual and StataCorp's Stata 18 release notes,
   but could not be executed here; the first real run must verify them
   against `help bmaregress`.
2. The exact `e()` name of the PIP matrix is unconfirmed; the script logs
   `ereturn list` so the name can be read off the first run.
3. Whether `bmaregress` factor-variable expansion in an `always` group uses
   base level = lowest value (matching Python's dropped first sorted level)
   must be confirmed from the executed log.

## Execution procedure (when a working Stata 18+ is available)

```stata
cd <repository root>
do validation/stata/small_no_fixed_effects.do
do validation/stata/small_individual_fixed_effects.do
do validation/stata/small_time_fixed_effects.do
do validation/stata/small_two_way_fixed_effects.do
do validation/stata/grunfeld_validation.do
```

Then run `python scripts/compare_stata_python.py`, which picks up the CSV
exports under `validation/stata/output/` and produces an unrounded
comparison report (quantity, reference, candidate, absolute difference,
relative difference, tolerance, pass/fail). The provisional tolerance is
1e-6 and must not be loosened merely to pass (CLAUDE.md rule 9); a
discrepancy beyond it indicates a parameterization difference to resolve.
