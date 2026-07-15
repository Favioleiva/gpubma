*----------------------------------------------------------------------
* GPUBMA Phase 1 — Stata validation oracle (public benchmark data)
* Dataset : data/public/grunfeld.dta — the ORIGINAL Stata Press file
*           (https://www.stata-press.com/data/r18/grunfeld.dta), so Stata
*           and Python share bit-identical observations.
* Model   : invest on optional {mvalue, kstock} => 2^2 = 4 candidate models
*           plus an individual (company) fixed-effects variant.
*
* STATUS: PREPARED, NOT YET EXECUTED (no working Stata on the dev machine).
* Requires Stata 18+ (bmaregress). Verify option keywords via help bmaregress.
*
* Priors fixed and documented: n = 200, p = 2 => benchmark g = max(200, 4)
* = 200; model prior beta-binomial(1,1). These match the Python reference.
*----------------------------------------------------------------------
version 18
clear all
set more off
set linesize 120
capture mkdir "validation/stata/output"
log using "validation/stata/output/grunfeld_validation.log", replace text

display "Stata version: " c(stata_version)
about

use "data/public/grunfeld.dta", clear
assert _N == 200
confirm variable company year invest mvalue kstock
isid company year
describe
summarize

* -- no fixed effects ---------------------------------------------------
bmaregress invest mvalue kstock, ///
    enumeration ///
    gprior(fixed 200) ///
    mprior(betabinomial 1 1)

ereturn list
bmastats models, top(4)
bmastats pip
bmastats msize

matrix b = e(b)
preserve
clear
svmat double b, names(col)
export delimited using "validation/stata/output/grunfeld_no_fe_e_b.csv", replace
restore

* -- company (individual) fixed effects, always included ---------------
bmaregress invest mvalue kstock (i.company, always), ///
    enumeration ///
    gprior(fixed 200) ///
    mprior(betabinomial 1 1)

ereturn list
bmastats models, top(4)
bmastats pip

matrix b2 = e(b)
preserve
clear
svmat double b2, names(col)
export delimited using "validation/stata/output/grunfeld_company_fe_e_b.csv", replace
restore

capture noisily {
    matrix PIP = e(pip)
    preserve
    clear
    svmat double PIP, names(col)
    export delimited using "validation/stata/output/grunfeld_pip.csv", replace
    restore
}

log close
exit, clear
