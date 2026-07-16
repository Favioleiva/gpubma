*----------------------------------------------------------------------
* GPUBMA Phase 1 — Stata validation oracle (public benchmark data)
* Dataset : data/public/grunfeld.dta — the ORIGINAL Stata Press file
*           (https://www.stata-press.com/data/r18/grunfeld.dta), so Stata
*           and Python share bit-identical observations.
* Model   : invest on optional {mvalue, kstock} => 2^2 = 4 candidate models
*           plus an individual (company) fixed-effects variant.
*
* STATUS: EXECUTED 2026-07-15 in batch mode on StataNow/SE 19.5.
* Option keywords and e() matrix names verified on that run.
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
display "Edition: " c(edition_real)
display "Born: " c(born_date)

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

* -- machine-readable exports (no manual copying of output) ------------
* Matrix names confirmed from `ereturn list` on Stata 19.5 (StataNow, SE):
* e(b_bma) = BMA posterior means, e(V_bma) = BMA posterior covariance,
* e(pip) = posterior inclusion probabilities. There is no e(b).
matrix b_bma = e(b_bma)
matrix pip = e(pip)
matrix vdiag = vecdiag(e(V_bma))
local names : colnames e(b_bma)
file open fh using "validation/stata/output/grunfeld_no_fe_colnames.txt", write replace
file write fh `"`names'"'
file close fh
preserve
clear
svmat double b_bma
export delimited using "validation/stata/output/grunfeld_no_fe_b_bma.csv", replace
clear
svmat double pip
export delimited using "validation/stata/output/grunfeld_no_fe_pip.csv", replace
clear
svmat double vdiag
export delimited using "validation/stata/output/grunfeld_no_fe_v_diag.csv", replace
clear
set obs 1
gen double msize_mean = e(msize_mean)
gen double msize_mean_prior = e(msize_mean_prior)
gen double p_always = e(p_always)
gen double p_groups = e(p_groups)
gen double g_used = e(g)
gen double shrinkage = e(shrinkage)
gen double sigma2 = e(sigma2)
gen double k_models = e(k_models)
gen double n_obs = e(N)
export delimited using "validation/stata/output/grunfeld_no_fe_scalars.csv", replace
restore

* -- company (individual) fixed effects, always included ---------------
bmaregress invest mvalue kstock (i.company, always), ///
    enumeration ///
    gprior(fixed 200) ///
    mprior(betabinomial 1 1)

ereturn list
bmastats models, top(4)
bmastats pip

* -- machine-readable exports (no manual copying of output) ------------
* Matrix names confirmed from `ereturn list` on Stata 19.5 (StataNow, SE):
* e(b_bma) = BMA posterior means, e(V_bma) = BMA posterior covariance,
* e(pip) = posterior inclusion probabilities. There is no e(b).
matrix b_bma = e(b_bma)
matrix pip = e(pip)
matrix vdiag = vecdiag(e(V_bma))
local names : colnames e(b_bma)
file open fh using "validation/stata/output/grunfeld_company_fe_colnames.txt", write replace
file write fh `"`names'"'
file close fh
preserve
clear
svmat double b_bma
export delimited using "validation/stata/output/grunfeld_company_fe_b_bma.csv", replace
clear
svmat double pip
export delimited using "validation/stata/output/grunfeld_company_fe_pip.csv", replace
clear
svmat double vdiag
export delimited using "validation/stata/output/grunfeld_company_fe_v_diag.csv", replace
clear
set obs 1
gen double msize_mean = e(msize_mean)
gen double msize_mean_prior = e(msize_mean_prior)
gen double p_always = e(p_always)
gen double p_groups = e(p_groups)
gen double g_used = e(g)
gen double shrinkage = e(shrinkage)
gen double sigma2 = e(sigma2)
gen double k_models = e(k_models)
gen double n_obs = e(N)
export delimited using "validation/stata/output/grunfeld_company_fe_scalars.csv", replace
restore

log close
exit, clear
