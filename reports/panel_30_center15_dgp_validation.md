# panel_30_center15 — DGP validation report

Audited from the canonical Parquet artifact (`data/synthetic/panel_30_center15.parquet`). Local phase only — **the full p = 30 exhaustive enumeration was NOT run**.

## Structure
- 2,000 observations x 35 columns; 30 candidates (15 true x1-x15, 15 proxies x16-x30); no missing/non-finite values; dtypes as frozen.
- Parquet SHA-256 matches metadata: `7b468ca0c09249a8…`
- Deterministic regeneration (seed 20260724): numerically exact; byte-identical in this environment: True.

## Linear algebra
- rank(X) = rank(residualized X) = 30.
- singular values of X: [13.702, 112.103]; cond(X) = 8.18; cond(X'X) = 66.94.
- residualized Gram eigenvalues [187.5, 12510.3], cond = 66.74; float64 Cholesky OK.
- cond(X'X) ~ 66.9 is ~14 orders of magnitude below the float64 danger zone (~1e15); Cholesky-based enumeration loses < 2 decimal digits worst-case. No stability concern.

## Correlation design
| block | mean abs r | min | max |
|---|---|---|---|
| A (true, within) | 0.521 | 0.400 | 0.662 |
| B (true, within) | 0.552 | 0.459 | 0.664 |
| C (true, within) | 0.508 | 0.397 | 0.639 |
| across blocks (true) | 0.017 | — | 0.046 |

- proxy-primary correlations: [0.792, 0.887]; no near-duplicates.
- strongest unintended (cross-block) |corr|: 0.056.

### Proxy-to-source mapping
| proxy | primary | partner | a | b | noise | r(primary) | r(partner) |
|---|---|---|---|---|---|---|---|
| x16 | x1 | x2 | 0.88 | 0.15 | 0.50 | 0.884 | 0.625 |
| x17 | x2 | x3 | 0.74 | 0.32 | 0.62 | 0.792 | 0.597 |
| x18 | x3 | x4 | 0.81 | 0.10 | 0.45 | 0.881 | 0.517 |
| x19 | x4 | x5 | 0.70 | 0.27 | 0.58 | 0.803 | 0.603 |
| x20 | x5 | x1 | 0.85 | 0.20 | 0.52 | 0.874 | 0.563 |
| x21 | x6 | x7 | 0.77 | 0.34 | 0.47 | 0.876 | 0.669 |
| x22 | x7 | x8 | 0.90 | 0.12 | 0.60 | 0.843 | 0.488 |
| x23 | x8 | x9 | 0.72 | 0.25 | 0.55 | 0.826 | 0.611 |
| x24 | x9 | x10 | 0.79 | 0.18 | 0.49 | 0.869 | 0.587 |
| x25 | x10 | x6 | 0.86 | 0.30 | 0.63 | 0.845 | 0.696 |
| x26 | x11 | x12 | 0.71 | 0.14 | 0.53 | 0.818 | 0.534 |
| x27 | x12 | x13 | 0.83 | 0.22 | 0.46 | 0.887 | 0.638 |
| x28 | x13 | x14 | 0.76 | 0.35 | 0.59 | 0.824 | 0.645 |
| x29 | x14 | x15 | 0.89 | 0.11 | 0.51 | 0.880 | 0.559 |
| x30 | x15 | x11 | 0.73 | 0.28 | 0.56 | 0.804 | 0.558 |

- Proxies are strong imperfect substitutes: |corr(proxy, primary)| in [0.792, 0.887] (never >= 0.995), and the 15 proxies ALONE recover 65.4% of the true model's R^2 (0.4577 vs 0.7000). This creates genuine posterior model competition rather than duplicate columns or ignorable noise.

## Realized signal
- realized R^2 (true spec [1, w1, w2, x1-x15]): **0.700000** (target 0.695-0.705).
- Var(X beta) = 4.6547; Var(W delta) = 1.0479; Var(signal) = 5.8543; Var(eps) = 2.5103; Var(y) = 8.2287.
- SNR = Var(signal)/Var(eps) = **2.3321** (consistent with R^2/(1-R^2) = 2.3333).

| OLS model | R^2 |
|---|---|
| true 15 (x1-x15) | 0.7000 |
| full 30 | 0.7030 |
| all 15 proxies replace all true | 0.4577 |
| single swap x1 -> x16 | 0.6697 |
| block A -> its proxies | 0.5469 |

- OLS on the true model recovers beta within max abs deviation 0.1066.
- Replacing ALL 15 true variables by their proxies retains R^2 = 0.4577 of the true model's 0.7000; a single swap (x1 -> x16) costs only 0.0303 R^2. Proxies imitate the signal closely enough that model selection cannot rely on fit alone — exactly the intended difficulty.

## Verdict
The frozen artifact matches its intended design exactly: central-layer size-15 truth, block-correlated true regressors, strong-but-imperfect structural-zero proxies, R^2 on target, and conditioning far inside float64 Cholesky safety margins.
