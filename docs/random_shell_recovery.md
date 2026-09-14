# Post-BFG Stratified Random Shell Recovery

## Methodological architecture

This is an exact-validated p=30 benchmark capability, separate from the primary discovery search. Production BFG discovery remains unchanged at 5,000 evaluations. Representative recovery is an explicit post-search operation with its own evaluation count and runtime.

The validated architecture consists of two decoupled, complementary operations:

$$\text{Targeted BFG Discovery} \quad+\quad \text{Representative Uniform Random Shell Recovery}$$

### Core scientific distinction and boundaries

- **EXACT FULL ENUMERATION**: Complete evaluation of all $2^{30} = 1,073,741,824$ models. Establishes the immutable reference for global exact $Z$, global exact PMP, and global exact PIPs.
- **BFG DISCOVERY**: Budgeted targeted search ($B=5,000$ unique models). Establishes champion discovery, MAP candidate, generating-model recovery, and high-evidence genealogy.
- **RANDOM SHELL RECOVERY**: Stratified uniform random sampling from undiscovered shell remainders. Establishes representative estimates of reticular score distributions across all 31 shells.

> [!IMPORTANT]
> Random shell recovery alone does **NOT** establish exact global $Z$, exact global PMP, or exact global PIPs.

### Critical rule: No equal pooling

Targeted BFG discoveries must **NOT** be pooled as if they were random draws when estimating reticular score distributions.

For candidate model size $k$ with shell population $N_k = \binom{p}{k}$:
- Let $D_k$ denote the number of models already discovered by BFG in shell $k$.
- The undiscovered remainder size is $M_k = N_k - D_k$.
- A uniform simple random sample without replacement of size $n_k$ is drawn from the $M_k$ undiscovered models, where:
  $$n_k = \min\left(\left\lceil 0.01 N_k \right\rceil, \text{cap}, N_k - D_k\right)$$
- Discovered models receive weight:
  $$w_{\text{disc}} = \frac{1}{N_k}$$
- Random remainder models receive weight:
  $$w_{\text{samp}} = \frac{N_k - D_k}{N_k \cdot n_k}$$

The weights sum identically to 1.0. Equal pooling ($1/(D_k + n_k)$) is mathematically invalid because BFG systematically samples high-evidence models rather than uniform random models.

---

## Comparative timing and evaluation accounting

All timing values represent measured algorithm and scoring runtime on NVIDIA GeForce RTX 3060; figure rendering and disk I/O are kept strictly separate.

| Method | Unique evaluated models | Fraction of $2^{30}$ | Discovery / recovery runtime | Approx total workflow runtime | Central mean KS | Central mean W1 | MAP recovered | True model recovered |
|:---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| Exact full enumeration | 1,073,741,824 | 100.0% | 379.812 s | ~380 s | 0.000000 | 0.000000 | YES | YES |
| BFG discovery only ($B=5,000$) | 5,000 | 0.000466% | 0.361 s | 0.361 s | N/A | N/A | YES | YES |
| BFG + recovery (Cap 10,000) | 190,360 | 0.0177% | 0.361 s + 0.527 s | ~0.89 s | 0.010310 | 1.777813 | YES | YES |
| **BFG + recovery (Cap 100,000) [Recommended]** | **1,478,136** | **0.1377%** | **0.361 s + 2.890 s** | **~3.25 s** | **0.003381** | **0.552063** | **YES** | **YES** |
| BFG + recovery (Cap 1,000,000) | 8,887,594 | 0.8277% | 0.361 s + 22.489 s | ~22.85 s | 0.000705 | 0.133140 | YES | YES |

*Notes:*
- Central means evaluate common shells $k=13–17$, where all three precision caps bind.
- BFG discovery alone finds the exact global MAP ($x_1–x_{14}, x_{30}$) at evaluation $N=1,203$ and true model ($x_1–x_{15}$) at $N=2,742$, capturing 10/10 exact top-10 models and 98/100 exact top-100 models. Recovery evaluations do not revise discovery measurements.
- Recovery throughput reaches **2,359,882 models/s** in the isolated GPU scorer region.

---

## Cap recommendation: 100,000 per shell

The practical recommended default for human review is **100,000 per shell**:
- **Cap 10k** provides a fast preliminary preview (0.527 s), but central mean KS is 1.03 percentage points.
- **Cap 100k** reduces central mean KS by **67.2%** (down to 0.338 percentage points) and Wasserstein-1 distance by **68.9%** (down to 0.552 score units), adding only ~2.36 s over the 10k ladder.
- **Cap 1M** achieves sub-0.1 percentage point central CDF agreement (central mean KS = 0.000705), but requires 7.4 million additional evaluations and an additional 19.6 s.

---

## Recommended next design refinement: Small-wing exact census

Under the 1% rule, boundary shells $k=3$ and $k=27$ contain only 41 random draws ($N_3 = N_{27} = 4,060$). This leaves the tails more sensitive to sampling variability.

A natural and well-motivated next design refinement is a hybrid exact/sampling architecture:
$$\text{Exact small-shell census} \quad+\quad \text{Uniform random recovery for large shells} \quad+\quad \text{Targeted BFG discovery}$$

Combinatorial costs for p=30:
- Exact census through $k \le 3$ and $k \ge 27$:
  $$2 \times \left(\binom{30}{0} + \binom{30}{1} + \binom{30}{2} + \binom{30}{3}\right) = 2 \times (1 + 30 + 435 + 4,060) = 9,052 \text{ models total}.$$
- Exact census through $k \le 4$ and $k \ge 26$:
  $$9,052 + 2 \times \binom{30}{4} = 9,052 + 2 \times 27,405 = 63,862 \text{ models total}.$$

This is documented as a **recommended next design refinement / experimental option**. Production BFG defaults, genealogy, and search budgets remain unchanged.

---

## Reusable public module

The validated recovery logic is exposed in the public module `gpubma.bfg.shell_recovery`:
- `shell_population_size(p, k)`: Returns $\binom{p}{k}$.
- `sample_size(p, k, discovered, cap)`: Evaluates the 1% capped sampling law.
- `rank_indices`, `rank_mask`, `unrank_combination`, `mask_from_rank`: Combinatorial ranking/unranking with arbitrary-precision integer support.
- `RemainderSampler`: Deterministic nested uniform without-replacement sampling from non-excluded shell remainders.
- `RecoveryScorer`: High-throughput GPU/CPU batched scoring without dictionary-cache overhead.
- `reconstruct_shell_mixture`: Implements the representative mixture weighting formula.
- `weighted_summary`, `weighted_distances`: Exact discrete-CDF KS and Wasserstein-1 distance computations.
