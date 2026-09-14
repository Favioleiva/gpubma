# B_total and the public BUDGET setting

**BUDGET is the maximum number of unique candidate models BFG may score.**

`B_total = BUDGET` maps directly to `BFGConfig(budget_models=BUDGET, budget_semantics="hard")`. `budget_models` is the public API field. `max_eval_budget` is its internal scorer argument, not a second public budget. There is no `budget` alias in this API.

The engine constructs one `BFGScorer` and passes that object to wings, forward/backward genealogy, beam and elite expansion; reconnaissance uses the same scorer. The scorer removes cached and intra-batch duplicate masks, truncates unseen requests to `max_eval_budget - len(cache)`, and never scores beyond that remainder. Repeated visits use the cache. Thus `N_unique_scored <= B_total` applies to the **whole search**, not separately to each stage. The discovery release accepts only hard semantics; an older experimental source supported `sampling_only`, which the public examples do not use.

The ceiling is not a requirement to consume every slot. BFG can finish early when the candidate universe or its available search work is exhausted. For Grunfeld's two candidates there are only four masks; BUDGET=5000 still produces at most four unique evaluations.

Before fitting the notebooks print p, `2**p`, BUDGET and the maximum possible explored fraction `min(BUDGET, 2**p)/2**p`. After fitting they print the requested budget, actual unique count and actual fraction, and assert the hard ceiling. For canonical p30, 5000/2^30 is 0.0004656612873077393%, displayed as 0.000466%.

To choose another budget, edit the single configuration cell. Examples are `1_000`, `5_000`, `10_000`, and `50_000`; no notebook automatically runs this sequence. Increasing the ceiling permits more search and can change the discovered set. It neither enumerates all models nor guarantees a particular posterior-mass fraction or champion recovery.

`DEVICE="auto"` is resolved by the shared example helper to CUDA if PyTorch reports CUDA available, otherwise CPU. The unmodified engine still records actual hardware/fallback. Runtime summaries include versions and the selected device. Refits for the regression table operate only on already evaluated masks and are timed separately; they do not add new candidate scores or rewrite search outputs.
