# GPUBMA — Non-negotiable project rules

1. `gpubma` is a Python open-source package (BSD-3-Clause). Stata is **never** a runtime dependency.
2. Stata is validation-only: an external oracle run on small frozen datasets.
3. Exhaustive enumeration of all `2^p` models is the long-term target.
4. Never replace enumeration with MC3 (or any sampler) without explicit authorization from the user.
5. Never assume the GPU model. Detect it (`gpubma doctor`) and record the actual device.
6. Never silently use float32. Precision must be explicit; float64 is the default reference precision.
7. Never report projections as measurements. Benchmark reports must label every number as
   `Measured`, `Projected`, or `Not evaluated`.
8. Never optimize an unvalidated formula. The CPU reference must be correct before GPU work.
9. Never loosen tolerances merely to pass tests. Investigate the discrepancy instead.
10. Keep generated data reproducible from scripts and fixed seeds (default seed: `20260715`).
11. Preserve checksums and provenance for every frozen data artifact.
12. Record unresolved statistical questions honestly in `STATUS.md` (e.g., the exact Stata
    `bmaregress` prior parameterization is provisional until validated against real Stata output).

## Practical notes

- Package layout: `src/gpubma/`; tests under `tests/`; frozen data under `data/`.
- Regenerate synthetic data: `python scripts/generate_synthetic_panels.py`.
- Regenerate Grunfeld snapshot: `python scripts/download_grunfeld.py`.
- Diagnostics: `python -m gpubma.doctor [--json reports/gpu_doctor.json]`.
- Benchmarks: `python -m gpubma.benchmark --max-predictors 15`.
- Tests: `python -m pytest` (GPU tests skip cleanly when CUDA is unavailable, with the skip reason printed).
- Stata oracle (validation only): `STATA_EXE = C:\Program Files\StataNow19\StataSE-64.exe`,
  batch mode `/e do <script>` from the repo root. After running, delete the
  banner logs Stata drops at the repo root and never commit serial-number or
  license-holder text (scan before committing; a test enforces this for
  `validation/stata/output/`).
- The `2^30` production run and the final CUDA enumerator are **out of scope** until Phase 1
  acceptance criteria pass and the user authorizes Phase 2.

# CONTRACT EXECUTION PROTOCOL

## Automatic Contract Invocation

When the researcher gives an instruction consisting of or containing a contract reference such as:

`Contract 1`

`Contract 2`

`Contract 15`

or equivalent wording such as:

`execute Contract 1`

`continue Contract 1`

`work on Contract 1`

you must interpret this as an instruction to **locate, read, and execute that contract autonomously**.

The researcher must not be required to repeat the execution protocol in the CLI.

---

## 1. Contract Discovery

For a requested contract number `N`, first search:

`contracts/`

for the corresponding contract specification.

Canonical expected naming convention:

`contracts/ContractN.txt`

Examples:

`Contract 1` → `contracts/Contract1.txt`

`Contract 7` → `contracts/Contract7.txt`

`Contract 30` → `contracts/Contract30.txt`

If an exact canonical filename is not found, search the `contracts/` directory for the closest unambiguous contract file matching that number.

Do not ask the researcher to provide the path if it can be discovered from the project.

---

## 2. Read Before Acting

Before executing any contract:

1. read the complete contract specification;
2. read this `CLAUDE.md`;
3. read any other governing project instructions referenced by either file;
4. inspect relevant repository documentation and existing project state;
5. determine whether the contract is new, partially executed, awaiting approval, or already complete.

Do not begin implementation based only on the contract number or conversation context.

The contract file is the authoritative task specification.

---

## 3. Independent Workspace for Every Contract

Every contract must have its own independent working directory.

For `ContractN`, use:

`contracts/ContractN/`

unless the contract itself explicitly defines another contract-local workspace.

Example:

```text
contracts/
├── Contract1.txt
├── Contract1/
│   ├── code/
│   ├── data/
│   ├── figures/
│   ├── tables/
│   ├── logs/
│   ├── validation/
│   ├── reports/
│   └── state/
│
├── Contract2.txt
├── Contract2/
│   ├── code/
│   ├── data/
│   ├── figures/
│   ├── tables/
│   ├── logs/
│   ├── validation/
│   ├── reports/
│   └── state/
```

Create only the subdirectories actually needed by the contract.

The essential rule is:

> **One contract = one independent and reproducible workspace.**

Do not scatter contract-specific experimental outputs across the repository.

Do not mix outputs from different contracts.

Do not overwrite artifacts belonging to previous contracts.

---

## 4. Contract Specification vs Contract Workspace

The specification remains at:

`contracts/ContractN.txt`

The execution workspace is:

`contracts/ContractN/`

These serve different functions.

### Specification

Defines what must be done.

### Workspace

Contains the actual execution state and deliverables.

The workspace may contain:

* scripts;
* experimental code;
* sampled model IDs;
* intermediate data;
* cached calculations;
* figures;
* PNG/JPG outputs;
* tables;
* logs;
* benchmark outputs;
* random seeds;
* QA results;
* validation results;
* notes;
* execution reports;
* final deliverables.

Do not move the original contract specification into the workspace unless explicitly required.

---

## 5. Resume Existing Work Automatically

When `contracts/ContractN/` already exists, do not automatically restart the contract.

Inspect the existing workspace first.

Determine:

* what has already been completed;
* which outputs are valid;
* which tasks remain;
* whether a previous execution stopped because of an error;
* whether the contract is waiting at a mandatory researcher checkpoint.

Reuse valid completed work.

Resume from the latest scientifically valid state.

Do not recompute expensive work merely because the agent session is new.

---

## 6. Persistent Execution State

Each contract workspace should maintain a concise machine/human-readable execution state.

Preferred file:

`contracts/ContractN/state/STATUS.md`

It should record at minimum:

* contract number;
* current status;
* tasks completed;
* task currently being executed;
* tasks remaining;
* important output paths;
* blocking issues if any;
* repository commit/hash where relevant;
* last meaningful execution checkpoint.

Update this file during substantial progress.

This allows a later agent session to recover the contract state without reconstructing everything from conversation history.

---

## 7. Autonomous Execution

When instructed to execute `Contract N`, do not merely:

* summarize the contract;
* explain what should be done;
* generate a proposed plan;
* stop after the first successful script;
* stop after the first figure;
* stop after a preliminary result.

Execute the contract.

Continue through all executable tasks defined by the specification.

Resolve ordinary implementation issues autonomously by inspecting:

* code;
* documentation;
* repository history/state;
* data;
* logs;
* tests;
* outputs.

A failed method is not automatically a reason to stop.

Diagnose the failure and proceed with the next scientifically defensible action permitted by the contract.

---

## 8. Repository Code vs Contract Artifacts

Reusable general-purpose functionality belongs in the appropriate main project directories such as:

* `src/`
* `tests/`
* `scripts/`
* `docs/`

when justified.

Contract-specific experimental artifacts belong in:

`contracts/ContractN/`

Do not duplicate general project functionality unnecessarily inside contract workspaces.

Likewise, do not pollute the general source tree with one-off experimental outputs.

---

## 9. Preserve Existing Functionality

Before modifying general repository code:

1. inspect the existing implementation;
2. determine whether the required functionality already exists;
3. reuse validated code whenever possible;
4. make the smallest coherent modification required;
5. preserve backwards compatibility unless the contract explicitly requires otherwise;
6. run relevant tests;
7. document material changes.

Never rewrite working canonical functionality simply because a new contract uses it.

---

## 10. Reproducibility

Contract execution must be reproducible.

Where applicable record:

* random seeds;
* input files;
* data hashes;
* repository commit;
* software versions;
* hardware;
* parameters;
* model specifications;
* execution commands or callable scripts;
* output paths.

Important computations must not depend on undocumented interactive state.

---

## 11. Contract Completion

A contract is not complete merely because its numerical tasks finished.

Follow the closing protocol specified by the contract.

If the contract contains a mandatory researcher approval checkpoint, stop there and report the results.

Status should then be:

`PENDING RESEARCHER APPROVAL`

Do not rename:

`ContractN.txt`

to:

`ContractN_complete.txt`

until explicit researcher approval has been received.

Do not begin the next contract unless instructed or explicitly authorized by the governing protocol.

---

## 12. Completion After Approval

Once explicit researcher approval is received:

1. perform all formal closure tasks required by the contract;
2. generate the final closure/execution report;
3. verify required deliverables;
4. update `STATUS.md`;
5. rename the contract specification to `_complete` if required by project convention;
6. verify that the rename succeeded;
7. leave the workspace in a clean reproducible state.

---

## 13. Minimal Researcher Command

Once this protocol exists, the researcher should be able to invoke work simply by writing:

`Contract 1`

That command means:

> Locate `contracts/Contract1.txt`, read the governing instructions, inspect or create `contracts/Contract1/`, resume any valid existing work, and execute all remaining Contract 1 tasks autonomously until the contract-defined researcher checkpoint or completion condition is reached.

The same rule applies to every contract number.

---

## 14. Core Operating Rule

The permanent project convention is:

$$
\boxed{
\text{Contract N}
\Rightarrow
\text{discover}
\rightarrow
\text{read}
\rightarrow
\text{resume/create isolated workspace}
\rightarrow
\text{execute}
\rightarrow
\text{validate}
\rightarrow
\text{report}
\rightarrow
\text{checkpoint}
}
$$

The researcher should never need to restate this protocol for each contract.

