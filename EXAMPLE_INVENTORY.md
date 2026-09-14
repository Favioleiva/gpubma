# Example inventory

| Notebook | Purpose | Default work |
|---|---|---|
| [Exact_BMA_Canonical_p30_Figures](examples/Exact_BMA_Canonical_p30_Figures.ipynb) | Complete exact p30 reference | Load compact frozen data; nine PNGs and exact tables; zero model evaluations |
| [BFG_Canonical_p30_Example](examples/BFG_Canonical_p30_Example.ipynb) | Budgeted p30 discovery | Hard global BUDGET=5,000 |
| [BFG_Canonical_p30_Shell_Recovery](examples/BFG_Canonical_p30_Shell_Recovery.ipynb) | Post-BFG reticular recovery | Budgeted BFG discovery + representative random shell sampling (default cap 100k) |
| [BFG_User_Dataset_Example](examples/BFG_User_Dataset_Example.ipynb) | User data | Configure DATA_PATH, TARGET, EXPLANATORY_VARIABLES, BUDGET |
| [BFG_Public_Stata_Example](examples/BFG_Public_Stata_Example.ipynb) | Grunfeld Stata compatibility / econometric parity | Four possible models; not a large-space demonstration |

Open a notebook in Jupyter or Colab and Run All. The exact notebook checks for a local public checkout first; standalone Colab downloads the committed, checksum-verified reference inputs and rendering helpers after this candidate has been published to the public repository. Install dependencies requires network access when not already available. No Drive mount is needed.

The three preserved BFG notebooks contain the transparent SHA-256-verified canonical release-candidate wheel. Their existing bootstrap and algorithm are unchanged. The exact notebook installs the checkout or public repository package, then loads frozen data. See [exact artifact provenance](docs/exact_p30_reference.md).

Every canonical figure follows Markdown/LaTeX title → title-free PNG → Markdown/LaTeX Notes. Exact exports go to `outputs/exact_p30/`. BFG exports use their separate output directories. No PDFs are needed for normal use.

Discovered-set BFG weights are not exact global PMP/PIP. Exact global figures and full-universe coefficient mixtures appear only in the exact notebook. Candidate sizes exclude intercept and always-in controls.

The user-data notebook intentionally requires real configuration; it has no silent toy fallback. Its release test supplies the canonical dataset explicitly. Live Colab UI execution is a separate publication check; local fresh-kernel results are recorded in [TEST_REPORT.md](TEST_REPORT.md).
