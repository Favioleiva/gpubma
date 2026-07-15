# GPUBMA Phase 1 benchmark report

- Generated (UTC): 2026-07-15T11:13:53.361983+00:00
- GPU: NVIDIA GeForce RTX 3060 (CC 8.6, 28 SMs)
- Seed: 20260715
- Phase 1 feasibility benchmark; NOT the production CUDA enumerator.

## Ladder (Measured / Not evaluated)

| p | models | status | CPU median s | CPU models/s | GPU cold s | GPU warm median s | GPU models/s | GPU==CPU max diff |
|---|--------|--------|--------------|--------------|------------|-------------------|--------------|-------------------|
| 8 | 256 | Measured | 0.0076 | 33,656 | 0.2089 | 0.0059 | 43,163 | 4.55e-13 |
| 10 | 1,024 | Measured | 0.0302 | 33,889 | 0.0076 | 0.0073 | 139,589 | 4.55e-13 |
| 12 | 4,096 | Measured | 0.1214 | 33,742 | 0.0098 | 0.0091 | 450,041 | 4.55e-13 |
| 15 | 32,768 | Measured | 0.9869 | 33,204 | 0.0214 | 0.0189 | 1,738,199 | 4.55e-13 |
| 18 | 262,144 | Not evaluated | — | — | — | — | — | — |
| 20 | 1,048,576 | Not evaluated | — | — | — | — | — | — |
| 24 | 16,777,216 | Not evaluated | — | — | — | — | — | — |
| 26 | 67,108,864 | Not evaluated | — | — | — | — | — | — |
| 28 | 268,435,456 | Not evaluated | — | — | — | — | — | — |
| 30 | 1,073,741,824 | Not evaluated | — | — | — | — | — | — |

## Projections (clearly labelled, LOW confidence)

- **Projected** cpu p=24 (16,777,216 models): ~505.3 s — source: measured cpu run at p=15 on this machine; assumption: constant models/second (linear scaling in model count); confidence: LOW. May fail because: per-model cost grows with model size k (larger Cholesky factors dominate at larger p); memory pressure and batching behaviour differ at larger p; the Phase 1 scorers are not the production enumerator; reduction/normalization costs are excluded from the scaling model.
- **Projected** cpu p=26 (67,108,864 models): ~2,021.1 s — source: measured cpu run at p=15 on this machine; assumption: constant models/second (linear scaling in model count); confidence: LOW. May fail because: per-model cost grows with model size k (larger Cholesky factors dominate at larger p); memory pressure and batching behaviour differ at larger p; the Phase 1 scorers are not the production enumerator; reduction/normalization costs are excluded from the scaling model.
- **Projected** cpu p=28 (268,435,456 models): ~8,084.5 s — source: measured cpu run at p=15 on this machine; assumption: constant models/second (linear scaling in model count); confidence: LOW. May fail because: per-model cost grows with model size k (larger Cholesky factors dominate at larger p); memory pressure and batching behaviour differ at larger p; the Phase 1 scorers are not the production enumerator; reduction/normalization costs are excluded from the scaling model.
- **Projected** cpu p=30 (1,073,741,824 models): ~32,338.2 s — source: measured cpu run at p=15 on this machine; assumption: constant models/second (linear scaling in model count); confidence: LOW. May fail because: per-model cost grows with model size k (larger Cholesky factors dominate at larger p); memory pressure and batching behaviour differ at larger p; the Phase 1 scorers are not the production enumerator; reduction/normalization costs are excluded from the scaling model.
- **Projected** gpu p=24 (16,777,216 models): ~9.7 s — source: measured gpu run at p=15 on this machine; assumption: constant models/second (linear scaling in model count); confidence: LOW. May fail because: per-model cost grows with model size k (larger Cholesky factors dominate at larger p); memory pressure and batching behaviour differ at larger p; the Phase 1 scorers are not the production enumerator; reduction/normalization costs are excluded from the scaling model.
- **Projected** gpu p=26 (67,108,864 models): ~38.6 s — source: measured gpu run at p=15 on this machine; assumption: constant models/second (linear scaling in model count); confidence: LOW. May fail because: per-model cost grows with model size k (larger Cholesky factors dominate at larger p); memory pressure and batching behaviour differ at larger p; the Phase 1 scorers are not the production enumerator; reduction/normalization costs are excluded from the scaling model.
- **Projected** gpu p=28 (268,435,456 models): ~154.4 s — source: measured gpu run at p=15 on this machine; assumption: constant models/second (linear scaling in model count); confidence: LOW. May fail because: per-model cost grows with model size k (larger Cholesky factors dominate at larger p); memory pressure and batching behaviour differ at larger p; the Phase 1 scorers are not the production enumerator; reduction/normalization costs are excluded from the scaling model.
- **Projected** gpu p=30 (1,073,741,824 models): ~617.7 s — source: measured gpu run at p=15 on this machine; assumption: constant models/second (linear scaling in model count); confidence: LOW. May fail because: per-model cost grows with model size k (larger Cholesky factors dominate at larger p); memory pressure and batching behaviour differ at larger p; the Phase 1 scorers are not the production enumerator; reduction/normalization costs are excluded from the scaling model.

All numbers above are labelled Measured, Projected, or Not evaluated (task rule 7).
