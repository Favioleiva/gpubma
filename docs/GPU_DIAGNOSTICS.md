# GPU diagnostics

## Commands

```bash
python -m gpubma.doctor                # console report
python -m gpubma.doctor --json reports/gpu_doctor.json
gpubma doctor                          # installed CLI equivalent
python -m gpubma.gpu.feasibility       # feasibility check + JSON report
```

## Policy

- **Never assume the GPU model.** The doctor queries `nvidia-smi`
  (name, UUID, VRAM total/free, driver, compute capability) and PyTorch
  device properties (SM count, VRAM), and records the OS and Python stack.
- **CUDA is reported usable only after a real float64 kernel ran** on the
  device and matched the CPU result (256×256 matmul, tolerance 1e-8). A
  present driver alone is never treated as usable CUDA.
- Architecture limits that cannot be queried at run time (max threads/block,
  shared memory/block, FP64:FP32 throughput ratio) are included only for the
  detected compute capability and labelled with their documentary source
  (NVIDIA CUDA C Programming Guide; GA102 whitepaper).

## Machine detected in Phase 1 (2026-07-15)

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12 GiB (12,288 MiB), CC 8.6, 28 SMs |
| Driver | 591.86 (driver-reported CUDA 13.1), WDDM, Windows 11 Pro |
| Python CUDA path | torch 2.5.1+cu121 (CUDA 12.1 build) |
| float64 verified | yes — matmul GPU vs CPU abs diff ≈ 1.8e-12 |
| Batched model scoring | 256..32,768 models, float64, max |log-score| diff vs CPU 4.5e-13 |
| nvcc | not installed (needed only for later custom-kernel phases) |
| FP64 caveat | GeForce Ampere executes FP64 at 1/64 of FP32 throughput (documented) — float64 works correctly but is relatively slow; this matters for production-kernel design |

Full machine-readable reports: `reports/environment_report.json`,
`reports/gpu_doctor.json`, `reports/gpu_feasibility.json`.

## Feasibility verdict

The feasibility layer (transfer sufficient statistics → batched float64
Cholesky scoring → synchronize → cold/warm timing → CPU comparison) declared
this machine **suitable** for later CUDA kernel development. Projected
billion-model performance is deliberately NOT derived from the small matmul;
only the batched model-scoring benchmark feeds (clearly labelled, LOW
confidence) projections.
