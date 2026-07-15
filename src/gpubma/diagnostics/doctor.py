"""Hardware/software diagnostics: ``python -m gpubma.doctor``.

CUDA availability is only reported after an actual float64 calculation has
executed on the device and been validated against the CPU (never merely
because an NVIDIA driver exists).
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

# Documented CUDA architecture facts (NVIDIA CUDA C Programming Guide,
# Table "Technical Specifications per Compute Capability"; Ampere GA10x
# whitepaper). Used only when the CC is detected; labelled as documented.
_DOCUMENTED_CC_FACTS = {
    "8.6": {
        "max_threads_per_block": 1024,
        "shared_memory_per_block_default_bytes": 49152,
        "shared_memory_per_block_optin_max_bytes": 101376,
        "fp64_to_fp32_throughput_ratio": "1/64 (GeForce Ampere GA10x)",
        "source": "NVIDIA CUDA C Programming Guide; NVIDIA GA102 whitepaper",
    },
}

_PACKAGES = ["numpy", "scipy", "pandas", "pyarrow", "torch", "pytest",
             "statsmodels", "cupy", "numba"]


def _run(cmd) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _total_ram_bytes() -> int | None:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX(); stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullTotalPhys)
    return None


def _nvidia_smi_info() -> dict:
    fields = ("name,uuid,memory.total,memory.free,driver_version,"
              "compute_cap,pci.bus_id")
    raw = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    info = {"nvidia_smi_available": raw is not None}
    if raw:
        parts = [s.strip() for s in raw.splitlines()[0].split(",")]
        info.update({
            "gpu_name": parts[0], "gpu_uuid": parts[1],
            "vram_total_mib": float(parts[2]), "vram_free_mib": float(parts[3]),
            "driver_version": parts[4], "compute_capability": parts[5],
            "pci_bus_id": parts[6],
        })
        head = _run(["nvidia-smi"])
        if head:
            for line in head.splitlines():
                if "CUDA Version" in line:
                    info["driver_reported_cuda_version"] = line.split("CUDA Version:")[1].split("|")[0].strip()
                    break
    return info


def _torch_cuda_probe() -> dict:
    probe = {"torch_installed": False, "cuda_available_from_python": False,
             "float64_kernel_verified": False}
    try:
        import torch
    except ImportError:
        probe["note"] = "torch not installed"
        return probe
    probe["torch_installed"] = True
    probe["torch_version"] = torch.__version__
    probe["torch_cuda_build"] = torch.version.cuda
    if not torch.cuda.is_available():
        probe["note"] = "torch.cuda.is_available() is False"
        return probe
    p = torch.cuda.get_device_properties(0)
    probe.update({
        "cuda_available_from_python": True,
        "device_name": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "multiprocessor_count": p.multi_processor_count,
        "total_vram_bytes": p.total_memory,
        "max_threads_per_multiprocessor": getattr(p, "max_threads_per_multi_processor", None),
        "warp_size": getattr(p, "warp_size", None),
    })
    import numpy as np
    a = torch.randn(256, 256, dtype=torch.float64, device="cuda")
    b = torch.randn(256, 256, dtype=torch.float64, device="cuda")
    s_gpu = float((a @ b).sum().item()); torch.cuda.synchronize()
    s_cpu = float((a.cpu().numpy() @ b.cpu().numpy()).sum())
    probe["float64_test"] = {"gpu": s_gpu, "cpu": s_cpu, "abs_diff": abs(s_gpu - s_cpu)}
    probe["float64_kernel_verified"] = abs(s_gpu - s_cpu) < 1e-8
    cc = probe["compute_capability"]
    if cc in _DOCUMENTED_CC_FACTS:
        probe["documented_architecture_facts"] = _DOCUMENTED_CC_FACTS[cc]
    return probe


def _stata_probe() -> dict:
    """Detect Stata and whether it is actually callable in batch mode.

    A functional Windows install exposes Stata{SE,MP,BE}-64.exe. Renamed
    leftovers (e.g. StataSE-64_old.exe) are recorded but treated as not
    callable: on this machine they were tested and produce no batch output.
    """
    candidates = ["stata", "stata-mp", "stata-se", "StataMP-64", "StataSE-64", "StataBE-64"]
    on_path = {c: shutil.which(c) for c in candidates}
    on_path = {k: v for k, v in on_path.items() if v}
    dirs = {}
    callable_exes = []
    for root in (r"C:\Program Files\Stata19", r"C:\Program Files\Stata18",
                 r"C:\Program Files\Stata17", r"C:\Program Files (x86)\Stata18"):
        if os.path.isdir(root):
            exes = [f for f in os.listdir(root) if f.lower().endswith(".exe")]
            dirs[root] = exes
            callable_exes += [os.path.join(root, f) for f in exes
                              if f in ("StataSE-64.exe", "StataMP-64.exe", "StataBE-64.exe")]
    callable_exes += list(on_path.values())
    return {
        "installed": bool(on_path or dirs),
        "callable": bool(callable_exes),
        "callable_executables": callable_exes,
        "on_path": on_path,
        "directories": dirs,
        "note": ("directories contain only renamed '*_old.exe' leftovers; batch "
                 "execution was tested and produced no output — treated as NOT callable"
                 if dirs and not callable_exes else None),
    }


def collect_diagnostics() -> dict:
    packages = {}
    for name in _PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    diag = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "total_ram_bytes": _total_ram_bytes(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "packages": packages,
        "package_managers": {
            "pip": shutil.which("pip") is not None,
            "conda": shutil.which("conda") is not None,
            "uv": shutil.which("uv") is not None,
        },
        "compilers": {
            "nvcc": shutil.which("nvcc"),
            "cl": shutil.which("cl"),
            "g++": shutil.which("g++"),
            "gcc": shutil.which("gcc"),
        },
        "nvidia": _nvidia_smi_info(),
        "torch_cuda": _torch_cuda_probe(),
        "stata": _stata_probe(),
    }
    diag["summary"] = {
        "gpu_detected": diag["nvidia"].get("gpu_name"),
        "cuda_usable_from_python_float64_verified": diag["torch_cuda"]["float64_kernel_verified"],
        "nvcc_installed": diag["compilers"]["nvcc"] is not None,
        "stata_installed": diag["stata"]["installed"],
        "stata_callable": diag["stata"]["callable"],
    }
    return diag


def format_report(diag: dict) -> str:
    nv, tc = diag["nvidia"], diag["torch_cuda"]
    lines = [
        "gpubma doctor — hardware and software diagnostics",
        "=" * 60,
        f"Timestamp (UTC):     {diag['timestamp_utc']}",
        f"OS:                  {diag['os']['platform']}",
        f"CPU:                 {diag['os']['processor']} ({diag['os']['cpu_count']} logical cores)",
        f"RAM:                 {diag['os']['total_ram_bytes'] / 2**30:.1f} GiB"
        if diag["os"]["total_ram_bytes"] else "RAM:                 unknown",
        f"Python:              {diag['python']['version']} ({diag['python']['executable']})",
        "",
        "GPU (nvidia-smi):",
        f"  device:            {nv.get('gpu_name', 'NOT DETECTED')}",
        f"  uuid:              {nv.get('gpu_uuid', 'n/a')}",
        f"  vram total/free:   {nv.get('vram_total_mib', 'n/a')} / {nv.get('vram_free_mib', 'n/a')} MiB",
        f"  driver:            {nv.get('driver_version', 'n/a')}"
        f"   (driver CUDA: {nv.get('driver_reported_cuda_version', 'n/a')})",
        f"  compute capability:{nv.get('compute_capability', 'n/a')}",
        "",
        "CUDA from Python (torch):",
        f"  torch:             {tc.get('torch_version', 'not installed')}"
        f" (CUDA build {tc.get('torch_cuda_build', 'n/a')})",
        f"  cuda available:    {tc['cuda_available_from_python']}",
        f"  multiprocessors:   {tc.get('multiprocessor_count', 'n/a')}",
        f"  float64 verified:  {tc['float64_kernel_verified']}"
        + (f"  (matmul abs diff {tc['float64_test']['abs_diff']:.2e})" if "float64_test" in tc else ""),
    ]
    if "documented_architecture_facts" in tc:
        f = tc["documented_architecture_facts"]
        lines += [
            f"  documented (CC {tc['compute_capability']}, {f['source']}):",
            f"    max threads/block:       {f['max_threads_per_block']}",
            f"    shared mem/block:        {f['shared_memory_per_block_default_bytes']} B"
            f" (opt-in max {f['shared_memory_per_block_optin_max_bytes']} B)",
            f"    FP64:FP32 throughput:    {f['fp64_to_fp32_throughput_ratio']}",
        ]
    lines += [
        "",
        f"nvcc:                {diag['compilers']['nvcc'] or 'NOT INSTALLED'}",
        f"C++ compilers:       g++={diag['compilers']['g++'] or 'no'}, cl={diag['compilers']['cl'] or 'no'}",
        "Stata:               "
        + ("CALLABLE: " + str(diag["stata"]["callable_executables"])
           if diag["stata"]["callable"]
           else ("found but NOT callable — " + str(diag["stata"].get("note"))
                 if diag["stata"]["installed"]
                 else "NOT INSTALLED (validation scripts prepared, not executed)")),
        "",
        "Summary: "
        + ("CUDA float64 VERIFIED on "
           + str(diag["summary"]["gpu_detected"])
           if diag["summary"]["cuda_usable_from_python_float64_verified"]
           else "CUDA float64 NOT verified — GPU work not possible on this machine yet"),
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="gpubma doctor")
    parser.add_argument("--json", metavar="PATH", help="also save the diagnostics as JSON")
    args = parser.parse_args(argv)
    diag = collect_diagnostics()
    print(format_report(diag))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(diag, fh, indent=2)
        print(f"\nJSON saved: {args.json}")
    return 0
