#!/usr/bin/env python3
"""GPU sanity checks before local 7B training/evaluation.

This script is intentionally diagnostic. It does not modify drivers or install
packages; if `nvidia-smi` reports an NVML mismatch, fix the host driver/runtime
first before starting P12 work.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--torch-smoke", action="store_true",
                    help="Allocate a tiny CUDA tensor and run one matrix multiply.")
    args = ap.parse_args()

    print("== nvidia-smi ==")
    smi = subprocess.run(["nvidia-smi"], text=True, capture_output=True, check=False)
    if smi.stdout:
        print(smi.stdout.strip())
    if smi.stderr:
        print(smi.stderr.strip(), file=sys.stderr)
    smi_failed = smi.returncode != 0
    if smi_failed:
        print(f"nvidia-smi failed with return code {smi.returncode}", file=sys.stderr)

    print("\n== NVIDIA kernel module ==")
    proc = subprocess.run(
        ["cat", "/proc/driver/nvidia/version"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
        print(proc.stderr.strip(), file=sys.stderr)

    print("\n== PyTorch CUDA ==")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"torch import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"torch={torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"cuda_available={cuda_available}")
    if not cuda_available:
        if smi_failed:
            print(
                "CUDA is unavailable in this process and nvidia-smi also failed; "
                "the host driver/userspace stack is not currently usable.",
                file=sys.stderr,
            )
        return 1
    print(f"device_count={torch.cuda.device_count()}")
    print(f"device_0={torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"vram_free_gb={free / 1e9:.2f}")
    print(f"vram_total_gb={total / 1e9:.2f}")

    if args.torch_smoke:
        x = torch.randn((512, 512), device="cuda")
        y = x @ x.T
        torch.cuda.synchronize()
        print(f"torch_smoke_sum={float(y.sum().detach().cpu()):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
