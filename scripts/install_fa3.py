#!/usr/bin/env python3
"""
Persistently install Flash Attention 3 wheel matching local torch/CUDA.

Usage:
    uv run python scripts/install_fa3.py
    uv run python scripts/install_fa3.py --dry-run
    uv run python scripts/install_fa3.py --index-key cu128_torch2100
"""

from __future__ import annotations

import argparse
import platform
import re
import subprocess
import urllib.parse
import urllib.request

import torch

WHEEL_INDEX_ROOT = "https://windreamer.github.io/flash-attention3-wheels"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Install compatible Flash Attention 3 wheel from windreamer index"
    )
    ap.add_argument(
        "--index-key",
        type=str,
        default=None,
        help="Override auto detection, e.g. cu128_torch2100",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print detected index and install command without executing",
    )
    return ap.parse_args()


def _parse_semver_triplet(version: str, label: str) -> tuple[int, int, int]:
    core = version.split("+", 1)[0]
    parts = core.split(".")
    assert len(parts) >= 2, f"{label} must have at least 2 segments, got: {version}"
    major = int(parts[0])
    minor = int(parts[1])
    patch = int(parts[2]) if len(parts) >= 3 else 0
    return major, minor, patch


def _detect_index_key() -> str:
    assert torch.version.cuda is not None, (
        "torch.version.cuda is None. CUDA-enabled torch is required for FA3."
    )
    cuda_major, cuda_minor, _ = _parse_semver_triplet(
        torch.version.cuda, "CUDA version"
    )
    torch_major, torch_minor, torch_patch = _parse_semver_triplet(
        torch.__version__, "torch version"
    )
    cuda_token = f"cu{cuda_major}{cuda_minor}"
    torch_token = f"torch{torch_major}{torch_minor}{torch_patch}"
    return f"{cuda_token}_{torch_token}"


def _fetch_available_index_keys() -> list[str]:
    with urllib.request.urlopen(WHEEL_INDEX_ROOT) as resp:
        html = resp.read().decode("utf-8")
    keys = sorted(set(re.findall(r"flash-attention3-wheels/(cu\d+_torch\d+)", html)))
    assert keys, f"No wheel indexes parsed from {WHEEL_INDEX_ROOT}"
    return keys


def _arch_token(machine: str) -> str:
    if machine in {"x86_64", "amd64"}:
        return "x86_64"
    if machine in {"aarch64", "arm64"}:
        return "aarch64"
    assert False, f"Unsupported machine architecture: {machine}"


def _fetch_wheel_url(index_key: str, machine: str) -> str:
    index_url = f"{WHEEL_INDEX_ROOT}/{index_key}"
    with urllib.request.urlopen(index_url) as resp:
        html = resp.read().decode("utf-8")

    hrefs = re.findall(r'href="([^"]+\.whl)"', html)
    assert hrefs, f"No wheel links found at {index_url}"

    arch = _arch_token(machine)
    linux_hrefs = [h for h in hrefs if "linux" in h and arch in h]
    assert linux_hrefs, (
        f"No Linux wheel found for arch={arch} at {index_url}. "
        f"Available wheels: {', '.join(hrefs)}"
    )

    abi3_hrefs = [h for h in linux_hrefs if "abi3" in h]
    selected = abi3_hrefs[0] if abi3_hrefs else linux_hrefs[0]
    return urllib.parse.urljoin(f"{index_url}/", selected)


def _build_mismatch_message(index_key: str, available: list[str]) -> str:
    cuda_part = index_key.split("_", 1)[0]
    same_cuda = [k for k in available if k.startswith(f"{cuda_part}_")]
    base = [f"Detected index key '{index_key}' is not available."]
    if same_cuda:
        base.append(
            f"Available keys with same CUDA ({cuda_part}): {', '.join(same_cuda)}"
        )
    else:
        base.append("No index found for this CUDA line.")
    base.append(f"Browse all indexes: {WHEEL_INDEX_ROOT}")
    base.append(
        'Fallback options: set attn_implementation to "flash_attention_2" or "sdpa".'
    )
    return "\n".join(base)


def main() -> None:
    args = parse_args()
    machine = platform.machine().lower()
    _arch_token(machine)

    index_key = args.index_key or _detect_index_key()
    available_keys = _fetch_available_index_keys()
    assert index_key in available_keys, _build_mismatch_message(
        index_key, available_keys
    )
    index_url = f"{WHEEL_INDEX_ROOT}/{index_key}"
    wheel_url = _fetch_wheel_url(index_key, machine)

    print(f"Detected platform: {platform.system()} {machine}")
    print(f"Using FA3 index: {index_url}")
    print(f"Resolved wheel: {wheel_url}")

    add_cmd = [
        "uv",
        "add",
        "--group",
        "flash-attn",
        f"flash-attn-3 @ {wheel_url} ; sys_platform == 'linux'",
    ]
    sync_cmd = ["uv", "sync", "--group", "flash-attn"]
    print("Persist command:")
    print(" ".join(add_cmd))
    print("Sync command:")
    print(" ".join(sync_cmd))

    if args.dry_run:
        return

    subprocess.run(add_cmd, check=True)
    subprocess.run(sync_cmd, check=True)
    print('Done. FA3 is now tracked by uv lock and synced with group "flash-attn".')
    print("Use `uv run --group flash-attn ...` to keep this group active during runs.")


if __name__ == "__main__":
    main()
