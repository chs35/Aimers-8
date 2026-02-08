"""
Patch ModelOpt FP8 checkpoints to be loadable by vLLM (EXAONE4) without
modifying vLLM code.

Problem:
  vLLM's EXAONE4 weight loader stacks q/k/v projection parameters by replacing
  ".{q,k,v}_proj" -> ".qkv_proj". This unintentionally also matches ModelOpt's
  KV-cache scale tensors:
    - *.self_attn.k_proj.k_scale
    - *.self_attn.v_proj.v_scale
  which leads to a KeyError for "*.self_attn.qkv_proj.{k,v}_scale".

Fix (model-only):
  Rename the KV-cache scale tensors so they no longer contain ".k_proj"/".v_proj"
  and let vLLM remap them to "*.self_attn.attn.{k,v}_scale" during loading.

  - *.self_attn.k_proj.k_scale -> *.self_attn.k_scale
  - *.self_attn.v_proj.v_scale -> *.self_attn.v_scale

This script rewrites the safetensors file (format is immutable).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def _resolve_safetensors_path(model_dir: Path) -> Path:
    if model_dir.is_file():
        return model_dir

    candidate = model_dir / "model.safetensors"
    if candidate.exists():
        return candidate

    # Fall back to a single *.safetensors file if present.
    safes = sorted(model_dir.glob("*.safetensors"))
    if len(safes) == 1:
        return safes[0]
    if not safes:
        raise FileNotFoundError(f"No .safetensors file found under: {model_dir}")
    raise FileExistsError(
        f"Multiple .safetensors files found under {model_dir}. "
        "Pass the exact file path."
    )


def _patched_key(key: str) -> str:
    if key.endswith(".self_attn.k_proj.k_scale"):
        return key.replace(".self_attn.k_proj.k_scale", ".self_attn.k_scale")
    if key.endswith(".self_attn.v_proj.v_scale"):
        return key.replace(".self_attn.v_proj.v_scale", ".self_attn.v_scale")
    return key


def main() -> int:
    p = argparse.ArgumentParser(
        description="Patch ModelOpt KV-cache scale tensor keys for vLLM compatibility."
    )
    p.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="HF model directory (or a .safetensors file path).",
    )
    p.add_argument(
        "--inplace",
        action="store_true",
        help="Replace the original file (creates a .bak backup).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would change; do not write any files.",
    )
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    in_path = _resolve_safetensors_path(model_dir)
    out_path = in_path.with_suffix(".patched.safetensors")

    renamed = []
    tensors = {}

    with safe_open(str(in_path), framework="pt", device="cpu") as f:
        meta = f.metadata()
        keys = list(f.keys())
        for k in keys:
            nk = _patched_key(k)
            if nk != k:
                renamed.append((k, nk))
            if nk in tensors:
                raise RuntimeError(f"Duplicate key after patch: {nk}")
            tensors[nk] = f.get_tensor(k)

    if not renamed:
        print(f"[OK] No KV-cache scale keys to patch in: {in_path}")
        return 0

    print(f"Input:  {in_path}")
    print(f"Output: {out_path if not args.inplace else in_path}")
    print(f"Renames: {len(renamed)}")
    for old, new in renamed[:20]:
        print(f"  {old} -> {new}")
    if len(renamed) > 20:
        print("  ...")

    if args.dry_run:
        print("[DRY-RUN] No files written.")
        return 0

    save_file(tensors, str(out_path), metadata=meta)

    if args.inplace:
        bak_path = in_path.with_suffix(in_path.suffix + ".bak")
        if bak_path.exists():
            raise FileExistsError(f"Backup already exists: {bak_path}")
        in_path.rename(bak_path)
        out_path.rename(in_path)
        print(f"[OK] Patched in place. Backup: {bak_path}")
    else:
        print("[OK] Patched file written.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

