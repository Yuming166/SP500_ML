#!/usr/bin/env python3
"""Dequantize Ling-3.0-tiny-int4 (compressed-tensors int4) → bf16 safetensors.

Ling's int4 weights are stored as triplets:
  weight_packed: int32 packed (4-bit values)
  weight_scale:  fp32 per-row scale
  weight_shape:  int64 original shape

This script reads the original safetensors, dequantizes each weight triplet
back to bf16 (unpacking 4-bit values and applying scale), and writes a new
safetensors file. The resulting model is ~22 GB (vs 5.5 GB int4) but loads
on stock transformers without compressed-tensors runtime support.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC_DIR = Path("/storage/gaoym/sp500-forecastability-lab/scripts/ling_server/ling_local")
DST_DIR = Path("/storage/gaoym/sp500-forecastability-lab/scripts/ling_server/ling_bf16")
DST_DIR.mkdir(parents=True, exist_ok=True)


def dequantize_int4(packed: torch.Tensor, scale: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """Unpack int4 values stored in int32 (8 values per int32) and apply scale.

    Ling stores int4 weights packed into int32: each int32 holds 8 nibbles.
    The packed tensor is reshaped using `shape` (the original Linear weight shape).
    """
    # packed: (N, 1) int32 (last dim is just a holder)
    flat = packed.view(torch.int32).flatten()  # int32 tensor
    # Convert to uint8 view
    u8 = flat.view(torch.uint8)  # (4*N,) bytes
    # Each int32 = 4 bytes; each byte holds 2 nibbles (high and low)
    low = (u8 & 0x0F).to(torch.uint8)
    high = ((u8 >> 4) & 0x0F).to(torch.uint8)
    # interleave: [low0, high0, low1, high1, ...]
    unpacked = torch.stack([low, high], dim=-1).flatten()  # (8*N,) uint8
    # reshape to original shape
    unpacked = unpacked[: int(shape.prod().item())]
    out = unpacked.view(shape.tolist()).to(torch.bfloat16)
    # apply scale (per-row typically)
    # scale shape might be (out_features, 1) or (out_features,)
    if scale.numel() == shape[0]:
        scale = scale.view(-1, *([1] * (out.dim() - 1)))
    out = out * scale.to(torch.bfloat16)
    return out


def main():
    src_files = sorted(SRC_DIR.glob("model-*.safetensors"))
    print(f"Found {len(src_files)} safetensors shards in {SRC_DIR}")

    # Read index
    index = json.loads((SRC_DIR / "model.safetensors.index.json").read_text())

    # Process each shard
    for shard_path in src_files:
        dst_path = DST_DIR / shard_path.name
        print(f"\nProcessing {shard_path.name}...")

        new_tensors: dict[str, torch.Tensor] = {}
        with safe_open(shard_path, framework="pt") as f:
            keys = list(f.keys())
            for key in keys:
                tensor = f.get_tensor(key)
                if key.endswith(".weight_packed"):
                    # Look for siblings
                    base = key[:-len(".weight_packed")]
                    scale_key = base + ".weight_scale"
                    shape_key = base + ".weight_shape"
                    if scale_key in keys and shape_key in keys:
                        scale = f.get_tensor(scale_key)
                        shape_t = f.get_tensor(shape_key)
                        try:
                            deq = dequantize_int4(tensor, scale, shape_t)
                            new_tensors[base + ".weight"] = deq
                            print(f"  dequantized {base}.weight: {deq.shape} {deq.dtype}")
                        except Exception as e:
                            print(f"  WARN: failed to dequantize {base}: {e}")
                            new_tensors[key] = tensor
                            new_tensors[scale_key] = scale
                            new_tensors[shape_key] = shape_t
                    else:
                        # fallback: keep as-is
                        new_tensors[key] = tensor
                elif key.endswith((".weight_scale", ".weight_shape")):
                    # skip; included in dequantized version
                    continue
                else:
                    new_tensors[key] = tensor

        # Save new shard
        save_file(new_tensors, dst_path, metadata={"format": "pt"})
        print(f"  wrote {dst_path.name} with {len(new_tensors)} tensors")

    # Copy non-shard files
    for f in SRC_DIR.iterdir():
        if f.is_file() and f.suffix not in (".safetensors", ".bin", ".pt"):
            dst_f = DST_DIR / f.name
            if not dst_f.exists():
                dst_f.write_bytes(f.read_bytes())
                print(f"copied {f.name}")

    # Rewrite index.json
    new_index = {"metadata": index.get("metadata", {}), "weight_map": {}}
    total_size = 0
    for k, v in index["weight_map"].items():
        if k.endswith(".weight_packed"):
            base = k[:-len(".weight_packed")]
            new_index["weight_map"][base + ".weight"] = v
        elif k.endswith((".weight_scale", ".weight_shape")):
            continue
        else:
            new_index["weight_map"][k] = v
    (DST_DIR / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2))
    print(f"\nWrote new index.json ({len(new_index['weight_map'])} tensors)")
    print(f"Output: {DST_DIR}")


if __name__ == "__main__":
    main()