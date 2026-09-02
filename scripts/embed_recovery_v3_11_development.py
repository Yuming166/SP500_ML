#!/usr/bin/env python3
"""Encode Recovery V3.11 development provenance paths with a frozen embedder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
from transformers import AutoModel, AutoTokenizer

TASK = "Given a factual claim, retrieve passages that verify or refute the claim"
RELATION_TASK = (
    "Represent the relation between a factual claim and evidence for classifying "
    "whether the evidence supports or refutes the claim"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return hidden[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), sequence_lengths]


def encode(
    texts: list[str],
    *,
    tokenizer: object,
    model: object,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            output = model(**batch)
            pooled = last_token_pool(output.last_hidden_state, batch["attention_mask"])
            pooled = functional.normalize(pooled.float(), p=2, dim=1)
        batches.append(pooled.cpu().numpy())
        print(f"encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(batches, axis=0)


def packet_text(packet: dict[str, object]) -> str:
    root = str(packet["root"]).replace("_", " ")
    evidence = " ".join(str(row["text"]) for row in packet["evidence"])
    return f"Wikipedia page: {root}\nEvidence: {evidence}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("results/recovery_v3_7_1/selection_manifest.json"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/storage/lianjh/modelzoos/Qwen/Qwen3-Embedding-0.6B"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/recovery_v3_11_development/provenance_scores.npz"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help=(
            "Omit labels, annotation roles, and reusable query/document vectors from "
            "the output consumed by formal route selection."
        ),
    )
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    examples = list(selection["examples"])
    if any(len(row["candidates"]) != 2 for row in examples):
        raise ValueError("every example must contain exactly two candidate paths")

    queries = [f"Instruct: {TASK}\nQuery: {row['claim']}" for row in examples]
    documents = [packet_text(candidate) for row in examples for candidate in row["candidates"]]
    relation_inputs = [
        f"Instruct: {RELATION_TASK}\nQuery: Claim: {row['claim']}\n{packet_text(candidate)}"
        for row in examples
        for candidate in row["candidates"]
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    device = torch.device(args.device)
    model = AutoModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    query_vectors = encode(
        queries,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    document_vectors = encode(
        documents,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ).reshape(len(examples), 2, -1)
    relation_vectors = encode(
        relation_inputs,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ).reshape(len(examples), 2, -1)
    scores = np.einsum("nd,nkd->nk", query_vectors, document_vectors)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "example_ids": np.asarray([row["example_id"] for row in examples]),
        "splits": np.asarray([row["split"] for row in examples]),
        "scores": scores.astype(np.float32),
        "relation_vectors": relation_vectors.astype(np.float16),
    }
    if not args.inference_only:
        arrays.update(
            {
                "labels": np.asarray([row["label"] for row in examples]),
                "annotated_indices": np.asarray(
                    [
                        next(
                            index
                            for index, candidate in enumerate(row["candidates"])
                            if candidate["annotation_role"] == "held_out_annotated_root"
                        )
                        for row in examples
                    ],
                    dtype=np.int8,
                ),
                "query_vectors": query_vectors.astype(np.float16),
                "document_vectors": document_vectors.astype(np.float16),
            }
        )
    np.savez_compressed(args.output, **arrays)
    metadata = {
        "status": (
            "formal_inference_features_without_outcome_fields"
            if args.inference_only
            else "development_only_not_a_frozen_formal_result"
        ),
        "selection": str(args.selection),
        "selection_sha256": sha256_path(args.selection),
        "model": str(args.model),
        "model_config_sha256": sha256_path(args.model / "config.json"),
        "task": TASK,
        "relation_task": RELATION_TASK,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "device": str(device),
        "examples": len(examples),
        "inference_only": args.inference_only,
        "output_fields": sorted(arrays),
        "output": str(args.output),
        "output_sha256": sha256_path(args.output),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
