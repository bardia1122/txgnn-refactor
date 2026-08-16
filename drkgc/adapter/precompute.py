"""Smoke-test the GCN adapter, and optionally export frozen local embeddings.

The adapter has no loss of its own — it is trained jointly with the LLM in step 5.
So the useful things to do with it before step 5 exists are:

1. **verify it runs**: shapes line up, isolated candidates still get a vector,
   gradients reach the GCN;
2. **export frozen local embeddings** (`--export`) for inspection or for an
   ablation where the GCN is *not* trained jointly. Note this is a deviation from
   the paper, which back-propagates into the GCN during LoRA fine-tuning; frozen
   local embeddings from an untrained GCN carry little signal and are for
   diagnostics, not for a headline result.

Run::

    python -m drkgc.adapter.precompute --out-dir drkgc/data_holdout \\
        --model-dir drkgc/models/rgcn_holdout_v2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.config import DEFAULT_OUT_DIR, DRKGC_ROOT, TARGET_RELATIONS
from drkgc.adapter.data import collate, load_global_embeddings, load_samples

DEFAULT_MODEL_DIR = DRKGC_ROOT / "models" / "rgcn"


def smoke_test(
    out_dir: Path,
    model_dir: Path,
    relation: str,
    split: str,
    hidden_dim: int,
    llm_dim: int,
    batch_size: int,
    limit: int,
) -> dict:
    import torch

    from drkgc.adapter.model import GCNAdapter
    from drkgc.ranker.data import build_dataset

    data = build_dataset(out_dir)
    global_embeddings = load_global_embeddings(model_dir)
    global_dim = int(global_embeddings.shape[1])
    print(
        f"global embeddings {tuple(global_embeddings.shape)} | "
        f"{data.num_relations} relations (with inverses)"
    )

    samples = load_samples(out_dir, relation, split, data.rel2id, limit)
    print(f"{len(samples)} subgraphs from {relation}/{split}")

    adapter = GCNAdapter(
        global_dim=global_dim,
        hidden_dim=hidden_dim,
        llm_dim=llm_dim,
        num_relations=data.num_relations,
    )
    print(f"adapter: {adapter.num_parameters:,} parameters")

    batch = collate(samples[:batch_size])
    features = global_embeddings[batch["node_ids"]]

    output = adapter(
        features,
        batch["edge_index"],
        batch["edge_type"],
        batch["query_index"],
        batch["candidate_index"],
    )

    expected_query = (len(batch["query_index"]), llm_dim)
    expected_candidates = (*batch["candidate_index"].shape, llm_dim)
    assert tuple(output["query"].shape) == expected_query, (
        f"query vectors {tuple(output['query'].shape)} != {expected_query}"
    )
    assert tuple(output["candidates"].shape) == expected_candidates, (
        f"candidate vectors {tuple(output['candidates'].shape)} != {expected_candidates}"
    )
    assert torch.isfinite(output["query"]).all(), "non-finite query vectors"
    assert torch.isfinite(output["candidates"]).all(), "non-finite candidate vectors"

    # an isolated candidate (no retrieved path) must still produce a usable vector
    degree = torch.zeros(features.shape[0], dtype=torch.long)
    if batch["edge_index"].numel():
        degree.scatter_add_(
            0, batch["edge_index"].reshape(-1),
            torch.ones(batch["edge_index"].numel(), dtype=torch.long),
        )
    isolated = (degree[batch["candidate_index"].reshape(-1)] == 0).sum().item()

    # gradients must reach the GCN, or joint training in step 5 is a no-op
    output["query"].sum().backward()
    gcn_grad = adapter.gcn.input_projection.weight.grad
    assert gcn_grad is not None and torch.isfinite(gcn_grad).all(), (
        "no gradient reached the subgraph GCN"
    )

    print(
        f"\nquery vectors      {tuple(output['query'].shape)}\n"
        f"candidate vectors  {tuple(output['candidates'].shape)}\n"
        f"local embeddings   {tuple(output['local'].shape)}\n"
        f"isolated candidates in this batch: {isolated} "
        f"(each still gets a vector from its global embedding)\n"
        f"gradient norm into the GCN: {gcn_grad.norm().item():.4f}"
    )
    return {
        "num_samples": len(samples),
        "global_dim": global_dim,
        "llm_dim": llm_dim,
        "adapter_parameters": adapter.num_parameters,
        "isolated_candidates_in_batch": int(isolated),
    }


def export_local(
    out_dir: Path,
    model_dir: Path,
    relations: Sequence[str],
    splits: Sequence[str],
    hidden_dim: int,
    llm_dim: int,
    batch_size: int,
    limit: Optional[int],
) -> None:
    """Frozen local embeddings, for diagnostics/ablation only (see module docstring)."""
    import torch

    from drkgc.adapter.model import GCNAdapter
    from drkgc.ranker.data import build_dataset

    data = build_dataset(out_dir)
    global_embeddings = load_global_embeddings(model_dir)
    adapter = GCNAdapter(
        global_dim=int(global_embeddings.shape[1]),
        hidden_dim=hidden_dim,
        llm_dim=llm_dim,
        num_relations=data.num_relations,
    ).eval()

    target = Path(out_dir) / "adapter"
    target.mkdir(parents=True, exist_ok=True)

    for relation in relations:
        for split in splits:
            samples = load_samples(out_dir, relation, split, data.rel2id, limit)
            if not samples:
                continue
            chunks = []
            with torch.no_grad():
                for start in range(0, len(samples), batch_size):
                    batch = collate(samples[start : start + batch_size])
                    features = global_embeddings[batch["node_ids"]]
                    local = adapter.gcn(
                        features, batch["edge_index"], batch["edge_type"]
                    )
                    flat = batch["candidate_index"].reshape(-1)
                    chunks.append(
                        {
                            "query": local[batch["query_index"]],
                            "candidates": local[flat].view(
                                *batch["candidate_index"].shape, -1
                            ),
                            "meta": batch["meta"],
                        }
                    )
            slug = relation.replace(" ", "_").replace("/", "_")
            path = target / f"{slug}_{split}_local.pt"
            torch.save(
                {
                    "query": torch.cat([c["query"] for c in chunks]),
                    "candidates": torch.cat([c["candidates"] for c in chunks]),
                    "meta": [m for c in chunks for m in c["meta"]],
                    "note": "UNTRAINED GCN - diagnostics only, see precompute.py",
                },
                path,
            )
            print(f"  {relation}/{split}: {len(samples)} queries -> {path}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--relation", default=TARGET_RELATIONS[0])
    parser.add_argument("--split", default="valid")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="GCN width (paper searched {128, 256})")
    parser.add_argument("--llm-dim", type=int, default=4096,
                        help="LLM hidden size (Llama-3-8B is 4096)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--export", action="store_true",
                        help="also write frozen local embeddings (diagnostics only)")
    parser.add_argument("--relations", nargs="+", default=list(TARGET_RELATIONS))
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    args = parser.parse_args(list(argv) if argv is not None else None)

    smoke_test(
        Path(args.out_dir), Path(args.model_dir), args.relation, args.split,
        args.hidden_dim, args.llm_dim, args.batch_size, args.limit,
    )
    if args.export:
        print("\nExporting frozen local embeddings ...")
        export_local(
            Path(args.out_dir), Path(args.model_dir), args.relations, args.splits,
            args.hidden_dim, args.llm_dim, args.batch_size, args.limit,
        )


if __name__ == "__main__":
    main()
