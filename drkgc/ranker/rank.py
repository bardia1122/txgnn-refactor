"""Export the two artifacts later DrKGC steps consume from the trained R-GCN.

    python -m drkgc.ranker.rank --model-dir drkgc/models/rgcn --out-dir drkgc/data

1. **Global structural embeddings** `E_global` for every entity — the GCN
   adapter initialises its node features from these.
2. **Top-k candidate drugs** per query, which become the constrained answer set
   in the LLM prompt (the paper uses k = 20).

Candidate generation follows DrKGC: for a query `(?, r, t)` every entity of the
right type is scored and the top k are kept, with known-true answers removed —
but never the gold, so the candidate set stays a fair evaluation target.
`--filter all` (default) removes every known triple, the standard filtered
protocol; `--filter train` removes only training triples, which is harsher and
matches a deployment setting. The gold's rank in the full ranking is recorded as
`gold_rank`, and `gold_in_candidates` gives the ceiling the LLM can reach.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.config import DEFAULT_OUT_DIR, DRKGC_ROOT, TARGET_RELATIONS
from drkgc.ranker.data import RankerData, build_dataset

DEFAULT_MODEL_DIR = DRKGC_ROOT / "models" / "rgcn"
DEFAULT_K = 20


def load_model(model_dir: Path, num_entities: int, num_relations: int, device):
    import torch

    from drkgc.ranker.model import RGCNRanker

    checkpoint = torch.load(model_dir / "model.pt", map_location=device, weights_only=False)
    config = checkpoint["config"]
    if config["num_entities"] != num_entities or config["num_relations"] != num_relations:
        raise ValueError(
            "the checkpoint was trained on a different dataset "
            f"({config['num_entities']} entities / {config['num_relations']} relations "
            f"vs {num_entities} / {num_relations}). Retrain, or point --out-dir at the "
            "artifacts the model was trained on."
        )
    model = RGCNRanker(
        num_entities,
        num_relations,
        config["dim"],
        config["num_layers"],
        config.get("num_bases"),
        config.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def export_embeddings(model, data: RankerData, edge_index, edge_type, model_dir: Path):
    import torch

    with torch.no_grad():
        z = model.encode(edge_index, edge_type)
    path = model_dir / "global_embeddings.pt"
    torch.save(
        {
            "embeddings": z.cpu(),
            "dim": int(z.shape[1]),
            "num_entities": int(z.shape[0]),
            "entity_table": "entities_global.csv",
            "note": "row i corresponds to global_id i in entities_global.csv",
        },
        path,
    )
    print(f"global embeddings {tuple(z.shape)} -> {path}")
    return z


def candidates_for_split(
    model,
    data: RankerData,
    z,
    relation: str,
    split: str,
    device,
    k: int = DEFAULT_K,
    direction: str = "head",
    batch_size: int = 128,
    filter_mode: str = "all",
) -> pd.DataFrame:
    """Top-k candidates for every query triple of one relation/split.

    `filter_mode` controls which known-true answers are removed from the ranking
    before the top-k is taken (the gold is never removed):

    ``all``
        The standard *filtered* protocol (Bordes et al. 2013) — every known
        triple in train+valid+test. Use this for anything you compare against
        published numbers, and for building the LLM's candidate sets: other
        held-out true drugs for the same disease are correct answers, so letting
        them crowd out the gold understates the ceiling for no good reason.
    ``train``
        Only training triples are removed. Harsher, and closer to a deployment
        setting where the held-out answers genuinely are not known yet.
    """
    import torch

    triples = data.eval_triples.get((relation, split))
    if triples is None or len(triples) == 0:
        return pd.DataFrame()

    table = data.entities.table
    name_of = dict(zip(table.global_id, table.node_name))
    type_of = dict(zip(table.global_id, table.node_type))
    idx_of = dict(zip(table.global_id, table.node_idx))

    pool_np = data.head_pool[relation] if direction == "head" else data.tail_pool[relation]
    pool = torch.as_tensor(np.array(pool_np), dtype=torch.long, device=device)
    positions = torch.full((data.entities.num_entities,), -1, dtype=torch.long, device=device)
    positions[pool] = torch.arange(len(pool), device=device)
    candidate_emb = z[pool]

    rel_id = data.rel2id[relation]
    anchor_col, gold_col = (2, 0) if direction == "head" else (0, 2)

    rows: List[Dict] = []
    with torch.no_grad():
        for start in range(0, len(triples), batch_size):
            chunk = triples[start : start + batch_size]
            chunk_t = torch.as_tensor(chunk, dtype=torch.long, device=device)
            anchor = z[chunk_t[:, anchor_col]]
            rel = torch.full((len(chunk),), rel_id, dtype=torch.long, device=device)
            scores = model.decoder.score_against(anchor, rel, candidate_emb)

            for i, triple in enumerate(chunk):
                gold = int(triple[gold_col])
                query = int(triple[anchor_col])
                row_scores = scores[i].clone()

                # rank over everything except *other* known-true answers
                if direction == "head":
                    source = data.true_heads if filter_mode == "all" else data.train_heads
                    known = source.get((rel_id, int(triple[2])), set())
                else:
                    source = data.true_tails if filter_mode == "all" else data.train_tails
                    known = source.get((int(triple[0]), rel_id), set())
                drop = [g for g in known if g != gold]
                if drop:
                    pos = positions[torch.as_tensor(drop, device=device)]
                    row_scores[pos[pos >= 0]] = float("-inf")

                gold_pos = int(positions[gold])
                gold_rank = (
                    int((row_scores > row_scores[gold_pos]).sum().item()) + 1
                    if gold_pos >= 0
                    else -1
                )

                top = torch.topk(row_scores, min(k, len(pool))).indices
                top_gids = pool[top].tolist()
                rows.append(
                    {
                        "relation": relation,
                        "split": split,
                        "direction": direction,
                        "query_global_id": query,
                        "query_type": type_of.get(query, ""),
                        "query_idx": idx_of.get(query, -1),
                        "query_name": name_of.get(query, ""),
                        "gold_global_id": gold,
                        "gold_name": name_of.get(gold, ""),
                        "gold_rank": gold_rank,
                        "gold_in_candidates": gold in top_gids,
                        "candidate_global_ids": json.dumps(top_gids),
                        "candidate_names": json.dumps(
                            [name_of.get(g, "") for g in top_gids]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def run(
    out_dir: Path = DEFAULT_OUT_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    relations: Sequence[str] = TARGET_RELATIONS,
    splits: Sequence[str] = ("valid", "test"),
    k: int = DEFAULT_K,
    direction: str = "head",
    capped: bool = True,
    device: str = "auto",
    filter_mode: str = "all",
) -> Dict:
    import torch

    out_dir, model_dir = Path(out_dir), Path(model_dir)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    data = build_dataset(out_dir, capped=capped)
    model, checkpoint = load_model(
        model_dir, data.entities.num_entities, data.num_relations, device
    )
    data.entities.save(model_dir / "entities_global.csv")

    edge_index = torch.as_tensor(data.edge_index, dtype=torch.long, device=device)
    edge_type = torch.as_tensor(data.edge_type, dtype=torch.long, device=device)
    z = export_embeddings(model, data, edge_index, edge_type, model_dir)

    candidates_dir = out_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict] = {}
    for relation in relations:
        for split in splits:
            frame = candidates_for_split(
                model, data, z, relation, split, device, k=k, direction=direction,
                filter_mode=filter_mode,
            )
            if frame.empty:
                print(f"  {relation}/{split}: no triples, skipped")
                continue
            slug = relation.replace(" ", "_").replace("/", "_")
            suffix = "" if filter_mode == "all" else f"_{filter_mode}filtered"
            path = candidates_dir / f"{slug}_{split}_candidates{suffix}.csv"
            frame.to_csv(path, index=False)

            recall = float(frame.gold_in_candidates.mean())
            mrr = float((1.0 / frame.gold_rank.clip(lower=1)).mean())
            summary[f"{relation}/{split}"] = {
                "num_queries": int(len(frame)),
                f"recall@{k}": round(recall, 5),
                "MRR": round(mrr, 5),
                "path": str(path),
            }
            print(
                f"  {relation}/{split}: {len(frame):,} queries | "
                f"recall@{k} {recall:.3f} | MRR {mrr:.4f} -> {path.name}"
            )

    report = {
        "k": k,
        "direction": direction,
        "filter_mode": filter_mode,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_valid_mrr": checkpoint.get("valid_mrr"),
        "summary": summary,
    }
    suffix = "" if filter_mode == "all" else f"_{filter_mode}filtered"
    (candidates_dir / f"candidates_report{suffix}.json").write_text(json.dumps(report, indent=2))
    print(
        f"\nrecall@{k} is the ceiling for the LLM on each split - it can only pick "
        "an answer that made the candidate set."
    )
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--relations", nargs="+", default=list(TARGET_RELATIONS))
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument("-k", "--top-k", type=int, default=DEFAULT_K)
    parser.add_argument("--direction", choices=["head", "tail"], default="head")
    parser.add_argument("--filter", dest="filter_mode", choices=["all", "train"],
                        default="all",
                        help="which known-true answers to remove before taking the "
                             "top-k: 'all' is the standard filtered protocol, 'train' "
                             "only removes training triples")
    parser.add_argument("--uncapped", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(list(argv) if argv is not None else None)

    run(
        Path(args.out_dir),
        Path(args.model_dir),
        args.relations,
        args.splits,
        args.top_k,
        args.direction,
        capped=not args.uncapped,
        device=args.device,
        filter_mode=args.filter_mode,
    )


if __name__ == "__main__":
    main()
