"""Train the R-GCN ranker and evaluate it with filtered MRR / Hits@k.

    python -m drkgc.ranker.train --out-dir drkgc/data --model-dir drkgc/models/rgcn

Protocol
--------
* **Head prediction** by default — given `(?, indication, disease)`, rank drugs.
  That is the drug-repurposing direction and the one DrKGC's case study uses.
  `--mode both` also evaluates tail prediction.
* **Filtered** ranking (Bordes et al. 2013): other entities that form a known
  true triple with the same query are removed from the ranking, using
  train+valid+test as the filter set. Ties are broken optimistically
  (`rank = 1 + #{candidates scoring strictly higher}`).
* Candidates are **type-constrained**: heads of `indication` are ranked against
  drugs only, never against genes or phenotypes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.config import DEFAULT_OUT_DIR, DRKGC_ROOT, SEED, TARGET_RELATIONS
from drkgc.ranker.data import RankerData, build_dataset, describe

DEFAULT_MODEL_DIR = DRKGC_ROOT / "models" / "rgcn"
HITS_AT = (1, 3, 10)


# ---------------------------------------------------------------------------
# negative sampling
# ---------------------------------------------------------------------------


def sample_negatives(
    triples,  # LongTensor [B, 3]
    data: RankerData,
    pools_head,  # Dict[int, LongTensor]
    pools_tail,
    num_negatives: int,
    generator,
):
    """Type-constrained corruption of both head and tail.

    Returns `[B, 2 * num_negatives, 3]` corrupted triples. Corruptions are drawn
    from the candidate pool of the *right node type* for that relation, so a
    negative for `(?, indication, disease)` is always some other drug.
    """
    import torch

    batch = triples.size(0)
    negatives = triples.unsqueeze(1).repeat(1, 2 * num_negatives, 1)

    for rel_id in torch.unique(triples[:, 1]).tolist():
        rows = (triples[:, 1] == rel_id).nonzero(as_tuple=True)[0]
        head_pool, tail_pool = pools_head[rel_id], pools_tail[rel_id]

        idx = torch.randint(
            len(head_pool), (len(rows), num_negatives), device=triples.device,
            generator=generator,
        )
        negatives[rows, :num_negatives, 0] = head_pool[idx]

        idx = torch.randint(
            len(tail_pool), (len(rows), num_negatives), device=triples.device,
            generator=generator,
        )
        negatives[rows, num_negatives:, 2] = tail_pool[idx]

    return negatives


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def _pool_positions(pool, num_entities: int):
    import torch

    positions = torch.full((num_entities,), -1, dtype=torch.long, device=pool.device)
    positions[pool] = torch.arange(len(pool), device=pool.device)
    return positions


def evaluate(
    model,
    data: RankerData,
    triples_np: np.ndarray,
    relation: str,
    device,
    edge_index,
    edge_type,
    mode: str = "head",
    batch_size: int = 128,
    z=None,
) -> Dict[str, float]:
    """Filtered MRR / Hits@k for one relation and one evaluation split."""
    import torch

    if len(triples_np) == 0:
        return {"num_triples": 0}

    model.eval()
    with torch.no_grad():
        if z is None:
            z = model.encode(edge_index, edge_type)

        rel_id = data.rel2id[relation]
        results: Dict[str, List[float]] = {}

        for direction in ("head", "tail") if mode == "both" else (mode,):
            pool_np = (
                data.head_pool[relation] if direction == "head" else data.tail_pool[relation]
            )
            pool = torch.as_tensor(pool_np, dtype=torch.long, device=device)
            positions = _pool_positions(pool, data.entities.num_entities)
            candidates = z[pool]
            filter_map = data.true_heads if direction == "head" else data.true_tails

            ranks: List[int] = []
            for start in range(0, len(triples_np), batch_size):
                chunk = triples_np[start : start + batch_size]
                chunk_t = torch.as_tensor(chunk, dtype=torch.long, device=device)
                anchor_col = 2 if direction == "head" else 0
                gold_col = 0 if direction == "head" else 2

                anchor = z[chunk_t[:, anchor_col]]
                rel = torch.full((len(chunk),), rel_id, dtype=torch.long, device=device)
                scores = model.decoder.score_against(anchor, rel, candidates)

                for i, triple in enumerate(chunk):
                    gold = int(triple[gold_col])
                    key = (
                        (rel_id, int(triple[2]))
                        if direction == "head"
                        else (int(triple[0]), rel_id)
                    )
                    known = filter_map.get(key, ())
                    if known:
                        drop = [g for g in known if g != gold]
                        if drop:
                            pos = positions[torch.as_tensor(drop, device=device)]
                            scores[i, pos[pos >= 0]] = float("-inf")
                    gold_pos = int(positions[gold])
                    if gold_pos < 0:  # gold not in the type pool - should not happen
                        ranks.append(len(pool))
                        continue
                    gold_score = scores[i, gold_pos]
                    ranks.append(int((scores[i] > gold_score).sum().item()) + 1)

            ranks_arr = np.asarray(ranks, dtype=np.float64)
            prefix = "" if mode != "both" else f"{direction}_"
            results[f"{prefix}MRR"] = float((1.0 / ranks_arr).mean())
            results[f"{prefix}MR"] = float(ranks_arr.mean())
            for k in HITS_AT:
                results[f"{prefix}Hits@{k}"] = float((ranks_arr <= k).mean())

        if mode == "both":
            for metric in ["MRR", "MR"] + [f"Hits@{k}" for k in HITS_AT]:
                results[metric] = float(
                    np.mean([results[f"head_{metric}"], results[f"tail_{metric}"]])
                )

    results["num_triples"] = int(len(triples_np))
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in results.items()}


def evaluate_all(
    model,
    data: RankerData,
    split: str,
    device,
    edge_index,
    edge_type,
    relations: Sequence[str] = TARGET_RELATIONS,
    mode: str = "head",
) -> Dict[str, Dict[str, float]]:
    import torch

    with torch.no_grad():
        z = model.encode(edge_index, edge_type)
    out = {
        relation: evaluate(
            model, data, data.eval_triples[(relation, split)], relation, device,
            edge_index, edge_type, mode=mode, z=z,
        )
        for relation in relations
        if (relation, split) in data.eval_triples
    }
    return out


def mean_mrr(metrics: Dict[str, Dict[str, float]]) -> float:
    values = [m["MRR"] for m in metrics.values() if m.get("num_triples")]
    return float(np.mean(values)) if values else 0.0


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train(
    out_dir: Path = DEFAULT_OUT_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    dim: int = 200,
    num_layers: int = 2,
    num_bases: Optional[int] = None,
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    epochs: int = 300,
    batch_size: int = 4096,
    num_negatives: int = 64,
    eval_every: int = 5,
    patience: int = 8,
    mode: str = "head",
    capped: bool = True,
    train_on_aux: bool = True,
    device: str = "auto",
    seed: int = SEED,
) -> Dict:
    import torch
    import torch.nn.functional as F

    from drkgc.ranker.model import RGCNRanker

    out_dir, model_dir = Path(out_dir), Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Assembling dataset ...")
    data = build_dataset(out_dir, capped=capped, train_on_aux=train_on_aux)
    print(describe(data))
    data.entities.save(model_dir / "entities_global.csv")

    edge_index = torch.as_tensor(data.edge_index, dtype=torch.long, device=device)
    edge_type = torch.as_tensor(data.edge_type, dtype=torch.long, device=device)
    train_triples = torch.as_tensor(data.train_triples, dtype=torch.long, device=device)

    pools_head = {
        data.rel2id[r]: torch.as_tensor(p, dtype=torch.long, device=device)
        for r, p in data.head_pool.items()
    }
    pools_tail = {
        data.rel2id[r]: torch.as_tensor(p, dtype=torch.long, device=device)
        for r, p in data.tail_pool.items()
    }

    model = RGCNRanker(
        data.entities.num_entities, data.num_relations, dim, num_layers, num_bases, dropout
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator(device=device).manual_seed(seed)

    print(
        f"\nmodel: {sum(p.numel() for p in model.parameters()):,} parameters "
        f"| device: {device} | {len(train_triples):,} training triples"
    )

    best = {"mrr": -1.0, "epoch": -1, "metrics": {}}
    history: List[Dict] = []
    since_improved = 0

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_triples), device=device, generator=generator)
        total_loss, num_batches = 0.0, 0
        started = time.time()

        for start in range(0, len(permutation), batch_size):
            batch = train_triples[permutation[start : start + batch_size]]
            optimiser.zero_grad()

            z = model.encode(edge_index, edge_type)
            pos_score = model.score(z, batch)
            negatives = sample_negatives(
                batch, data, pools_head, pools_tail, num_negatives, generator
            )
            neg_score = model.score(z, negatives.reshape(-1, 3)).reshape(len(batch), -1)

            scores = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)
            labels = torch.zeros_like(scores)
            labels[:, 0] = 1.0
            loss = F.binary_cross_entropy_with_logits(scores, labels)

            loss.backward()
            optimiser.step()
            total_loss += float(loss.item())
            num_batches += 1

        epoch_loss = total_loss / max(num_batches, 1)
        line = f"epoch {epoch:>4}  loss {epoch_loss:.4f}  ({time.time() - started:.1f}s)"

        if epoch % eval_every == 0 or epoch == epochs:
            metrics = evaluate_all(model, data, "valid", device, edge_index, edge_type, mode=mode)
            score = mean_mrr(metrics)
            line += f"  valid MRR {score:.4f}"
            history.append({"epoch": epoch, "loss": epoch_loss, "valid": metrics})

            if score > best["mrr"]:
                best = {"mrr": score, "epoch": epoch, "metrics": metrics}
                since_improved = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": {
                            "dim": dim,
                            "num_layers": num_layers,
                            "num_bases": num_bases,
                            "dropout": dropout,
                            "num_entities": data.entities.num_entities,
                            "num_relations": data.num_relations,
                            "relations": data.relations,
                            "capped": capped,
                            "train_on_aux": train_on_aux,
                            "seed": seed,
                        },
                        "epoch": epoch,
                        "valid_mrr": score,
                    },
                    model_dir / "model.pt",
                )
                line += "  *"
            else:
                since_improved += 1
        print(line, flush=True)

        if since_improved >= patience:
            print(f"early stopping: no valid-MRR improvement for {patience} evaluations")
            break

    # ---- final evaluation with the best checkpoint -------------------------
    if best["epoch"] < 0:
        raise SystemExit(
            f"no checkpoint was written - training ran {epochs} epoch(s) but "
            f"--eval-every is {eval_every}, so validation never ran. Lower "
            "--eval-every or raise --epochs."
        )
    checkpoint = torch.load(model_dir / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    final = {
        "valid": evaluate_all(model, data, "valid", device, edge_index, edge_type, mode=mode),
        "test": evaluate_all(model, data, "test", device, edge_index, edge_type, mode=mode),
    }

    report = {
        "best_epoch": best["epoch"],
        "best_valid_mrr": best["mrr"],
        "mode": mode,
        "metrics": final,
        "dataset": data.meta,
        "hyperparameters": {
            "dim": dim, "num_layers": num_layers, "num_bases": num_bases,
            "dropout": dropout, "lr": lr, "weight_decay": weight_decay,
            "epochs": epochs, "batch_size": batch_size,
            "num_negatives": num_negatives, "seed": seed,
        },
        "history": history,
    }
    (model_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'=' * 72}\nBEST EPOCH {best['epoch']} (valid MRR {best['mrr']:.4f})\n{'=' * 72}")
    for split, per_relation in final.items():
        for relation, metrics in per_relation.items():
            if not metrics.get("num_triples"):
                continue
            summary = "  ".join(
                f"{k}={metrics[k]:.4f}"
                for k in ("MRR", "Hits@1", "Hits@3", "Hits@10")
                if k in metrics
            )
            print(f"{split:<6} {relation:<18} n={metrics['num_triples']:>6,}  {summary}")
    print(f"\nsaved -> {model_dir}")
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="step-1 artifact directory")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--num-bases", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-negatives", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--mode", choices=["head", "tail", "both"], default="head")
    parser.add_argument("--uncapped", action="store_true",
                        help="train on the uncapped context graph")
    parser.add_argument("--no-aux-supervision", action="store_true",
                        help="use auxiliary edges for message passing only")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(list(argv) if argv is not None else None)

    train(
        out_dir=Path(args.out_dir),
        model_dir=Path(args.model_dir),
        dim=args.dim,
        num_layers=args.layers,
        num_bases=args.num_bases,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        eval_every=args.eval_every,
        patience=args.patience,
        mode=args.mode,
        capped=not args.uncapped,
        train_on_aux=not args.no_aux_supervision,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
