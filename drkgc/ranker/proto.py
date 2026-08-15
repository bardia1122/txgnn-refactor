"""Prototype embeddings for zero-shot diseases, ported from TxGNN.

A disease held out of training has no drug edges, so its learned embedding is
essentially untrained noise and the ranker cannot place it. TxGNN's answer
(`pyg_implementation/txgnn/model.py`, `DistMultPredictor` proto-learning) is to
*borrow*: describe every disease by the neighbours it has in the auxiliary
graph, find the training diseases with the most similar description, and mix
their embeddings into the query's.

The port keeps TxGNN's structure and defaults:

1. **Profile** — a binary incidence vector over the disease's neighbours in the
   context graph (`obtain_disease_profile`, `utils.py:1015`).
2. **Similarity** — cosine between profiles (`sim_matrix`, `utils.py:960`).
3. **Prototype** — the `proto_num`(=5) most similar *training* diseases, their
   similarities L1-normalised into weights, applied to their embeddings.
4. **Blend** — `z = (1 - a) * z_own + a * z_proto`, with `a` from `agg_measure`;
   the default `rarity` sets `a = lambda * exp(-lambda * degree) + 0.2`
   (`exponential`, `utils.py:1036`), so a disease with **no** training edges
   leans almost entirely on the prototype while well-connected diseases keep
   their own embedding.

Two deliberate deviations from TxGNN, both noted in the README:

* the prototype context is built **once over all target relations** rather than
  per relation — with a shared disease partition a held-out disease has zero
  edges of every target relation anyway;
* self-matches are masked out of the top-k explicitly, instead of TxGNN's
  "drop column 0 during training, keep it during evaluation" shape heuristic,
  which silently keeps the identity match whenever the query and key sets
  happen to differ in size.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.config import CONTEXT_DIR, DEFAULT_OUT_DIR, DISEASE_TYPE, TARGET_RELATIONS
from drkgc.ranker.data import RankerData

#: prototype weight per aggregation mode ('rarity' is computed per disease)
STATIC_ALPHA = {"100proto": 1.0, "avg": 0.5, "heuristics-0.8": 0.2}
AGG_MODES = ("rarity", "avg", "heuristics-0.8", "100proto")


def exponential(x: np.ndarray, lamb: float) -> np.ndarray:
    """TxGNN's rarity coefficient (`utils.py:1036`)."""
    return lamb * np.exp(-lamb * x) + 0.2


@dataclass
class PrototypeContext:
    """Everything needed to rewrite the disease rows of an embedding matrix."""

    disease_gids: np.ndarray  # [D] global ids of every disease
    key_gids: np.ndarray  # [K] diseases that have training target edges
    topk_idx: np.ndarray  # [D, k] indices into key_gids
    topk_coef: np.ndarray  # [D, k] L1-normalised similarities
    alpha: np.ndarray  # [D, 1] prototype weight per disease
    stats: Dict

    def to_torch(self, device):
        import torch

        self._t = {
            "disease_gids": torch.as_tensor(self.disease_gids, dtype=torch.long, device=device),
            "key_gids": torch.as_tensor(self.key_gids, dtype=torch.long, device=device),
            "topk_idx": torch.as_tensor(self.topk_idx, dtype=torch.long, device=device),
            "topk_coef": torch.as_tensor(self.topk_coef, dtype=torch.float, device=device),
            "alpha": torch.as_tensor(self.alpha, dtype=torch.float, device=device),
        }
        return self

    def apply(self, z):
        """Return `z` with the disease rows replaced by their blended prototype.

        Differentiable, so this works inside the training loop as well as at
        evaluation time.
        """
        t = self._t
        key_emb = z[t["key_gids"]]  # [K, dim]
        neighbours = key_emb[t["topk_idx"]]  # [D, k, dim]
        proto = (neighbours * t["topk_coef"].unsqueeze(-1)).sum(dim=1)  # [D, dim]

        own = z[t["disease_gids"]]
        alpha = t["alpha"]
        blended = (1.0 - alpha) * own + alpha * proto

        out = z.clone()
        out[t["disease_gids"]] = blended
        return out


def _profile_matrix(
    out_dir: Path,
    data: RankerData,
    capped: bool = True,
) -> Tuple["object", List[str]]:
    """Row-normalised sparse incidence matrix: one row per disease.

    Columns are every neighbour a disease has in the context graph, so with the
    default auxiliary relations this is TxGNN's `protein_profile`. Add
    `disease_disease` to the context graph and it becomes `all_nodes_profile`,
    which is what TxGNN uses by default — the code picks up whatever is there.
    """
    from scipy import sparse

    name = "context_edges_capped.csv" if capped else "context_edges.csv"
    edges = pd.read_csv(
        Path(out_dir) / CONTEXT_DIR / name,
        dtype={"head_id": str, "tail_id": str},
        keep_default_na=False,
    )

    incident = edges[(edges.head_type == DISEASE_TYPE) | (edges.tail_type == DISEASE_TYPE)]
    if len(incident) == 0:
        raise ValueError(
            "no disease-incident edges in the context graph - the prototype needs "
            "at least one relation connecting diseases to other nodes."
        )

    disease_gids = data.entities.gids_of_type(DISEASE_TYPE)
    gid2row = {int(g): i for i, g in enumerate(disease_gids)}

    # neighbour columns are keyed by (node_type, node_idx) across every relation
    rows: List[int] = []
    cols: List[int] = []
    col_of: Dict[Tuple[str, int], int] = {}
    used_relations = sorted(incident.relation.unique())

    for h_type, h_idx, t_type, t_idx in zip(
        incident.head_type, incident.head_idx, incident.tail_type, incident.tail_idx
    ):
        if h_type == DISEASE_TYPE:
            disease_key, other = (str(h_type), int(h_idx)), (str(t_type), int(t_idx))
        else:
            disease_key, other = (str(t_type), int(t_idx)), (str(h_type), int(h_idx))
        gid = data.entities.key2gid.get(disease_key)
        if gid is None or int(gid) not in gid2row:
            continue
        rows.append(gid2row[int(gid)])
        cols.append(col_of.setdefault(other, len(col_of)))

    profile = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(disease_gids), max(len(col_of), 1)),
    )
    profile.data[:] = 1.0  # duplicate (disease, neighbour) pairs collapse to 1

    # L2-normalise rows so a plain dot product is the cosine similarity
    norms = np.sqrt(np.asarray(profile.multiply(profile).sum(axis=1))).ravel()
    norms[norms == 0] = 1.0
    profile = sparse.diags(1.0 / norms) @ profile
    return profile.tocsr(), used_relations


def build_prototype_context(
    data: RankerData,
    out_dir: Path = DEFAULT_OUT_DIR,
    proto_num: int = 5,
    agg_measure: str = "rarity",
    exp_lambda: float = 0.7,
    target_relations: Sequence[str] = TARGET_RELATIONS,
    capped: bool = True,
    chunk: int = 2048,
) -> PrototypeContext:
    """Precompute the top-k similar training diseases for every disease.

    Profiles are static, so the neighbour lists and weights are computed once
    and reused every epoch — the per-step cost is one gather plus a weighted sum.
    """
    if agg_measure not in AGG_MODES:
        raise ValueError(f"agg_measure must be one of {AGG_MODES}, got {agg_measure!r}")

    profile, used_relations = _profile_matrix(out_dir, data, capped)
    disease_gids = data.entities.gids_of_type(DISEASE_TYPE)
    gid2row = {int(g): i for i, g in enumerate(disease_gids)}

    # keys: diseases carrying at least one *training* target-relation edge,
    # and the training degree that drives the rarity coefficient
    target_ids = {data.rel2id[r] for r in target_relations if r in data.rel2id}
    degree = np.zeros(len(disease_gids), dtype=np.float32)
    for h, r, t in data.train_triples:
        if int(r) in target_ids:
            row = gid2row.get(int(t))
            if row is not None:
                degree[row] += 1

    key_rows = np.where(degree > 0)[0]
    if len(key_rows) == 0:
        raise ValueError("no training diseases to build prototypes from")
    key_gids = disease_gids[key_rows]
    k = min(proto_num, len(key_rows))

    keys_t = profile[key_rows].T.tocsc()
    row_to_key_slot = {int(r): i for i, r in enumerate(key_rows)}

    topk_idx = np.zeros((len(disease_gids), k), dtype=np.int64)
    topk_coef = np.zeros((len(disease_gids), k), dtype=np.float32)
    num_without_profile = 0

    for start in range(0, len(disease_gids), chunk):
        stop = min(start + chunk, len(disease_gids))
        sims = (profile[start:stop] @ keys_t).toarray()

        # never let a disease be its own prototype
        for local, row in enumerate(range(start, stop)):
            slot = row_to_key_slot.get(row)
            if slot is not None:
                sims[local, slot] = -np.inf

        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k] if k < sims.shape[1] else (
            np.tile(np.arange(sims.shape[1]), (sims.shape[0], 1))
        )
        chunk_sims = np.take_along_axis(sims, idx, axis=1)
        order = np.argsort(-chunk_sims, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        chunk_sims = np.take_along_axis(chunk_sims, order, axis=1)

        chunk_sims[~np.isfinite(chunk_sims)] = 0.0
        chunk_sims = np.clip(chunk_sims, 0.0, None)
        totals = chunk_sims.sum(axis=1, keepdims=True)
        empty = totals.ravel() <= 0
        num_without_profile += int(empty.sum())
        totals[totals <= 0] = 1.0

        topk_idx[start:stop] = idx
        topk_coef[start:stop] = chunk_sims / totals

    if agg_measure == "rarity":
        alpha = exponential(degree, exp_lambda).astype(np.float32)
    else:
        alpha = np.full(len(disease_gids), STATIC_ALPHA[agg_measure], dtype=np.float32)

    # a disease with no usable profile has nothing to borrow - keep its own
    no_profile = topk_coef.sum(axis=1) <= 0
    alpha[no_profile] = 0.0

    zero_degree = degree == 0
    stats = {
        "profile_relations": used_relations,
        "profile_dims": int(profile.shape[1]),
        "num_diseases": int(len(disease_gids)),
        "num_key_diseases": int(len(key_rows)),
        "proto_num": int(k),
        "agg_measure": agg_measure,
        "exp_lambda": float(exp_lambda) if agg_measure == "rarity" else None,
        "num_diseases_without_profile": int(no_profile.sum()),
        "num_zero_degree_diseases": int(zero_degree.sum()),
        "num_zero_degree_without_profile": int((zero_degree & no_profile).sum()),
        "mean_alpha_zero_degree": float(alpha[zero_degree].mean()) if zero_degree.any() else 0.0,
        "mean_alpha_trained": float(alpha[~zero_degree].mean()) if (~zero_degree).any() else 0.0,
    }

    return PrototypeContext(
        disease_gids=disease_gids,
        key_gids=key_gids,
        topk_idx=topk_idx,
        topk_coef=topk_coef,
        alpha=alpha.reshape(-1, 1),
        stats=stats,
    )


def describe(ctx: PrototypeContext) -> str:
    s = ctx.stats
    lines = [
        f"prototype profile from {s['profile_relations']} "
        f"({s['profile_dims']:,} dims)",
        f"  {s['num_key_diseases']:,} training diseases usable as prototypes, "
        f"top-{s['proto_num']} per query",
        f"  aggregation {s['agg_measure']!r}"
        + (f" (lambda={s['exp_lambda']})" if s["exp_lambda"] else ""),
        f"  zero-degree diseases: {s['num_zero_degree_diseases']:,} "
        f"(mean prototype weight {s['mean_alpha_zero_degree']:.3f}) | "
        f"trained diseases mean weight {s['mean_alpha_trained']:.3f}",
    ]
    if s["num_diseases_without_profile"]:
        lines.append(
            f"  ! {s['num_diseases_without_profile']:,} diseases have no context "
            f"neighbours and keep their own embedding "
            f"({s['num_zero_degree_without_profile']:,} of them are zero-degree, "
            "so they remain unreachable)"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    from drkgc.ranker.data import build_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--proto-num", type=int, default=5)
    parser.add_argument("--agg", choices=list(AGG_MODES), default="rarity")
    parser.add_argument("--exp-lambda", type=float, default=0.7)
    args = parser.parse_args()

    dataset = build_dataset(Path(args.out_dir))
    context = build_prototype_context(
        dataset, Path(args.out_dir), args.proto_num, args.agg, args.exp_lambda
    )
    print(describe(context))
