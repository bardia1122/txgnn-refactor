"""Assemble the step-1 artifacts into tensors the R-GCN can train on.

Everything is derived from `drkgc/data/` — this module never touches the raw KG.

Two index spaces meet here:

* step 1 works in PrimeKG's **per-node-type** index space (`(node_type, node_idx)`),
  which is what TxGNN's HeteroData uses;
* the R-GCN needs a single **global** entity index.

`EntityIndex` is the bridge, and it is written to disk so every later step
(subgraph retrieval, GCN adapter, prompts) refers to the same global ids.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.config import (
    AUX_RELATIONS,
    CONTEXT_DIR,
    DEFAULT_OUT_DIR,
    DRUG_TYPE,
    SPLITS_DIR,
    TARGET_RELATIONS,
)

SPLIT_NAMES = ("train", "valid", "test")


# ---------------------------------------------------------------------------
# entity index
# ---------------------------------------------------------------------------


@dataclass
class EntityIndex:
    """(node_type, node_idx) <-> global id, plus names."""

    table: pd.DataFrame  # global_id, node_type, node_idx, node_id, node_name
    key2gid: Dict[Tuple[str, int], int]

    @property
    def num_entities(self) -> int:
        return len(self.table)

    def gids_of_type(self, node_type: str) -> np.ndarray:
        return self.table.loc[self.table.node_type == node_type, "global_id"].values

    def map(self, node_types: Sequence[str], node_idx: Sequence[int]) -> np.ndarray:
        out = np.empty(len(node_idx), dtype=np.int64)
        for i, (t, n) in enumerate(zip(node_types, node_idx)):
            out[i] = self.key2gid[(str(t), int(n))]
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.table.to_csv(path, index=False)

    @classmethod
    def load(cls, path: Path) -> "EntityIndex":
        table = pd.read_csv(path, dtype={"node_id": str}, keep_default_na=False)
        return cls(table, _key2gid(table))


def _key2gid(table: pd.DataFrame) -> Dict[Tuple[str, int], int]:
    return {
        (str(t), int(i)): int(g)
        for t, i, g in zip(table.node_type, table.node_idx, table.global_id)
    }


def build_entity_index(out_dir: Path = DEFAULT_OUT_DIR) -> EntityIndex:
    """Global index over every entity in `context_graph/entities.csv`.

    That table already covers each node type present in the context graph plus
    drug and disease, at the *full-KG* index range, so global ids stay stable
    whether or not a node happens to have edges in this particular split.
    """
    entities = pd.read_csv(
        Path(out_dir) / CONTEXT_DIR / "entities.csv",
        dtype={"node_id": str},
        keep_default_na=False,
    )
    entities = (
        entities.sort_values(["node_type", "node_idx"])
        .drop_duplicates(subset=["node_type", "node_idx"])
        .reset_index(drop=True)
    )
    entities.insert(0, "global_id", np.arange(len(entities), dtype=np.int64))
    return EntityIndex(entities, _key2gid(entities))


# ---------------------------------------------------------------------------
# the assembled dataset
# ---------------------------------------------------------------------------


@dataclass
class RankerData:
    entities: EntityIndex
    relations: List[str]
    rel2id: Dict[str, int]
    #: (head_type, tail_type) per relation, used for type-constrained sampling
    rel_types: Dict[str, Tuple[str, str]]

    #: message-passing graph (training edges only), with inverse relations
    edge_index: "np.ndarray"  # [2, E]
    edge_type: "np.ndarray"  # [E]

    #: (h, r, t) global-id triples used as training supervision
    train_triples: "np.ndarray"  # [N, 3]
    #: evaluation triples per (relation, split)
    eval_triples: Dict[Tuple[str, str], "np.ndarray"]

    #: candidate pools per relation, as global ids
    head_pool: Dict[str, "np.ndarray"]
    tail_pool: Dict[str, "np.ndarray"]

    #: filtered-ranking support: known true heads/tails over train+valid+test
    true_heads: Dict[Tuple[int, int], Set[int]] = field(default_factory=dict)
    true_tails: Dict[Tuple[int, int], Set[int]] = field(default_factory=dict)
    #: the same, restricted to training triples (used for candidate generation)
    train_heads: Dict[Tuple[int, int], Set[int]] = field(default_factory=dict)
    train_tails: Dict[Tuple[int, int], Set[int]] = field(default_factory=dict)

    meta: Dict = field(default_factory=dict)

    @property
    def num_relations(self) -> int:
        """Including the inverse relations used for message passing."""
        return 2 * len(self.relations)

    def inverse_of(self, rel_id: int) -> int:
        return rel_id + len(self.relations)


def _load_split_frames(
    out_dir: Path, relations: Sequence[str]
) -> Dict[Tuple[str, str], pd.DataFrame]:
    frames = {}
    for relation in relations:
        slug = relation.replace(" ", "_").replace("/", "_")
        for name in SPLIT_NAMES:
            path = Path(out_dir) / SPLITS_DIR / f"{slug}_{name}.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found - run drkgc.data_prep.run_all first."
                )
            frames[(relation, name)] = pd.read_csv(
                path, dtype={"head_id": str, "tail_id": str}, keep_default_na=False
            )
    return frames


def build_dataset(
    out_dir: Path = DEFAULT_OUT_DIR,
    target_relations: Sequence[str] = TARGET_RELATIONS,
    aux_relations: Sequence[str] = AUX_RELATIONS,
    capped: bool = True,
    train_on_aux: bool = True,
) -> RankerData:
    """Turn `drkgc/data/` into a `RankerData`.

    Parameters
    ----------
    capped
        Use the degree-capped context graph (the default going forward).
    train_on_aux
        Also use the auxiliary edges as supervision, not just as message-passing
        structure. The DrKGC paper enriches its training set the same way
        (appendix A.1).
    """
    out_dir = Path(out_dir)
    entities = build_entity_index(out_dir)

    ctx_name = "context_edges_capped.csv" if capped else "context_edges.csv"
    ctx = pd.read_csv(
        Path(out_dir) / CONTEXT_DIR / ctx_name,
        dtype={"head_id": str, "tail_id": str},
        keep_default_na=False,
    )
    ctx = ctx[ctx.relation.isin(aux_relations)]

    frames = _load_split_frames(out_dir, target_relations)

    relations = list(target_relations) + sorted(ctx.relation.unique())
    rel2id = {r: i for i, r in enumerate(relations)}

    rel_types: Dict[str, Tuple[str, str]] = {}
    for relation in target_relations:
        sample = frames[(relation, "train")]
        rel_types[relation] = (str(sample.head_type.iloc[0]), str(sample.tail_type.iloc[0]))
    for relation in sorted(ctx.relation.unique()):
        sub = ctx[ctx.relation == relation]
        rel_types[relation] = (str(sub.head_type.iloc[0]), str(sub.tail_type.iloc[0]))

    # ---- triples -----------------------------------------------------------
    def to_triples(frame: pd.DataFrame, relation: str) -> np.ndarray:
        if len(frame) == 0:
            return np.zeros((0, 3), dtype=np.int64)
        h = entities.map(frame.head_type.values, frame.head_idx.values)
        t = entities.map(frame.tail_type.values, frame.tail_idx.values)
        r = np.full(len(frame), rel2id[relation], dtype=np.int64)
        return np.stack([h, r, t], axis=1)

    target_train = [to_triples(frames[(rel, "train")], rel) for rel in target_relations]
    aux_triples = [
        to_triples(ctx[ctx.relation == rel], rel) for rel in sorted(ctx.relation.unique())
    ]

    graph_triples = np.concatenate(target_train + aux_triples, axis=0)
    train_triples = (
        graph_triples if train_on_aux else np.concatenate(target_train, axis=0)
    )

    eval_triples = {
        (rel, name): to_triples(frames[(rel, name)], rel)
        for rel in target_relations
        for name in ("valid", "test")
    }

    # ---- message-passing graph (train edges + inverses) --------------------
    n_rel = len(relations)
    src, rel, dst = graph_triples[:, 0], graph_triples[:, 1], graph_triples[:, 2]
    edge_index = np.concatenate(
        [np.stack([src, dst]), np.stack([dst, src])], axis=1
    ).astype(np.int64)
    edge_type = np.concatenate([rel, rel + n_rel]).astype(np.int64)

    # ---- candidate pools ---------------------------------------------------
    head_pool, tail_pool = {}, {}
    for relation in relations:
        h_type, t_type = rel_types[relation]
        head_pool[relation] = entities.gids_of_type(h_type)
        tail_pool[relation] = entities.gids_of_type(t_type)

    # ---- filter sets -------------------------------------------------------
    true_heads: Dict[Tuple[int, int], Set[int]] = {}
    true_tails: Dict[Tuple[int, int], Set[int]] = {}
    train_heads: Dict[Tuple[int, int], Set[int]] = {}
    train_tails: Dict[Tuple[int, int], Set[int]] = {}

    def register(triples: np.ndarray, into_train: bool) -> None:
        for h, r, t in triples:
            true_heads.setdefault((int(r), int(t)), set()).add(int(h))
            true_tails.setdefault((int(h), int(r)), set()).add(int(t))
            if into_train:
                train_heads.setdefault((int(r), int(t)), set()).add(int(h))
                train_tails.setdefault((int(h), int(r)), set()).add(int(t))

    register(graph_triples, into_train=True)
    for key, triples in eval_triples.items():
        register(triples, into_train=False)

    meta = {
        "context_edges_file": ctx_name,
        "num_entities": entities.num_entities,
        "num_entities_by_type": {
            str(k): int(v) for k, v in entities.table.node_type.value_counts().items()
        },
        "relations": relations,
        "relation_types": {k: list(v) for k, v in rel_types.items()},
        "num_graph_edges": int(edge_index.shape[1]),
        "num_train_triples": int(len(train_triples)),
        "train_on_aux": bool(train_on_aux),
        "num_eval_triples": {f"{r}/{s}": int(len(v)) for (r, s), v in eval_triples.items()},
    }

    return RankerData(
        entities=entities,
        relations=relations,
        rel2id=rel2id,
        rel_types=rel_types,
        edge_index=edge_index,
        edge_type=edge_type,
        train_triples=train_triples,
        eval_triples=eval_triples,
        head_pool=head_pool,
        tail_pool=tail_pool,
        true_heads=true_heads,
        true_tails=true_tails,
        train_heads=train_heads,
        train_tails=train_tails,
        meta=meta,
    )


def describe(data: RankerData) -> str:
    lines = [
        f"entities        {data.entities.num_entities:,} "
        f"({data.meta['num_entities_by_type']})",
        f"relations       {len(data.relations)} (+{len(data.relations)} inverse) "
        f"{data.relations}",
        f"graph edges     {data.edge_index.shape[1]:,} (train edges + inverses)",
        f"train triples   {len(data.train_triples):,} "
        f"(aux as supervision: {data.meta['train_on_aux']})",
    ]
    for (rel, name), triples in sorted(data.eval_triples.items()):
        lines.append(f"eval {rel}/{name:<6} {len(triples):,}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--uncapped", action="store_true")
    args = parser.parse_args()

    dataset = build_dataset(Path(args.out_dir), capped=not args.uncapped)
    print(describe(dataset))
    print("\ndrug pool size:", len(dataset.head_pool[TARGET_RELATIONS[0]]))
