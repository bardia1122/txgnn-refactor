"""Turn retrieved subgraphs into per-query graphs the GCN can run on.

Each JSONL record from `drkgc.retrieval.subgraph` becomes a small graph:

* nodes = every entity appearing in the subgraph, **plus** the query entity and
  all candidates even when a candidate ended up unreachable — the adapter must
  emit a vector for every candidate the prompt will list, connected or not;
* node features = that entity's global embedding from step 2;
* edges = the retrieved triples, typed by relation id.

Node ids are remapped to a local 0..N-1 range per subgraph, with `query_index`
and `candidate_index` pointing into it, because the GCN runs per query.

Run standalone::

    python -m drkgc.adapter.data --out-dir drkgc/data_holdout \\
        --model-dir drkgc/models/rgcn_holdout_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.config import DEFAULT_OUT_DIR, DRKGC_ROOT, TARGET_RELATIONS

DEFAULT_MODEL_DIR = DRKGC_ROOT / "models" / "rgcn"


@dataclass
class SubgraphSample:
    """One query, ready for the GCN."""

    #: global entity ids, in local order
    node_ids: np.ndarray
    #: [2, E] local indices
    edge_index: np.ndarray
    #: [E] relation ids
    edge_type: np.ndarray
    #: local index of the query entity
    query_index: int
    #: local indices of the candidates, in prompt order
    candidate_index: np.ndarray
    #: bookkeeping carried through for the prompt builder
    meta: Dict

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)


def iter_subgraph_records(path: Path) -> Iterator[Dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def record_to_sample(record: Dict, rel2id: Dict[str, int]) -> SubgraphSample:
    """Remap one JSONL record into local index space."""
    query = int(record["query_global_id"])
    candidates = [int(c) for c in record["candidates"]]

    # the query and every candidate must have a node, even with no edges
    local: Dict[int, int] = {}

    def index_of(global_id: int) -> int:
        if global_id not in local:
            local[global_id] = len(local)
        return local[global_id]

    index_of(query)
    for candidate in candidates:
        index_of(candidate)

    rows, cols, types = [], [], []
    for head, relation, tail in record["triples"]:
        relation_id = rel2id.get(relation)
        if relation_id is None:
            continue  # a relation the ranker did not know about
        rows.append(index_of(int(head)))
        cols.append(index_of(int(tail)))
        types.append(relation_id)

    node_ids = np.empty(len(local), dtype=np.int64)
    for global_id, position in local.items():
        node_ids[position] = global_id

    edge_index = (
        np.stack([np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)])
        if rows
        else np.zeros((2, 0), dtype=np.int64)
    )

    return SubgraphSample(
        node_ids=node_ids,
        edge_index=edge_index,
        edge_type=np.asarray(types, dtype=np.int64),
        query_index=local[query],
        candidate_index=np.asarray([local[c] for c in candidates], dtype=np.int64),
        meta={
            "relation": record["relation"],
            "split": record["split"],
            "query_global_id": query,
            "query_name": record.get("query_name", ""),
            "gold_global_id": record.get("gold_global_id"),
            "gold_in_candidates": record.get("gold_in_candidates"),
            "candidates": candidates,
            "num_triples": len(record["triples"]),
        },
    )


def load_global_embeddings(model_dir: Path):
    """The `E_global` table exported by `drkgc.ranker.rank`."""
    import torch

    path = Path(model_dir) / "global_embeddings.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python -m drkgc.ranker.rank` first."
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return payload["embeddings"]


def subgraph_path(out_dir: Path, relation: str, split: str) -> Path:
    slug = relation.replace(" ", "_").replace("/", "_")
    return Path(out_dir) / "subgraphs" / f"{slug}_{split}_subgraphs.jsonl"


def load_samples(
    out_dir: Path,
    relation: str,
    split: str,
    rel2id: Dict[str, int],
    limit: Optional[int] = None,
) -> List[SubgraphSample]:
    path = subgraph_path(out_dir, relation, split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python -m drkgc.retrieval.subgraph` first."
        )
    samples = []
    for i, record in enumerate(iter_subgraph_records(path)):
        if limit and i >= limit:
            break
        samples.append(record_to_sample(record, rel2id))
    return samples


def collate(samples: Sequence[SubgraphSample]):
    """Batch several subgraphs as one disconnected graph (PyG's convention).

    Offsetting each subgraph's local indices keeps them isolated, so one GCN call
    processes a whole batch of queries without them exchanging messages.
    """
    import torch

    widths = {len(s.candidate_index) for s in samples}
    if len(widths) > 1:
        raise ValueError(
            f"subgraphs have differing candidate counts {sorted(widths)}; "
            "collate assumes a uniform k per query"
        )

    node_ids, edge_index, edge_type = [], [], []
    query_index, candidate_index, batch_vector = [], [], []
    offset = 0
    for i, sample in enumerate(samples):
        node_ids.append(sample.node_ids)
        if sample.edge_index.shape[1]:
            edge_index.append(sample.edge_index + offset)
            edge_type.append(sample.edge_type)
        query_index.append(sample.query_index + offset)
        candidate_index.append(sample.candidate_index + offset)
        batch_vector.append(np.full(sample.num_nodes, i, dtype=np.int64))
        offset += sample.num_nodes

    return {
        "node_ids": torch.as_tensor(np.concatenate(node_ids), dtype=torch.long),
        "edge_index": torch.as_tensor(
            np.concatenate(edge_index, axis=1) if edge_index else np.zeros((2, 0)),
            dtype=torch.long,
        ),
        "edge_type": torch.as_tensor(
            np.concatenate(edge_type) if edge_type else np.zeros(0), dtype=torch.long
        ),
        "query_index": torch.as_tensor(np.asarray(query_index), dtype=torch.long),
        # already offset in the loop above; stacking needs a uniform candidate
        # count, which holds because rank.py emits exactly k per query
        "candidate_index": torch.as_tensor(
            np.stack(candidate_index), dtype=torch.long
        ),
        "batch": torch.as_tensor(np.concatenate(batch_vector), dtype=torch.long),
        "meta": [s.meta for s in samples],
    }


def main(argv: Iterable[str] | None = None) -> None:
    from drkgc.ranker.data import build_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--relation", default=TARGET_RELATIONS[0])
    parser.add_argument("--split", default="valid")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args(list(argv) if argv is not None else None)

    data = build_dataset(Path(args.out_dir))
    samples = load_samples(
        Path(args.out_dir), args.relation, args.split, data.rel2id, args.limit
    )
    print(f"{len(samples)} subgraphs from {args.relation}/{args.split}\n")
    for sample in samples:
        print(
            f"  {sample.meta['query_name'][:40]:<40} "
            f"nodes={sample.num_nodes:>4}  edges={sample.edge_index.shape[1]:>4}  "
            f"candidates={len(sample.candidate_index)}"
        )
    batch = collate(samples)
    print(
        f"\nbatched: {batch['node_ids'].numel()} nodes, "
        f"{batch['edge_index'].shape[1]} edges, "
        f"candidate_index {tuple(batch['candidate_index'].shape)}"
    )


if __name__ == "__main__":
    main()
