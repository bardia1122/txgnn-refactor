"""Build the auxiliary ("context") graph and run the leakage check.

The context graph holds *only* the mechanism relations

    drug_protein     drug        -> gene/protein   (drug target)
    disease_protein  gene/protein-> disease        (gene-disease association)
    protein_protein  gene/protein-> gene/protein   (PPI)

and deliberately **no** indication / contraindication / off-label-use edges:
those are the prediction target. The stored orientation of each relation is
discovered from the data (see `kg_loader.canonical_edge_type`), never hardcoded.

Artifacts (under ``<out>/context_graph/``):

    context_edges.csv     inspectable edge list (relation, types, idx, raw ids)
    context_graph.pt      the same edges as a PyG ``HeteroData``
    context_graph_meta.json
    entities.csv          node_type / node_idx / node_id / node_name
    leakage_check.json    result of the target-relation leakage check

Run standalone::

    python -m drkgc.data_prep.build_context_graph --data-folder data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.config import (
    ALL_DD_RELATIONS,
    AUX_RELATIONS,
    CONTEXT_DIR,
    DEFAULT_KG_FOLDER,
    DEFAULT_OUT_DIR,
    DISEASE_TYPE,
    DRUG_TYPE,
    TARGET_RELATIONS,
    TRIPLES_DIR,
)
from drkgc.data_prep.kg_loader import (
    build_entity_table,
    load_kg_directed,
    node_type_sizes,
)

EDGE_COLUMNS = [
    "relation",
    "head_type",
    "head_idx",
    "head_id",
    "tail_type",
    "tail_idx",
    "tail_id",
]


# ---------------------------------------------------------------------------
# edge list
# ---------------------------------------------------------------------------


def build_context_edges(
    df: pd.DataFrame,
    aux_relations: Sequence[str] = AUX_RELATIONS,
    split_filter: str | None = "train",
) -> pd.DataFrame:
    """Edge list of the auxiliary relations, in their stored orientation.

    `split_filter` matters for disease-area KGs: those carry a `split` column
    marking the edges TxGNN removes from training (all drug-disease edges into
    the held-out area *plus* a sample of its 2-hop neighbourhood, which includes
    gene/protein edges). Keeping only `split == 'train'` is what makes the
    setting genuinely zero-shot — otherwise the held-out diseases keep their
    gene associations and the context graph leaks them straight back in.
    Ignored when the KG has no `split` column (the full KG).
    """
    if split_filter and "split" in df.columns:
        before = len(df)
        df = df[df.split == split_filter]
        print(
            f"  split filter {split_filter!r}: kept {len(df):,} of {before:,} KG rows "
            f"({before - len(df):,} held-out rows excluded from the context graph)"
        )
    elif split_filter and "split" not in df.columns:
        print("  (KG has no 'split' column - nothing to filter, using every edge)")

    frames: List[pd.DataFrame] = []
    for relation in aux_relations:
        sub = df[df.relation == relation]
        if len(sub) == 0:
            raise KeyError(
                f"auxiliary relation {relation!r} not found in the KG - check "
                "config.AUX_RELATIONS against data/kg_inventory.json"
            )
        signatures = sub.groupby(["x_type", "y_type"]).size().sort_values(ascending=False)
        if len(signatures) > 1:
            print(
                f"  ! {relation!r} appears with several type signatures "
                f"{dict(signatures)}; keeping all of them."
            )
        frames.append(
            pd.DataFrame(
                {
                    "relation": relation,
                    "head_type": sub["x_type"].astype(str).values,
                    "head_idx": sub["x_idx"].astype(int).values,
                    "head_id": sub["x_id"].astype(str).values,
                    "tail_type": sub["y_type"].astype(str).values,
                    "tail_idx": sub["y_idx"].astype(int).values,
                    "tail_id": sub["y_id"].astype(str).values,
                }
            )
        )
        print(f"  {relation:<18} {len(frames[-1]):>9,} edges  {list(signatures.index)}")

    edges = pd.concat(frames, ignore_index=True)[EDGE_COLUMNS]
    before = len(edges)
    edges = edges.drop_duplicates(
        subset=["relation", "head_type", "head_idx", "tail_type", "tail_idx"]
    ).reset_index(drop=True)
    if len(edges) != before:
        print(f"  dropped {before - len(edges):,} duplicate edges")
    return edges


def context_edge_types(edges: pd.DataFrame) -> List[tuple]:
    """The canonical (src_type, relation, dst_type) triples present in `edges`."""
    combos = edges[["head_type", "relation", "tail_type"]].drop_duplicates()
    return [tuple(str(v) for v in row) for row in combos.values]


# ---------------------------------------------------------------------------
# PyG graph
# ---------------------------------------------------------------------------


def to_hetero_data(edges: pd.DataFrame, node_sizes: Dict[str, int]):
    """Convert the edge list to a PyG ``HeteroData``.

    ``num_nodes`` per type is taken from the *full* KG (same rule as
    ``txgnn.utils.create_pyg_graph``) so node indices stay interchangeable with
    the TxGNN graph. Only node types that participate in the auxiliary
    relations are added.
    """
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()
    used_types = set(edges.head_type.unique()) | set(edges.tail_type.unique())
    for ntype in sorted(used_types):
        if ntype not in node_sizes:
            raise KeyError(f"no node count for type {ntype!r}")
        data[ntype].num_nodes = int(node_sizes[ntype])

    for etype in context_edge_types(edges):
        src_type, relation, dst_type = etype
        sub = edges[
            (edges.head_type == src_type)
            & (edges.relation == relation)
            & (edges.tail_type == dst_type)
        ]
        src = torch.as_tensor(sub.head_idx.values, dtype=torch.long)
        dst = torch.as_tensor(sub.tail_idx.values, dtype=torch.long)
        data[etype].edge_index = torch.stack([src, dst])
    return data


def save_context_graph(data, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def load_context_graph(path: Path):
    """Load a saved ``HeteroData`` (torch>=2.6 defaults to weights_only=True)."""
    import torch

    try:
        return torch.load(path, weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only kwarg
        return torch.load(path)


# ---------------------------------------------------------------------------
# leakage check
# ---------------------------------------------------------------------------


def _pair_key(head_type, head_idx, tail_type, tail_idx):
    return (str(head_type), int(head_idx), str(tail_type), int(tail_idx))


def leakage_check(
    edges: pd.DataFrame,
    target_triples: pd.DataFrame,
    forbidden_relations: Sequence[str] = ALL_DD_RELATIONS,
) -> Dict:
    """Verify no target (drug,disease) pair is present in the auxiliary graph.

    Two independent checks:

    1. *relation level* - no indication / contraindication / off-label-use
       relation name occurs among the auxiliary relations;
    2. *pair level* - no (type, idx) endpoint pair of a target triple occurs as
       an auxiliary edge, in either orientation.

    The auxiliary relations connect different node types, so both are expected
    to come back clean; the point is to prove it rather than assume it.
    """
    aux_relations = sorted(edges.relation.unique())
    offending_relations = sorted(set(aux_relations) & set(forbidden_relations))

    aux_pairs = set()
    for head_type, head_idx, tail_type, tail_idx in zip(
        edges.head_type, edges.head_idx, edges.tail_type, edges.tail_idx
    ):
        aux_pairs.add(_pair_key(head_type, head_idx, tail_type, tail_idx))
        aux_pairs.add(_pair_key(tail_type, tail_idx, head_type, head_idx))

    overlapping: List[Dict] = []
    for row in target_triples.itertuples(index=False):
        key = _pair_key(row.head_type, row.head_idx, row.tail_type, row.tail_idx)
        if key in aux_pairs:
            overlapping.append(
                {
                    "relation": row.relation,
                    "head": [key[0], key[1]],
                    "tail": [key[2], key[3]],
                }
            )

    result = {
        "auxiliary_relations": aux_relations,
        "num_auxiliary_edges": int(len(edges)),
        "num_target_triples_checked": int(len(target_triples)),
        "forbidden_relations_present": offending_relations,
        "num_overlapping_pairs": len(overlapping),
        "overlapping_examples": overlapping[:20],
        "passed": not offending_relations and not overlapping,
    }
    return result


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def build(
    data_folder: Path = DEFAULT_KG_FOLDER,
    out_dir: Path = DEFAULT_OUT_DIR,
    aux_relations: Sequence[str] = AUX_RELATIONS,
    df: pd.DataFrame | None = None,
    write_pyg: bool = True,
    area: str | None = None,
    split_filter: str | None = "train",
) -> Dict:
    data_folder, out_dir = Path(data_folder), Path(out_dir)
    ctx_dir = out_dir / CONTEXT_DIR
    ctx_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = load_kg_directed(data_folder, area=area)

    print("Building auxiliary/context graph ...")
    edges = build_context_edges(df, aux_relations, split_filter)
    edges_path = ctx_dir / "context_edges.csv"
    edges.to_csv(edges_path, index=False)
    print(f"  {len(edges):,} edges -> {edges_path}")

    node_sizes = node_type_sizes(df)
    meta = {
        "area": area,
        "split_filter": split_filter if "split" in df.columns else None,
        "aux_relations": list(aux_relations),
        "edge_types": [list(et) for et in context_edge_types(edges)],
        "num_edges": int(len(edges)),
        "num_edges_per_relation": {
            str(k): int(v) for k, v in edges.relation.value_counts().items()
        },
        "node_types": sorted(set(edges.head_type) | set(edges.tail_type)),
        "num_nodes": {
            k: int(node_sizes[k]) for k in sorted(set(edges.head_type) | set(edges.tail_type))
        },
        "note": (
            "num_nodes follows txgnn.utils.create_pyg_graph (max index in the full "
            "KG + 1), so node indices are interchangeable with the TxGNN graph. "
            "Only the stored orientation is kept; add reverse edges at message-"
            "passing time if the GCN needs them."
        ),
    }
    (ctx_dir / "context_graph_meta.json").write_text(json.dumps(meta, indent=2))

    if write_pyg:
        data = to_hetero_data(edges, node_sizes)
        save_context_graph(data, ctx_dir / "context_graph.pt")
        print(f"  PyG HeteroData -> {ctx_dir / 'context_graph.pt'}")

    # entity table (names for prompts later) for every type in play
    entity_types = sorted(
        set(edges.head_type) | set(edges.tail_type) | {DRUG_TYPE, DISEASE_TYPE}
    )
    entities = build_entity_table(df, data_folder, entity_types)
    entities.to_csv(ctx_dir / "entities.csv", index=False)
    print(f"  {len(entities):,} entities -> {ctx_dir / 'entities.csv'}")

    # leakage check against every target triple we extracted
    target_path = out_dir / TRIPLES_DIR / "target_triples.csv"
    if target_path.exists():
        targets = pd.read_csv(target_path, dtype={"head_id": str, "tail_id": str},
                              keep_default_na=False)
    else:
        print(f"  ! {target_path} missing - run extract_triples first; "
              "falling back to extracting targets from the KG in memory")
        targets = df[df.relation.isin(TARGET_RELATIONS)].rename(
            columns={
                "x_type": "head_type",
                "x_idx": "head_idx",
                "y_type": "tail_type",
                "y_idx": "tail_idx",
            }
        )[["relation", "head_type", "head_idx", "tail_type", "tail_idx"]]

    report = leakage_check(edges, targets)
    (ctx_dir / "leakage_check.json").write_text(json.dumps(report, indent=2))
    status = "PASSED" if report["passed"] else "FAILED"
    print(
        f"  leakage check {status}: {report['num_overlapping_pairs']} overlapping "
        f"pairs over {report['num_target_triples_checked']:,} target triples; "
        f"forbidden relations present: {report['forbidden_relations_present']}"
    )
    return {"edges": edges, "meta": meta, "leakage": report}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", default=str(DEFAULT_KG_FOLDER))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--aux-relations", nargs="+", default=list(AUX_RELATIONS))
    parser.add_argument("--area", default=None,
                        help="build from this disease area's KG instead of the full KG")
    parser.add_argument("--split-filter", default="train",
                        help="KG split to keep for area KGs ('none' to keep everything)")
    parser.add_argument(
        "--no-pyg",
        action="store_true",
        help="only write the CSV edge list (skips the torch/PyG import)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    build(
        Path(args.data_folder),
        Path(args.out_dir),
        args.aux_relations,
        write_pyg=not args.no_pyg,
        area=args.area,
        split_filter=None if args.split_filter == "none" else args.split_filter,
    )


if __name__ == "__main__":
    main()
