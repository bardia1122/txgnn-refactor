"""Degree statistics and degree capping for hub gene/protein nodes.

PrimeKG's gene/protein nodes are extremely heavy-tailed (a handful of hub
proteins dominate the PPI network). DrKGC caps the degree of gene/protein
entities so that subgraph retrieval is not swamped by hubs. This module

1. reports the degree distribution (min / median / mean / p95 / p99 / max), and
2. produces a capped copy of the auxiliary graph, keeping the uncapped one
   around for comparison.

"Degree" here is the *incident* degree in the auxiliary graph: every edge that
touches the node counts once, regardless of the stored orientation or relation.

Run standalone::

    python -m drkgc.data_prep.degree_cap --out-dir drkgc/data --percentile 95
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.config import (
    CONTEXT_DIR,
    DEFAULT_KG_FOLDER,
    DEFAULT_OUT_DIR,
    DEGREE_CAP_PERCENTILE,
    DISEASE_TYPE,
    DRUG_TYPE,
    HUB_NODE_TYPE,
    SEED,
)
from drkgc.data_prep.build_context_graph import (
    save_context_graph,
    to_hetero_data,
)
from drkgc.data_prep.kg_loader import load_kg_directed, node_type_sizes


# ---------------------------------------------------------------------------
# degrees
# ---------------------------------------------------------------------------


def incident_positions(edges: pd.DataFrame, node_type: str) -> Dict[int, List[int]]:
    """node_idx -> positions (row numbers) of the edges incident to that node."""
    positions: Dict[int, List[int]] = defaultdict(list)
    head_type = edges.head_type.values
    head_idx = edges.head_idx.values
    tail_type = edges.tail_type.values
    tail_idx = edges.tail_idx.values
    for pos in range(len(edges)):
        if head_type[pos] == node_type:
            positions[int(head_idx[pos])].append(pos)
        if tail_type[pos] == node_type and not (
            head_type[pos] == node_type and head_idx[pos] == tail_idx[pos]
        ):
            # a self-loop must not be counted twice for the same node
            positions[int(tail_idx[pos])].append(pos)
    return dict(positions)


def compute_degrees(edges: pd.DataFrame, node_type: str = HUB_NODE_TYPE) -> pd.Series:
    """Incident degree of every node of `node_type` that has at least one edge."""
    head = edges.loc[edges.head_type == node_type, "head_idx"]
    tail = edges.loc[edges.tail_type == node_type, "tail_idx"]
    counts = pd.concat([head, tail]).value_counts()

    # self-loops were counted twice above
    self_loops = edges[
        (edges.head_type == node_type)
        & (edges.tail_type == node_type)
        & (edges.head_idx == edges.tail_idx)
    ]
    if len(self_loops):
        adjust = self_loops.head_idx.value_counts()
        counts = counts.subtract(adjust, fill_value=0)

    counts = counts.astype(int).sort_index()
    counts.index = counts.index.astype(int)
    counts.name = "degree"
    counts.index.name = "node_idx"
    return counts


def degree_stats(degrees: pd.Series, percentiles: Iterable[float] = (50, 75, 90, 95, 99)) -> Dict:
    """min / max / mean / median / percentiles of a degree series."""
    if len(degrees) == 0:
        return {"num_nodes_with_edges": 0}
    values = degrees.values.astype(float)
    stats = {
        "num_nodes_with_edges": int(len(values)),
        "num_edge_endpoints": int(values.sum()),
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
    }
    for p in percentiles:
        stats[f"p{int(p)}"] = float(np.percentile(values, p))
    return stats


# ---------------------------------------------------------------------------
# capping
# ---------------------------------------------------------------------------


def cap_degrees(
    edges: pd.DataFrame,
    node_type: str = HUB_NODE_TYPE,
    cap: Optional[int] = None,
    percentile: float = DEGREE_CAP_PERCENTILE,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, Dict]:
    """Randomly subsample the edges of every over-connected `node_type` node.

    Parameters
    ----------
    cap
        Hard degree cap. If ``None`` it is the `percentile`-th percentile of the
        current degree distribution (rounded down, at least 1).
    percentile
        Percentile used when `cap` is None (default 95, exposed as a parameter).
    seed
        Fixed seed - the same edges are removed on every run.

    Returns
    -------
    (capped_edges, info)
        `capped_edges` has the same columns as `edges`, in the original row
        order minus the removed rows.

    Nodes are processed in descending-degree order and their *remaining*
    incident edges are subsampled, so after one pass every node of `node_type`
    is at or below the cap (removals afterwards can only lower a degree).
    """
    explicit_cap = cap is not None
    degrees = compute_degrees(edges, node_type)
    before = degree_stats(degrees)

    if cap is None:
        if len(degrees) == 0:
            raise ValueError(f"no {node_type!r} nodes in the auxiliary graph")
        cap = max(1, int(np.floor(np.percentile(degrees.values.astype(float), percentile))))
    cap = int(cap)

    rng = np.random.default_rng(seed)
    positions = incident_positions(edges, node_type)
    keep = np.ones(len(edges), dtype=bool)

    degree_of = {int(k): int(v) for k, v in degrees.items()}
    # descending degree, node index as a deterministic tie-break
    order = sorted(degree_of, key=lambda n: (-degree_of[n], n))

    num_capped_nodes = 0
    for node in order:
        if degree_of[node] <= cap:
            break  # sorted descending: everything after this is already fine
        remaining = [p for p in positions.get(node, ()) if keep[p]]
        if len(remaining) <= cap:
            continue
        n_remove = len(remaining) - cap
        remove = rng.choice(np.asarray(remaining), size=n_remove, replace=False)
        keep[remove] = False
        num_capped_nodes += 1

    capped = edges[keep].reset_index(drop=True)
    after = degree_stats(compute_degrees(capped, node_type))

    info = {
        "node_type": node_type,
        "cap": cap,
        "cap_source": "explicit" if explicit_cap else f"p{percentile:g}",
        "cap_percentile": None if explicit_cap else float(percentile),
        "seed": int(seed),
        "num_nodes_over_cap": int((degrees > cap).sum()),
        "num_nodes_capped": int(num_capped_nodes),
        "num_edges_before": int(len(edges)),
        "num_edges_after": int(len(capped)),
        "num_edges_removed": int(len(edges) - len(capped)),
        "degree_stats_before": before,
        "degree_stats_after": after,
        "edges_removed_per_relation": {
            str(k): int(v)
            for k, v in edges.loc[~keep, "relation"].value_counts().items()
        },
    }
    info.update(_isolation_report(edges, capped))
    return capped, info


def _isolation_report(edges: pd.DataFrame, capped: pd.DataFrame) -> Dict:
    """How many drug / disease nodes lost every auxiliary edge through capping."""
    out = {}
    for ntype, key in ((DRUG_TYPE, "drug"), (DISEASE_TYPE, "disease")):
        before = set(
            edges.loc[edges.head_type == ntype, "head_idx"].tolist()
            + edges.loc[edges.tail_type == ntype, "tail_idx"].tolist()
        )
        after = set(
            capped.loc[capped.head_type == ntype, "head_idx"].tolist()
            + capped.loc[capped.tail_type == ntype, "tail_idx"].tolist()
        )
        out[f"num_{key}_nodes_with_context_before"] = len(before)
        out[f"num_{key}_nodes_with_context_after"] = len(after)
        out[f"num_{key}_nodes_isolated_by_capping"] = len(before - after)
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run(
    out_dir: Path = DEFAULT_OUT_DIR,
    data_folder: Path = DEFAULT_KG_FOLDER,
    node_type: str = HUB_NODE_TYPE,
    cap: Optional[int] = None,
    percentile: float = DEGREE_CAP_PERCENTILE,
    seed: int = SEED,
    df: pd.DataFrame | None = None,
    write_pyg: bool = True,
) -> Dict:
    out_dir = Path(out_dir)
    ctx_dir = out_dir / CONTEXT_DIR
    edges_path = ctx_dir / "context_edges.csv"
    if not edges_path.exists():
        raise FileNotFoundError(
            f"{edges_path} not found - run build_context_graph first."
        )
    edges = pd.read_csv(edges_path, dtype={"head_id": str, "tail_id": str},
                        keep_default_na=False)

    degrees = compute_degrees(edges, node_type)
    stats_before = degree_stats(degrees)
    print(f"{node_type} degree distribution (uncapped auxiliary graph):")
    for key in ("num_nodes_with_edges", "min", "median", "mean", "p90", "p95", "p99", "max"):
        if key in stats_before:
            print(f"  {key:<22} {stats_before[key]:,.2f}" if isinstance(stats_before[key], float)
                  else f"  {key:<22} {stats_before[key]:,}")

    degrees_name = f"{node_type.replace('/', '_')}_degrees.csv"
    degrees.reset_index().to_csv(ctx_dir / degrees_name, index=False)

    capped, info = cap_degrees(edges, node_type, cap, percentile, seed)
    print(
        f"\ncap = {info['cap']} (p{percentile:g} of the uncapped distribution)"
        if cap is None
        else f"\ncap = {info['cap']} (explicit)"
    )
    print(
        f"  {info['num_nodes_over_cap']:,} nodes were over the cap; "
        f"removed {info['num_edges_removed']:,} of {info['num_edges_before']:,} edges "
        f"({info['num_edges_removed'] / max(info['num_edges_before'], 1):.1%})"
    )
    print(f"  max degree {info['degree_stats_before']['max']:,} -> "
          f"{info['degree_stats_after']['max']:,}")

    capped_path = ctx_dir / "context_edges_capped.csv"
    capped.to_csv(capped_path, index=False)
    print(f"  capped edge list -> {capped_path}")

    if write_pyg:
        if df is None:
            df = load_kg_directed(Path(data_folder))
        data = to_hetero_data(capped, node_type_sizes(df))
        save_context_graph(data, ctx_dir / "context_graph_capped.pt")
        print(f"  capped PyG HeteroData -> {ctx_dir / 'context_graph_capped.pt'}")

    (ctx_dir / "degree_stats.json").write_text(json.dumps(info, indent=2))
    return info


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--data-folder", default=str(DEFAULT_KG_FOLDER))
    parser.add_argument("--node-type", default=HUB_NODE_TYPE)
    parser.add_argument("--cap", type=int, default=None,
                        help="explicit degree cap; overrides --percentile")
    parser.add_argument("--percentile", type=float, default=DEGREE_CAP_PERCENTILE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-pyg", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    run(
        Path(args.out_dir),
        Path(args.data_folder),
        args.node_type,
        args.cap,
        args.percentile,
        args.seed,
        write_pyg=not args.no_pyg,
    )


if __name__ == "__main__":
    main()
