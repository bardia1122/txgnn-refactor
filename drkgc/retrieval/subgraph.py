"""Per-query bottom-up subgraph retrieval (DrKGC section 3.5).

For each query `(?, r, t)` with a candidate set `C` from the ranker, build a small
subgraph that explains how each candidate could reach the query entity:

1. **Connectivity first** — the shortest path from every candidate to the query
   entity, so both endpoints and all candidates are in the subgraph and it is
   connected. This is the paper's first step.
2. **Rule-guided enrichment** — walk the mined rules in descending confidence and
   collect their groundings between candidates and the query, until the subgraph
   reaches `tau` triples.

Everything is drawn from the **training** graph, so no held-out edge can enter a
subgraph.

One knob the paper does not have: `--reserve-mechanism`. On this data the mined
rules are dominated by ontology shortcuts (`disease_disease`, confidence 0.137)
over mechanism paths (`drug_protein & disease_protein`, 0.006), so pure confidence
ordering fills nearly the whole budget with "drugs used for similar diseases" and
rarely surfaces the drug->target->disease chains DrKGC's case study is built on.
Reserving a fraction of `tau` for mechanism rules makes that trade-off measurable
instead of implicit. Default 0.0 = faithful to the paper.

Run standalone::

    python -m drkgc.retrieval.subgraph --out-dir drkgc/data_holdout \\
        --model-dir drkgc/models/rgcn_holdout_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.config import DEFAULT_OUT_DIR, DRKGC_ROOT, TARGET_RELATIONS
from drkgc.ranker.data import RankerData, build_dataset
from drkgc.retrieval.rules import Rule, load_rules

DEFAULT_MODEL_DIR = DRKGC_ROOT / "models" / "rgcn"
DEFAULT_TAU = 100

#: relations that carry a biological mechanism rather than an ontology shortcut
MECHANISM_RELATIONS = {"drug_protein", "disease_protein", "protein_protein"}

#: cap on groundings collected per (rule, candidate), so one hub cannot flood tau
MAX_GROUNDINGS = 5

#: a canonical triple: (head global id, relation name, tail global id)
Triple = Tuple[int, str, int]


# ---------------------------------------------------------------------------
# the retrieval graph
# ---------------------------------------------------------------------------


@dataclass
class RetrievalGraph:
    """The training graph, in the three shapes retrieval needs."""

    #: symmetric boolean adjacency for shortest-path search
    adjacency: object
    #: (u, v) -> relation name, in the direction the KG stores it
    edge_relation: Dict[Tuple[int, int], str]
    #: (relation, inverted) -> csr adjacency, for rule grounding
    matrices: Dict[Tuple[str, bool], object]
    num_entities: int
    #: `adjacency` with hub nodes removed; falls back to `adjacency` when unused
    path_adjacency: object = None
    #: nodes excluded from path finding because their degree exceeds the limit
    hub_nodes: Set[int] = field(default_factory=set)

    def canonical(self, u: int, v: int) -> Optional[Triple]:
        """Return the stored triple for an edge traversed u -> v, either way."""
        relation = self.edge_relation.get((u, v))
        if relation is not None:
            return (u, relation, v)
        relation = self.edge_relation.get((v, u))
        if relation is not None:
            return (v, relation, u)
        return None


def build_retrieval_graph(
    data: RankerData, max_hub_degree: int = 0
) -> RetrievalGraph:
    """Assemble the training graph. Uses the forward half of `data.edge_index`
    (the second half is the generated inverses) so each edge appears once.

    `max_hub_degree > 0` removes very high-degree nodes from the **path-finding**
    graph only. A category node like `Mendelian disease` (degree 1,524) creates a
    2-hop path between almost any pair of genetic conditions, and "both are
    Mendelian diseases" is not evidence. The R-GCN keeps those edges — for
    embedding propagation a category hub is a useful bridge, and capping it
    measurably hurt the ranker (see drkgc/ranker/README.md). The value of a hub
    depends on what you use it for, so the two graphs differ deliberately.
    """
    from scipy import sparse

    n = data.entities.num_entities
    id2rel = {v: k for k, v in data.rel2id.items()}

    forward = data.edge_index.shape[1] // 2
    src = data.edge_index[0, :forward]
    dst = data.edge_index[1, :forward]
    rel = data.edge_type[:forward]

    edge_relation: Dict[Tuple[int, int], str] = {}
    matrices: Dict[Tuple[str, bool], object] = {}
    for rel_id, relation in id2rel.items():
        mask = rel == rel_id
        if not mask.any():
            continue
        rows, cols = src[mask], dst[mask]
        matrix = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n)
        )
        matrix.data[:] = 1
        matrices[(relation, False)] = matrix
        matrices[(relation, True)] = matrix.T.tocsr()
        for u, v in zip(rows.tolist(), cols.tolist()):
            edge_relation.setdefault((u, v), relation)

    undirected = sparse.csr_matrix(
        (np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n, n)
    )
    undirected = undirected + undirected.T
    undirected.data[:] = 1

    path_adjacency, hub_nodes = undirected, set()
    if max_hub_degree > 0:
        degree = np.asarray(undirected.sum(axis=1)).ravel()
        hubs = np.where(degree > max_hub_degree)[0]
        if len(hubs):
            keep = np.ones(n, dtype=np.int8)
            keep[hubs] = 0
            mask = sparse.diags(keep)
            path_adjacency = (mask @ undirected @ mask).tocsr()
            path_adjacency.eliminate_zeros()
            hub_nodes = set(hubs.tolist())
        print(
            f"  excluded {len(hub_nodes):,} hub nodes (degree > {max_hub_degree}) "
            "from path finding"
        )

    return RetrievalGraph(
        undirected, edge_relation, matrices, n, path_adjacency, hub_nodes
    )


# ---------------------------------------------------------------------------
# step 1: shortest paths
# ---------------------------------------------------------------------------


def shortest_path_triples(
    graph: RetrievalGraph,
    query: int,
    candidates: Sequence[int],
) -> Tuple[List[Triple], Dict[int, int]]:
    """Shortest path from every candidate to `query`, as canonical triples.

    One `dijkstra` call from the query gives predecessors for the whole graph,
    so all candidates are covered by a single traversal (in C, not Python).
    """
    from scipy.sparse.csgraph import dijkstra

    adjacency = graph.path_adjacency if graph.path_adjacency is not None else graph.adjacency
    distances, predecessors = dijkstra(
        adjacency, indices=query, unweighted=True, return_predecessors=True
    )
    if graph.hub_nodes and not np.isfinite(distances[list(candidates)]).any():
        # the query itself was a hub, or hub removal disconnected it entirely -
        # fall back to the full graph rather than return an empty subgraph
        distances, predecessors = dijkstra(
            graph.adjacency, indices=query, unweighted=True, return_predecessors=True
        )

    triples: List[Triple] = []
    lengths: Dict[int, int] = {}
    for candidate in candidates:
        distance = distances[candidate]
        if not np.isfinite(distance):
            lengths[candidate] = -1  # unreachable from the query
            continue
        lengths[candidate] = int(distance)

        node = candidate
        while node != query and node >= 0:
            parent = int(predecessors[node])
            if parent < 0:
                break
            triple = graph.canonical(node, parent)
            if triple is not None:
                triples.append(triple)
            node = parent
    return triples, lengths


# ---------------------------------------------------------------------------
# step 2: rule-guided enrichment
# ---------------------------------------------------------------------------


def is_mechanism(rule: Rule) -> bool:
    """A rule counts as mechanism if any body atom is a molecular relation."""
    return any(relation in MECHANISM_RELATIONS for relation, _ in rule.body)


def rule_groundings(
    graph: RetrievalGraph,
    rule: Rule,
    candidate: int,
    query: int,
    max_groundings: int = MAX_GROUNDINGS,
) -> List[Triple]:
    """Instantiate `rule`'s body as concrete paths from `candidate` to `query`.

    Meets in the middle: expand forward from the candidate through the body's
    leading atoms and backward from the query through the trailing atom, then
    intersect. Bodies are at most length 3, so this stays cheap.
    """
    steps = [graph.matrices.get(step) for step in rule.body]
    if any(matrix is None for matrix in steps):
        return []

    paths: List[List[int]] = []
    if len(steps) == 1:
        if steps[0][candidate, query]:
            paths.append([candidate, query])

    elif len(steps) == 2:
        forward = set(steps[0][candidate].indices.tolist())
        backward = set(steps[1].T.tocsr()[query].indices.tolist())
        for middle in list(forward & backward)[:max_groundings]:
            paths.append([candidate, middle, query])

    elif len(steps) == 3:
        first = steps[0][candidate].indices
        last = steps[2].T.tocsr()[query].indices
        if len(first) and len(last):
            middle = steps[1][first][:, last]
            rows, cols = middle.nonzero()
            for row, col in list(zip(rows.tolist(), cols.tolist()))[:max_groundings]:
                paths.append([candidate, int(first[row]), int(last[col]), query])

    triples: List[Triple] = []
    for path in paths:
        for u, v in zip(path, path[1:]):
            triple = graph.canonical(u, v)
            if triple is not None:
                triples.append(triple)
    return triples


# ---------------------------------------------------------------------------
# assembling one subgraph
# ---------------------------------------------------------------------------


@dataclass
class Subgraph:
    query: int
    candidates: List[int]
    triples: List[Triple] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)


def retrieve_subgraph(
    graph: RetrievalGraph,
    query: int,
    candidates: Sequence[int],
    rules: Sequence[Rule],
    tau: int = DEFAULT_TAU,
    reserve_mechanism: float = 0.0,
    max_groundings: int = MAX_GROUNDINGS,
) -> Subgraph:
    """Build one query's subgraph, capped at `tau` distinct triples."""
    collected: List[Triple] = []
    seen: Set[Triple] = set()

    def add(triples: Iterable[Triple], budget: int) -> int:
        added = 0
        for triple in triples:
            if len(collected) >= budget:
                break
            if triple in seen:
                continue
            seen.add(triple)
            collected.append(triple)
            added += 1
        return added

    # --- step 1: connectivity ------------------------------------------------
    path_triples, path_lengths = shortest_path_triples(graph, query, candidates)
    num_paths = add(path_triples, tau)

    # --- step 2: rule-guided enrichment -------------------------------------
    mechanism_rules = [r for r in rules if is_mechanism(r)]
    other_rules = [r for r in rules if not is_mechanism(r)]

    reserved = int(round(reserve_mechanism * tau))
    num_mechanism = 0
    if reserved > 0 and mechanism_rules:
        # spend the reserved slice on mechanism rules before anything else
        budget = min(tau, len(collected) + reserved)
        for rule in mechanism_rules:
            for candidate in candidates:
                if len(collected) >= budget:
                    break
                num_mechanism += add(
                    rule_groundings(graph, rule, candidate, query, max_groundings),
                    budget,
                )

    num_rule = 0
    for rule in rules:  # already sorted by descending confidence
        if len(collected) >= tau:
            break
        for candidate in candidates:
            if len(collected) >= tau:
                break
            num_rule += add(
                rule_groundings(graph, rule, candidate, query, max_groundings), tau
            )

    relation_counts: Dict[str, int] = defaultdict(int)
    for _, relation, _ in collected:
        relation_counts[relation] += 1
    mechanism_triples = sum(
        count for relation, count in relation_counts.items()
        if relation in MECHANISM_RELATIONS
    )

    reachable = [d for d in path_lengths.values() if d >= 0]
    stats = {
        "relation_counts": dict(relation_counts),
        "num_mechanism_triples": mechanism_triples,
        "frac_mechanism": (
            round(mechanism_triples / len(collected), 3) if collected else 0.0
        ),
        "num_triples": len(collected),
        "num_from_shortest_paths": num_paths,
        "num_from_reserved_mechanism": num_mechanism,
        "num_from_rules": num_rule,
        "num_candidates": len(candidates),
        "num_candidates_reachable": len(reachable),
        "mean_path_length": round(float(np.mean(reachable)), 3) if reachable else None,
        "max_path_length": int(max(reachable)) if reachable else None,
        "saturated_tau": len(collected) >= tau,
    }
    return Subgraph(query=query, candidates=list(candidates), triples=collected, stats=stats)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _candidate_frame(out_dir: Path, relation: str, split: str) -> pd.DataFrame:
    slug = relation.replace(" ", "_").replace("/", "_")
    path = Path(out_dir) / "candidates" / f"{slug}_{split}_candidates.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run drkgc.ranker.rank first to produce candidates."
        )
    return pd.read_csv(path)


def run(
    out_dir: Path = DEFAULT_OUT_DIR,
    relations: Sequence[str] = TARGET_RELATIONS,
    splits: Sequence[str] = ("valid", "test"),
    tau: int = DEFAULT_TAU,
    reserve_mechanism: float = 0.0,
    max_groundings: int = MAX_GROUNDINGS,
    capped: bool = True,
    limit: Optional[int] = None,
    max_hub_degree: int = 0,
) -> Dict:
    out_dir = Path(out_dir)
    data = build_dataset(out_dir, capped=capped)
    graph = build_retrieval_graph(data, max_hub_degree)
    print(
        f"retrieval graph: {graph.num_entities:,} entities, "
        f"{len(graph.edge_relation):,} directed training edges"
    )

    table = data.entities.table
    name_of = dict(zip(table.global_id, table.node_name))

    subgraph_dir = out_dir / "subgraphs"
    subgraph_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict] = {}
    for relation in relations:
        rules = load_rules(out_dir, relation)
        print(
            f"\n{relation}: {len(rules)} rules "
            f"({sum(is_mechanism(r) for r in rules)} mechanism)"
        )
        for split in splits:
            frame = _candidate_frame(out_dir, relation, split)
            if limit:
                frame = frame.head(limit)

            slug = relation.replace(" ", "_").replace("/", "_")
            path = subgraph_dir / f"{slug}_{split}_subgraphs.jsonl"
            aggregate = defaultdict(list)
            composition: Dict[str, int] = defaultdict(int)

            with path.open("w", encoding="utf-8") as handle:
                for row in frame.itertuples(index=False):
                    candidates = json.loads(row.candidate_global_ids)
                    subgraph = retrieve_subgraph(
                        graph, int(row.query_global_id), candidates, rules,
                        tau, reserve_mechanism, max_groundings,
                    )
                    handle.write(
                        json.dumps(
                            {
                                "relation": relation,
                                "split": split,
                                "query_global_id": subgraph.query,
                                "query_name": name_of.get(subgraph.query, ""),
                                "gold_global_id": int(row.gold_global_id),
                                "gold_in_candidates": bool(row.gold_in_candidates),
                                "candidates": subgraph.candidates,
                                "triples": [
                                    [h, r, t] for h, r, t in subgraph.triples
                                ],
                                "triples_named": [
                                    [name_of.get(h, ""), r, name_of.get(t, "")]
                                    for h, r, t in subgraph.triples
                                ],
                                "stats": subgraph.stats,
                            }
                        )
                        + "\n"
                    )
                    for key, value in subgraph.stats.items():
                        if isinstance(value, (int, float)) and value is not None:
                            aggregate[key].append(value)
                    for relation, count in subgraph.stats["relation_counts"].items():
                        composition[relation] += count

            means = {k: round(float(np.mean(v)), 3) for k, v in aggregate.items()}
            total_triples = sum(composition.values()) or 1
            summary[f"{relation}/{split}"] = {
                "num_queries": int(len(frame)),
                "path": str(path),
                "mean": means,
                "relation_composition": {
                    rel: {"count": count, "frac": round(count / total_triples, 3)}
                    for rel, count in sorted(
                        composition.items(), key=lambda kv: -kv[1]
                    )
                },
            }
            print(
                f"  {split}: {len(frame):,} subgraphs | "
                f"mean {means.get('num_triples', 0):.1f} triples "
                f"({means.get('num_from_shortest_paths', 0):.1f} from paths) | "
                f"mean path length {means.get('mean_path_length', 0):.2f} | "
                f"{means.get('num_candidates_reachable', 0):.1f}/"
                f"{means.get('num_candidates', 0):.0f} candidates reachable "
                f"-> {path.name}"
            )
            shares = ", ".join(
                f"{rel} {count / total_triples:.0%}"
                for rel, count in sorted(composition.items(), key=lambda kv: -kv[1])
            )
            print(f"    composition: {shares}")

    report = {
        "max_hub_degree": max_hub_degree,
        "num_hub_nodes_excluded": len(graph.hub_nodes),
        "tau": tau,
        "reserve_mechanism": reserve_mechanism,
        "max_groundings": max_groundings,
        "summary": summary,
    }
    (subgraph_dir / "subgraph_report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--relations", nargs="+", default=list(TARGET_RELATIONS))
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--tau", type=int, default=DEFAULT_TAU,
                        help="maximum triples per subgraph (paper uses 100)")
    parser.add_argument("--reserve-mechanism", type=float, default=0.0,
                        help="fraction of tau reserved for mechanism rules "
                             "(0.0 = paper-faithful confidence ordering)")
    parser.add_argument("--max-groundings", type=int, default=MAX_GROUNDINGS)
    parser.add_argument("--uncapped", action="store_true")
    parser.add_argument("--max-hub-degree", type=int, default=0,
                        help="exclude nodes above this degree from path finding "
                             "(0 = off). Category hubs make vacuous 2-hop paths.")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N queries (for a quick look)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    run(
        Path(args.out_dir),
        args.relations,
        args.splits,
        args.tau,
        args.reserve_mechanism,
        args.max_groundings,
        capped=not args.uncapped,
        limit=args.limit,
        max_hub_degree=args.max_hub_degree,
    )


if __name__ == "__main__":
    main()
