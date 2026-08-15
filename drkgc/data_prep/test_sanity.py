"""Assert-based sanity checks over the step-1 artifacts.

    python -m drkgc.data_prep.test_sanity --out-dir drkgc/data

Verifies:

* the triple tables are well formed (drug -> disease, deduplicated, named);
* the split is a true partition of the triples and has **no entity-safety
  violations** (every drug/disease in valid/test also occurs in train);
* the auxiliary graph has **no leakage** of target relations;
* degree capping actually lowered the max gene/protein degree to the cap, only
  ever removed edges, and is reproducible for a fixed seed.

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.config import (
    ALL_DD_RELATIONS,
    CONTEXT_DIR,
    DEFAULT_OUT_DIR,
    DISEASE_TYPE,
    DRUG_TYPE,
    HUB_NODE_TYPE,
    TARGET_RELATIONS,
    TRIPLES_DIR,
)
from drkgc.data_prep.build_context_graph import leakage_check
from drkgc.data_prep.degree_cap import cap_degrees, compute_degrees
from drkgc.data_prep.extract_triples import load_triples
from drkgc.data_prep.split_base import load_split

_FAILURES: List[str] = []


def check(name: str, fn: Callable[[], str | None]) -> None:
    """Run one check; record the failure instead of aborting the whole run."""
    try:
        detail = fn()
    except AssertionError as exc:
        _FAILURES.append(name)
        print(f"  FAIL  {name}\n        {exc}")
    except Exception as exc:  # noqa: BLE001 - surface anything unexpected
        _FAILURES.append(name)
        print(f"  ERROR {name}\n        {type(exc).__name__}: {exc}")
    else:
        print(f"  ok    {name}" + (f" - {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_triples(out_dir: Path, relations: Sequence[str]) -> None:
    print("\n[triples]")
    for relation in relations:

        def _run(relation=relation) -> str:
            triples = load_triples(out_dir, relation)
            assert len(triples) > 0, f"{relation}: empty triple table"
            assert (triples.head_type == DRUG_TYPE).all(), "head_type is not 'drug'"
            assert (triples.tail_type == DISEASE_TYPE).all(), "tail_type is not 'disease'"
            dupes = triples.duplicated(subset=["head_idx", "tail_idx"]).sum()
            assert dupes == 0, f"{dupes} duplicated (drug, disease) pairs"
            assert triples.head_idx.min() >= 0 and triples.tail_idx.min() >= 0, (
                "negative node indices"
            )
            named = int((triples.head_name != "").sum())
            return (
                f"{len(triples):,} triples, {triples.head_idx.nunique():,} drugs, "
                f"{triples.tail_idx.nunique():,} diseases, {named:,} named heads"
            )

        check(f"{relation}: triple table well formed", _run)


def check_splits(out_dir: Path, relations: Sequence[str]) -> None:
    print("\n[splits]")
    for relation in relations:

        def _entity_safety(relation=relation) -> str:
            result = load_split(relation, out_dir)
            train_heads = set(result.train.head_idx)
            train_tails = set(result.train.tail_idx)
            for name in ("valid", "test"):
                part = getattr(result, name)
                unseen_h = set(part.head_idx) - train_heads
                unseen_t = set(part.tail_idx) - train_tails
                assert not unseen_h, f"{name}: {len(unseen_h)} drugs never seen in train"
                assert not unseen_t, f"{name}: {len(unseen_t)} diseases never seen in train"
            return (
                f"train={len(result.train):,} valid={len(result.valid):,} "
                f"test={len(result.test):,}"
            )

        def _partition(relation=relation) -> str:
            result = load_split(relation, out_dir)
            triples = load_triples(out_dir, relation)
            keys = {
                name: set(
                    zip(getattr(result, name).head_idx, getattr(result, name).tail_idx)
                )
                for name in ("train", "valid", "test")
            }
            for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
                overlap = keys[a] & keys[b]
                assert not overlap, f"{len(overlap)} triples shared between {a} and {b}"
            union = keys["train"] | keys["valid"] | keys["test"]
            original = set(zip(triples.head_idx, triples.tail_idx))
            missing = original - union
            extra = union - original
            assert not extra, f"{len(extra)} triples in the split that are not in the source"

            safety = result.stats.get("entity_safety", {})
            policy = safety.get("policy", "reassign")
            if policy == "reassign":
                assert not missing, (
                    f"{len(missing)} source triples missing from the split "
                    "(on_violation='reassign' must not lose any triple)"
                )
                return f"{len(union):,} triples partitioned, no overlap, nothing lost"

            expected_dropped = sum(
                safety.get(name, {}).get("num_dropped", 0) for name in ("valid", "test")
            )
            assert len(missing) == expected_dropped, (
                f"{len(missing)} triples missing but {expected_dropped} were reported "
                "as dropped by the entity-safety constraint"
            )
            return (
                f"{len(union):,} triples partitioned, no overlap, "
                f"{len(missing):,} dropped as reported"
            )

        check(f"{relation}: entity-safe split", _entity_safety)
        check(f"{relation}: split is a partition of the triples", _partition)


def check_leakage(out_dir: Path, relations: Sequence[str]) -> None:
    print("\n[leakage]")
    ctx_dir = out_dir / CONTEXT_DIR
    targets = pd.read_csv(
        out_dir / TRIPLES_DIR / "target_triples.csv",
        dtype={"head_id": str, "tail_id": str},
        keep_default_na=False,
    )

    for fname, label in (
        ("context_edges.csv", "uncapped"),
        ("context_edges_capped.csv", "capped"),
    ):

        def _run(fname=fname) -> str:
            edges = pd.read_csv(
                ctx_dir / fname, dtype={"head_id": str, "tail_id": str},
                keep_default_na=False,
            )
            report = leakage_check(edges, targets)
            assert not report["forbidden_relations_present"], (
                f"target relations present in the auxiliary graph: "
                f"{report['forbidden_relations_present']}"
            )
            assert report["num_overlapping_pairs"] == 0, (
                f"{report['num_overlapping_pairs']} target (drug, disease) pairs also "
                f"exist as auxiliary edges, e.g. {report['overlapping_examples'][:3]}"
            )
            assert not (set(edges.relation) & set(ALL_DD_RELATIONS)), (
                "drug-disease relation leaked into the auxiliary edge list"
            )
            return (
                f"{len(edges):,} auxiliary edges vs {len(targets):,} target triples, "
                "0 overlaps"
            )

        check(f"auxiliary graph ({label}) is leakage free", _run)

    def _stored_report() -> str:
        report = json.loads((ctx_dir / "leakage_check.json").read_text())
        assert report["passed"], f"stored leakage_check.json says FAILED: {report}"
        return "stored leakage_check.json reports passed=true"

    check("stored leakage report agrees", _stored_report)


def check_degree_cap(out_dir: Path, node_type: str = HUB_NODE_TYPE) -> None:
    print("\n[degree cap]")
    ctx_dir = out_dir / CONTEXT_DIR
    info = json.loads((ctx_dir / "degree_stats.json").read_text())
    read = lambda name: pd.read_csv(  # noqa: E731
        ctx_dir / name, dtype={"head_id": str, "tail_id": str}, keep_default_na=False
    )

    def _reduced() -> str:
        edges = read("context_edges.csv")
        capped = read("context_edges_capped.csv")
        cap = int(info["cap"])
        deg_before = compute_degrees(edges, node_type)
        deg_after = compute_degrees(capped, node_type)
        assert deg_after.max() <= cap, (
            f"max degree after capping is {deg_after.max()}, above the cap {cap}"
        )
        if deg_before.max() > cap:
            assert deg_after.max() < deg_before.max(), (
                "capping did not reduce the maximum degree"
            )
            assert len(capped) < len(edges), "capping removed no edges"
        assert (deg_after <= deg_before.reindex(deg_after.index)).all(), (
            "some node gained degree during capping"
        )
        return (
            f"cap={cap}; max degree {int(deg_before.max()):,} -> "
            f"{int(deg_after.max()):,}; {len(edges) - len(capped):,} edges removed"
        )

    def _subset() -> str:
        edges = read("context_edges.csv")
        capped = read("context_edges_capped.csv")
        key = ["relation", "head_type", "head_idx", "tail_type", "tail_idx"]
        base = set(map(tuple, edges[key].values.tolist()))
        sub = set(map(tuple, capped[key].values.tolist()))
        extra = sub - base
        assert not extra, f"{len(extra)} capped edges are not in the uncapped graph"
        return f"{len(sub):,} capped edges are a subset of the {len(base):,} uncapped ones"

    def _deterministic() -> str:
        edges = read("context_edges.csv")
        capped = read("context_edges_capped.csv")
        again, _ = cap_degrees(
            edges,
            node_type=node_type,
            cap=int(info["cap"]),
            seed=int(info["seed"]),
        )
        assert len(again) == len(capped), (
            f"rerun produced {len(again):,} edges, artifact has {len(capped):,}"
        )
        key = ["relation", "head_type", "head_idx", "tail_type", "tail_idx"]
        assert set(map(tuple, again[key].values.tolist())) == set(
            map(tuple, capped[key].values.tolist())
        ), "rerunning cap_degrees with the same seed produced a different edge set"
        return "same seed reproduces the capped graph exactly"

    check("capping reduced max gene/protein degree to the cap", _reduced)
    check("capped graph is a subset of the uncapped graph", _subset)
    check("capping is reproducible", _deterministic)


def check_pyg_graphs(out_dir: Path) -> None:
    print("\n[pyg artifacts]")
    ctx_dir = out_dir / CONTEXT_DIR
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  skip  torch not installed - skipping HeteroData checks")
        return

    from drkgc.data_prep.build_context_graph import load_context_graph

    for pt_name, csv_name in (
        ("context_graph.pt", "context_edges.csv"),
        ("context_graph_capped.pt", "context_edges_capped.csv"),
    ):

        def _run(pt_name=pt_name, csv_name=csv_name) -> str:
            path = ctx_dir / pt_name
            assert path.exists(), f"{path} missing (was --no-pyg used?)"
            data = load_context_graph(path)
            edges = pd.read_csv(ctx_dir / csv_name, keep_default_na=False)
            total = sum(data[et].edge_index.shape[1] for et in data.edge_types)
            assert total == len(edges), (
                f"{pt_name} has {total:,} edges, {csv_name} has {len(edges):,}"
            )
            assert not (
                {et[1] for et in data.edge_types} & set(ALL_DD_RELATIONS)
            ), "drug-disease relation present in the saved HeteroData"
            return f"{len(data.edge_types)} edge types, {total:,} edges"

        check(f"{pt_name} matches {csv_name}", _run)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--relations", nargs="+", default=list(TARGET_RELATIONS))
    parser.add_argument("--node-type", default=HUB_NODE_TYPE)
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.out_dir)
    print(f"Sanity checks on {out_dir}")

    check_triples(out_dir, args.relations)
    check_splits(out_dir, args.relations)
    check_leakage(out_dir, args.relations)
    check_degree_cap(out_dir, args.node_type)
    check_pyg_graphs(out_dir)

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} CHECK(S) FAILED: {', '.join(_FAILURES)}")
        return 1
    print("All sanity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
