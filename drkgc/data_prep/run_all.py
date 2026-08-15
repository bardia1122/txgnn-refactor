"""Run the whole of DrKGC step 1 (data preparation) end to end.

    python -m drkgc.data_prep.run_all --data-folder data --out-dir drkgc/data

Stages
------
1. load PrimeKG through txgnn (downloading / preprocessing it if needed)
2. extract the (drug, indication|contraindication, disease) triples
3. entity-safe random split of each target relation
4. build the auxiliary/context graph + leakage check
5. gene/protein degree statistics + capped auxiliary graph

Everything is idempotent: rerunning overwrites the artifacts under ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.config import (
    AUX_RELATIONS,
    DEFAULT_KG_FOLDER,
    DEFAULT_OUT_DIR,
    DEGREE_CAP_PERCENTILE,
    DISEASE_AREAS,
    HUB_NODE_TYPE,
    SEED,
    SPLIT_FRACS,
    TARGET_RELATIONS,
)
from drkgc.data_prep import build_context_graph, degree_cap, extract_triples
from drkgc.data_prep.kg_loader import load_kg_directed
from drkgc.data_prep.split_base import get_split_fn, save_split


def _banner(step: int, title: str) -> None:
    print(f"\n{'=' * 72}\n[{step}/5] {title}\n{'=' * 72}")


def _guard_output_dir(out_dir: Path, split_strategy: str, area: str | None, force: bool) -> None:
    """Refuse to mix artifacts from different splits in one output directory."""
    report_path = out_dir / "step1_report.json"
    if not report_path.exists() or force:
        return
    try:
        previous = json.loads(report_path.read_text()).get("config", {})
    except (ValueError, OSError):
        return
    same = (
        previous.get("split_strategy") == split_strategy
        and previous.get("area") == area
    )
    if not same:
        raise SystemExit(
            f"{out_dir} already holds artifacts for split "
            f"{previous.get('split_strategy')!r} / area {previous.get('area')!r}, "
            f"but this run is {split_strategy!r} / {area!r}.\n"
            f"Use a different --out-dir (e.g. drkgc/data_{area or split_strategy}) "
            "or pass --force to overwrite."
        )


def run(
    data_folder: Path = DEFAULT_KG_FOLDER,
    out_dir: Path = DEFAULT_OUT_DIR,
    split_strategy: str = "random",
    relations=TARGET_RELATIONS,
    aux_relations=AUX_RELATIONS,
    fracs=SPLIT_FRACS,
    seed: int = SEED,
    on_violation: str = "reassign",
    node_type: str = HUB_NODE_TYPE,
    cap: int | None = None,
    percentile: float = DEGREE_CAP_PERCENTILE,
    write_pyg: bool = True,
    area: str | None = None,
    force: bool = False,
) -> dict:
    data_folder, out_dir = Path(data_folder), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _guard_output_dir(out_dir, split_strategy, area, force)
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "data_folder": str(data_folder),
            "out_dir": str(out_dir),
            "split_strategy": split_strategy,
            "area": area,
            "target_relations": list(relations),
            "aux_relations": list(aux_relations),
            "fracs": list(fracs),
            "seed": seed,
            "on_violation": on_violation,
            "degree_cap_node_type": node_type,
            "degree_cap": cap,
            "degree_cap_percentile": percentile,
        },
    }

    _banner(1, "Loading PrimeKG (via pyg_implementation/txgnn)")
    df = load_kg_directed(data_folder, area=area, seed=seed)
    report["kg"] = {"num_edges": int(len(df)),
                    "num_relations": int(df.relation.nunique()),
                    "area": area}

    _banner(2, "Extracting target triples")
    triples = extract_triples.extract_all(data_folder, out_dir, relations, df=df, area=area)
    report["triples"] = {
        rel: extract_triples.triple_stats(tab) for rel, tab in triples.items()
    }

    _banner(3, f"Splitting target triples ({split_strategy})")
    split_fn = get_split_fn(split_strategy)
    report["splits"] = {}

    # the disease holdout must be shared across relations, or a disease held out
    # for indication could still be seen through contraindication
    disease_partition = None
    if split_strategy == "disease_holdout":
        from drkgc.data_prep.split_disease_area import compute_disease_partition

        disease_partition = compute_disease_partition(list(triples.values()), seed=seed)
        report["disease_partition"] = {
            k: int(len(v)) for k, v in disease_partition.items()
        }
        print(
            "shared disease partition: "
            + ", ".join(f"{k}={len(v):,}" for k, v in disease_partition.items())
        )

    for relation, table in triples.items():
        print(f"\n{relation}: {len(table):,} triples")
        result = split_fn(
            table,
            fracs=fracs,
            seed=seed,
            on_violation=on_violation,
            area=area,
            df=df,
            data_folder=data_folder,
            disease_partition=disease_partition,
        )
        save_split(result, relation, out_dir)
        report["splits"][relation] = result.stats
        for name in ("train", "valid", "test"):
            print(
                f"  {name:<5} {result.stats['sizes'][name]:>7,} "
                f"({result.stats['fractions'][name]:.2%})  "
                f"drugs={result.stats['unique_heads'][name]:,}  "
                f"diseases={result.stats['unique_tails'][name]:,}"
            )
        print(f"  entity safety: {result.stats['entity_safety']}")

    _banner(4, "Building auxiliary/context graph")
    ctx = build_context_graph.build(
        data_folder, out_dir, aux_relations, df=df, write_pyg=write_pyg, area=area
    )
    report["context_graph"] = ctx["meta"]
    report["leakage_check"] = ctx["leakage"]

    _banner(5, "Degree statistics + capping")
    report["degree_cap"] = degree_cap.run(
        out_dir,
        data_folder,
        node_type=node_type,
        cap=cap,
        percentile=percentile,
        seed=seed,
        df=df,
        write_pyg=write_pyg,
    )

    report_path = out_dir / "step1_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote summary report -> {report_path}")
    _summary(report)
    return report


def _summary(report: dict) -> None:
    print(f"\n{'=' * 72}\nSTEP 1 SUMMARY\n{'=' * 72}")
    for relation, stats in report["splits"].items():
        sizes = stats["sizes"]
        print(
            f"{relation:<18} train={sizes['train']:>7,}  valid={sizes['valid']:>6,}  "
            f"test={sizes['test']:>6,}"
        )
    leak = report["leakage_check"]
    print(
        f"leakage check      {'PASSED' if leak['passed'] else 'FAILED'} "
        f"({leak['num_overlapping_pairs']} overlapping pairs)"
    )
    dc = report["degree_cap"]
    print(
        f"degree cap         {dc['cap']} ({dc['cap_source']}); max degree "
        f"{dc['degree_stats_before']['max']:,} -> {dc['degree_stats_after']['max']:,}; "
        f"{dc['num_edges_removed']:,} edges removed"
    )
    print("\nNext: python -m drkgc.data_prep.test_sanity")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", default=str(DEFAULT_KG_FOLDER),
                        help="folder holding kg.csv / kg_directed.csv")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--split", default="random",
                        choices=["random", "disease_holdout", "disease_area"],
                        help="split strategy")
    parser.add_argument("--area", default=None, choices=list(DISEASE_AREAS),
                        help="disease area to hold out (required for --split disease_area)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an output dir built with a different split")
    parser.add_argument("--relations", nargs="+", default=list(TARGET_RELATIONS))
    parser.add_argument("--aux-relations", nargs="+", default=list(AUX_RELATIONS))
    parser.add_argument("--fracs", nargs=3, type=float, default=list(SPLIT_FRACS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--on-violation", choices=["reassign", "drop"], default="reassign")
    parser.add_argument("--node-type", default=HUB_NODE_TYPE)
    parser.add_argument("--cap", type=int, default=None)
    parser.add_argument("--percentile", type=float, default=DEGREE_CAP_PERCENTILE)
    parser.add_argument("--no-pyg", action="store_true",
                        help="write only CSV artifacts (skips torch/PyG)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.split == "disease_area" and not args.area:
        parser.error("--area is required for --split disease_area")
    if args.split != "disease_area" and args.area:
        parser.error("--area only applies to --split disease_area")

    run(
        Path(args.data_folder),
        Path(args.out_dir),
        args.split,
        args.relations,
        args.aux_relations,
        tuple(args.fracs),
        args.seed,
        args.on_violation,
        args.node_type,
        args.cap,
        args.percentile,
        write_pyg=not args.no_pyg,
        area=args.area,
        force=args.force,
    )


if __name__ == "__main__":
    main()
