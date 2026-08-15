"""Extract (drug, indication|contraindication, disease) triples from PrimeKG.

Output: one CSV per target relation under ``<out>/triples/`` holding *both* the
numeric per-node-type index (needed for graph ops) and the resolved human
readable name (needed for LLM prompts).

Run standalone::

    python -m drkgc.data_prep.extract_triples --data-folder data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.config import (
    DEFAULT_KG_FOLDER,
    DEFAULT_OUT_DIR,
    DISEASE_TYPE,
    DRUG_TYPE,
    TARGET_RELATIONS,
    TRIPLES_DIR,
)
from drkgc.data_prep.kg_loader import (
    build_id2name,
    build_idx2id,
    load_kg_directed,
    node_type_sizes,
    relation_inventory,
    resolve_name,
)

TRIPLE_COLUMNS = [
    "relation",
    "head_type",
    "head_idx",
    "head_id",
    "head_name",
    "tail_type",
    "tail_idx",
    "tail_id",
    "tail_name",
]


def extract_relation_triples(
    df: pd.DataFrame,
    relation: str,
    id2name: Dict[str, Dict[str, str]],
    head_type: str = DRUG_TYPE,
    tail_type: str = DISEASE_TYPE,
) -> pd.DataFrame:
    """All triples of `relation`, oriented head_type -> tail_type.

    `preprocess_kg` keeps only one of the two mirrored directions of every
    relation, and which one survives depends on the row order in kg.csv. We
    therefore accept either orientation and flip when needed.
    """
    sub = df[df.relation == relation]
    if len(sub) == 0:
        raise KeyError(f"relation {relation!r} not found in the KG")

    forward = sub[(sub.x_type == head_type) & (sub.y_type == tail_type)]
    backward = sub[(sub.x_type == tail_type) & (sub.y_type == head_type)]
    if len(backward):
        print(
            f"  note: {len(backward):,}/{len(sub):,} {relation!r} rows are stored as "
            f"{tail_type}->{head_type}; flipping them."
        )
        flipped = backward.rename(
            columns={
                "x_type": "y_type",
                "x_id": "y_id",
                "x_idx": "y_idx",
                "y_type": "x_type",
                "y_id": "x_id",
                "y_idx": "x_idx",
            }
        )
        forward = pd.concat([forward, flipped], ignore_index=True)

    dropped = len(sub) - len(forward)
    if dropped:
        print(
            f"  warning: {dropped:,} {relation!r} rows have unexpected node types "
            "and were skipped."
        )

    out = pd.DataFrame(
        {
            "relation": relation,
            "head_type": head_type,
            "head_idx": forward["x_idx"].astype(int).values,
            "head_id": forward["x_id"].astype(str).values,
            "tail_type": tail_type,
            "tail_idx": forward["y_idx"].astype(int).values,
            "tail_id": forward["y_id"].astype(str).values,
        }
    )
    out = out.drop_duplicates(subset=["head_idx", "tail_idx"]).reset_index(drop=True)

    head_names = id2name.get(head_type, {})
    tail_names = id2name.get(tail_type, {})
    out["head_name"] = [resolve_name(i, head_names) for i in out["head_id"]]
    out["tail_name"] = [resolve_name(i, tail_names) for i in out["tail_id"]]

    n_missing_head = int((out["head_name"] == "").sum())
    n_missing_tail = int((out["tail_name"] == "").sum())
    if n_missing_head or n_missing_tail:
        print(
            f"  warning: unresolved names - {n_missing_head} heads, "
            f"{n_missing_tail} tails (left as empty strings)."
        )
    return out[TRIPLE_COLUMNS]


def triple_stats(triples: pd.DataFrame) -> Dict[str, int]:
    return {
        "num_triples": int(len(triples)),
        "num_unique_heads": int(triples.head_idx.nunique()),
        "num_unique_tails": int(triples.tail_idx.nunique()),
        "num_unnamed_heads": int((triples.head_name == "").sum()),
        "num_unnamed_tails": int((triples.tail_name == "").sum()),
    }


def extract_all(
    data_folder: Path = DEFAULT_KG_FOLDER,
    out_dir: Path = DEFAULT_OUT_DIR,
    relations: Sequence[str] = TARGET_RELATIONS,
    df: pd.DataFrame | None = None,
) -> Dict[str, pd.DataFrame]:
    """Extract every target relation and write the CSVs. Returns {relation: df}."""
    data_folder, out_dir = Path(data_folder), Path(out_dir)
    triples_dir = out_dir / TRIPLES_DIR
    triples_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = load_kg_directed(data_folder)
    print(f"Loaded directed KG: {len(df):,} edges")

    # a small inventory dump makes the NOTES/README reproducible on any machine
    inventory = {
        "num_edges": int(len(df)),
        "node_type_sizes": node_type_sizes(df),
        "relations": relation_inventory(df).to_dict(orient="records"),
    }
    (out_dir / "kg_inventory.json").write_text(json.dumps(inventory, indent=2))

    id2name = build_id2name(data_folder, node_types=[DRUG_TYPE, DISEASE_TYPE])

    out: Dict[str, pd.DataFrame] = {}
    stats = {}
    for relation in relations:
        print(f"\nExtracting {relation!r} ...")
        triples = extract_relation_triples(df, relation, id2name)
        path = triples_dir / f"{_slug(relation)}_triples.csv"
        triples.to_csv(path, index=False)
        out[relation] = triples
        stats[relation] = triple_stats(triples)
        print(f"  {stats[relation]['num_triples']:,} triples -> {path}")
        print(
            f"  unique drugs: {stats[relation]['num_unique_heads']:,} | "
            f"unique diseases: {stats[relation]['num_unique_tails']:,}"
        )

    combined = pd.concat(out.values(), ignore_index=True)
    combined.to_csv(triples_dir / "target_triples.csv", index=False)
    (triples_dir / "triple_stats.json").write_text(json.dumps(stats, indent=2))
    return out


def load_triples(out_dir: Path = DEFAULT_OUT_DIR, relation: str = "indication") -> pd.DataFrame:
    """Read back a triple table written by `extract_all`."""
    path = Path(out_dir) / TRIPLES_DIR / f"{_slug(relation)}_triples.csv"
    return pd.read_csv(path, dtype={"head_id": str, "tail_id": str}, keep_default_na=False)


def _slug(relation: str) -> str:
    return relation.replace(" ", "_").replace("/", "_")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", default=str(DEFAULT_KG_FOLDER))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--relations", nargs="+", default=list(TARGET_RELATIONS))
    args = parser.parse_args(list(argv) if argv is not None else None)

    extract_all(Path(args.data_folder), Path(args.out_dir), args.relations)


if __name__ == "__main__":
    main()
