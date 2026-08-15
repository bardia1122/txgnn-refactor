"""Load PrimeKG through the existing TxGNN code and resolve node names.

Nothing here reimplements data loading: the KG download and the raw-CSV ->
directed-CSV preprocessing are both delegated to
``pyg_implementation/txgnn/utils.py``.  The only thing this module adds is a
*generalised* id/name resolver: ``TxData.retrieve_id_mapping()`` only covers
`drug` and `disease`, and we also need `gene/protein` for the context graph and
for later LLM prompts.

See NOTES.md for the data-structure findings this module is based on.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

if __package__ in (None, ""):  # allow `python drkgc/data_prep/kg_loader.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drkgc.config import DEFAULT_KG_FOLDER, PYG_ROOT

# ---------------------------------------------------------------------------
# txgnn interop
# ---------------------------------------------------------------------------

_KG_FILES = {
    "kg.csv": "https://dataverse.harvard.edu/api/access/datafile/7144484",
    "node.csv": "https://dataverse.harvard.edu/api/access/datafile/7144482",
    "edges.csv": "https://dataverse.harvard.edu/api/access/datafile/7144483",
}


def add_txgnn_to_path(pyg_root: Path = PYG_ROOT) -> None:
    """Put `pyg_implementation` first on sys.path so `import txgnn` is the PyG one."""
    root = str(pyg_root)
    if sys.path[:1] != [root]:
        while root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)


def txgnn_utils():
    """Import and return `txgnn.utils` from the PyG implementation."""
    add_txgnn_to_path()
    from txgnn import utils  # noqa: WPS433 (deliberate late import)

    if "pyg_implementation" not in utils.__file__:
        raise RuntimeError(
            f"Wrong txgnn on sys.path: {utils.__file__}. Expected the copy under "
            "pyg_implementation/."
        )
    return utils


# ---------------------------------------------------------------------------
# Raw KG
# ---------------------------------------------------------------------------


def ensure_raw_kg(
    data_folder: Path = DEFAULT_KG_FOLDER,
    required: Iterable[str] = ("kg.csv",),
) -> Path:
    """Make sure the raw KG files this pipeline needs are in `data_folder`.

    Only `kg.csv` is actually read (by `preprocess_kg` and by the name
    resolver); `node.csv` / `edges.csv` are downloaded by `TxData.__init__` but
    never used here, so we do not insist on them. If a file is missing it is
    fetched from Harvard Dataverse with txgnn's own downloader.
    """
    utils = txgnn_utils()
    data_folder = Path(data_folder)
    data_folder.mkdir(parents=True, exist_ok=True)
    for fname in required:
        if (data_folder / fname).exists():
            continue
        if fname not in _KG_FILES:
            raise FileNotFoundError(f"{data_folder / fname} is missing and cannot be downloaded")
        utils.data_download_wrapper(_KG_FILES[fname], str(data_folder / fname))
    return data_folder


def load_kg_directed(
    data_folder: Path = DEFAULT_KG_FOLDER,
    download: bool = True,
) -> pd.DataFrame:
    """Return the full directed PrimeKG dataframe (`kg_directed.csv`).

    Columns: x_type, x_id, relation, y_type, y_id, x_idx, y_idx.

    * one row per undirected edge (TxGNN's `preprocess_kg` drops the mirrored
      direction), and *no* `rev_*` relations — those are only added per split by
      `reverse_rel_generation`.
    * `x_idx` / `y_idx` are contiguous per-node-type integer indices, stored as
      floats in the CSV. They are the indices used by `create_pyg_graph`.

    If the file does not exist yet it is generated with TxGNN's own
    `preprocess_kg` (a few minutes, one time only).
    """
    utils = txgnn_utils()
    data_folder = Path(data_folder)
    kg_directed = data_folder / "kg_directed.csv"

    if not kg_directed.exists():
        if download:
            ensure_raw_kg(data_folder)
        if not (data_folder / "kg.csv").exists():
            raise FileNotFoundError(
                f"{data_folder / 'kg.csv'} not found and download was disabled."
            )
        print("kg_directed.csv not found - running txgnn.utils.preprocess_kg ...")
        # `split` only matters for the disease-area variants; 'random' takes the
        # plain "read kg.csv" branch and writes kg_directed.csv next to it.
        utils.preprocess_kg(str(data_folder), split="random")

    df = pd.read_csv(kg_directed, low_memory=False)
    df["x_idx"] = df["x_idx"].astype(int)
    df["y_idx"] = df["y_idx"].astype(int)
    df["x_id"] = df["x_id"].astype(str)
    df["y_id"] = df["y_id"].astype(str)
    return df


# ---------------------------------------------------------------------------
# id / name resolution
# ---------------------------------------------------------------------------


def build_idx2id(df: pd.DataFrame, node_type: str) -> Dict[int, str]:
    """idx -> raw PrimeKG id, for one node type. Mirrors TxData.retrieve_id_mapping."""
    pairs = df[df.x_type == node_type][["x_idx", "x_id"]].drop_duplicates().values
    mapping = {int(i): str(v) for i, v in pairs}
    pairs = df[df.y_type == node_type][["y_idx", "y_id"]].drop_duplicates().values
    mapping.update({int(i): str(v) for i, v in pairs})
    return mapping


def build_id2name(
    data_folder: Path = DEFAULT_KG_FOLDER,
    node_types: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, str]]:
    """{node_type: {raw id -> human readable name}} read from the raw kg.csv.

    Generalisation of `TxData.retrieve_id_mapping`, which hardcodes drug/disease.
    Ids go through txgnn's `convert2str` so they match `kg_directed.csv`.
    """
    utils = txgnn_utils()
    raw = pd.read_csv(Path(data_folder) / "kg.csv", low_memory=False)
    raw["x_id"] = raw.x_id.apply(utils.convert2str)
    raw["y_id"] = raw.y_id.apply(utils.convert2str)

    if node_types is None:
        node_types = sorted(set(raw.x_type.unique()) | set(raw.y_type.unique()))

    out: Dict[str, Dict[str, str]] = {}
    for ntype in node_types:
        m = dict(raw[raw.x_type == ntype][["x_id", "x_name"]].drop_duplicates().values)
        m.update(dict(raw[raw.y_type == ntype][["y_id", "y_name"]].drop_duplicates().values))
        out[ntype] = {str(k): str(v) for k, v in m.items()}
    return out


def resolve_name(raw_id: str, id2name: Dict[str, str], default: str = "") -> str:
    """Resolve one raw id to a name.

    Merged PrimeKG disease nodes carry underscore-joined ids such as
    `'8450_15304'`; kg.csv gives those a name of their own, so the direct lookup
    normally succeeds. The fallback joins the names of the parts, trying both
    `'8450'` and `'8450.0'` because `convert2str` only floats *unmerged* ids.
    """
    raw_id = str(raw_id)
    if raw_id in id2name:
        return id2name[raw_id]
    if "_" in raw_id:  # merged node: join the parts we can resolve
        parts = []
        for piece in raw_id.split("_"):
            for key in (piece, _as_float_str(piece)):
                if key and key in id2name:
                    parts.append(id2name[key])
                    break
        if parts:
            return " | ".join(dict.fromkeys(parts))
    return default


def _as_float_str(value: str) -> str:
    try:
        return str(float(value))
    except (TypeError, ValueError):
        return ""


def build_entity_table(
    df: pd.DataFrame,
    data_folder: Path = DEFAULT_KG_FOLDER,
    node_types: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """One row per (node_type, node_idx) with its raw id and human readable name."""
    if node_types is None:
        node_types = sorted(set(df.x_type.unique()) | set(df.y_type.unique()))
    node_types = list(node_types)

    id2name_all = build_id2name(data_folder, node_types)
    rows = []
    for ntype in node_types:
        idx2id = build_idx2id(df, ntype)
        id2name = id2name_all.get(ntype, {})
        for idx, raw_id in sorted(idx2id.items()):
            rows.append(
                {
                    "node_type": ntype,
                    "node_idx": idx,
                    "node_id": raw_id,
                    "node_name": resolve_name(raw_id, id2name),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# inventory / relation lookup
# ---------------------------------------------------------------------------


def relation_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Every (x_type, relation, y_type) present in the directed KG + edge counts."""
    inv = (
        df.groupby(["x_type", "relation", "y_type"])
        .size()
        .reset_index(name="num_edges")
        .sort_values(["relation", "num_edges"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return inv


def node_type_sizes(df: pd.DataFrame) -> Dict[str, int]:
    """num_nodes per node type, using the same rule as txgnn.utils.create_pyg_graph
    (max index seen anywhere in the full KG, + 1)."""
    sizes: Dict[str, int] = {}
    for type_col, idx_col in (("x_type", "x_idx"), ("y_type", "y_idx")):
        for ntype, max_idx in df.groupby(type_col)[idx_col].max().items():
            sizes[ntype] = max(sizes.get(ntype, -1), int(max_idx))
    return {k: v + 1 for k, v in sizes.items()}


def canonical_edge_type(df: pd.DataFrame, relation: str):
    """Return the (x_type, relation, y_type) triple actually stored for `relation`.

    `preprocess_kg` keeps only one of the two mirrored directions, and which one
    it keeps depends on the row order in kg.csv - so never hardcode it.
    """
    sub = df[df.relation == relation]
    if len(sub) == 0:
        raise KeyError(f"relation {relation!r} not present in the KG")
    counts = sub.groupby(["x_type", "y_type"]).size().sort_values(ascending=False)
    (x_type, y_type) = counts.index[0]
    if len(counts) > 1:
        print(
            f"  ! relation {relation!r} appears with several type signatures "
            f"{list(counts.index)}; using the most frequent one."
        )
    return (str(x_type), str(relation), str(y_type))


if __name__ == "__main__":  # quick manual inspection
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", default=str(DEFAULT_KG_FOLDER))
    args = parser.parse_args()

    kg = load_kg_directed(Path(args.data_folder))
    print(f"kg_directed.csv: {len(kg):,} edges")
    print("\nnode types:")
    for ntype, n in sorted(node_type_sizes(kg).items()):
        print(f"  {ntype:<25} {n:>8,}")
    print("\nrelations:")
    print(relation_inventory(kg).to_string(index=False))
