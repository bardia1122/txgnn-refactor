"""Central configuration for the DrKGC pipeline.

Paths are resolved relative to the repository root so the package works no
matter which directory the scripts are launched from.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

#: repository root (the folder that contains `pyg_implementation/` and `drkgc/`)
REPO_ROOT = Path(__file__).resolve().parents[1]

#: the PyG TxGNN refactor — added to sys.path so `import txgnn` resolves here
#: (and NOT to `dgl_implementation/txgnn`).
PYG_ROOT = REPO_ROOT / "pyg_implementation"

def _detect_kg_folder() -> Path:
    """Folder holding kg.csv / kg_directed.csv.

    `TxData` is normally pointed at `REPO_ROOT/data`, but the official PrimeKG
    release unpacks into `REPO_ROOT/data/kg`. Prefer whichever actually holds
    the data; `$TXGNN_DATA_FOLDER` overrides both.
    """
    env = os.environ.get("TXGNN_DATA_FOLDER")
    if env:
        return Path(env)
    candidates = (REPO_ROOT / "data" / "kg", REPO_ROOT / "data")
    for marker in ("kg_directed.csv", "kg.csv"):
        for candidate in candidates:
            if (candidate / marker).exists():
                return candidate
    return REPO_ROOT / "data"


#: where kg.csv lives and where kg_directed.csv will be written
DEFAULT_KG_FOLDER = _detect_kg_folder()


def _detect_disease_files() -> Path:
    """Folder holding the per-disease-area node lists (`<area>.csv`).

    `process_disease_area_split` expects `<data_folder>/disease_files`, but in
    this repo they live in `REPO_ROOT/data/disease_files` while the KG sits in
    `REPO_ROOT/data/kg`.
    """
    for candidate in (DEFAULT_KG_FOLDER / "disease_files", REPO_ROOT / "data" / "disease_files"):
        if candidate.is_dir():
            return candidate
    return REPO_ROOT / "data" / "disease_files"


DISEASE_FILES_DIR = _detect_disease_files()

#: root for every artifact this package produces
DRKGC_ROOT = REPO_ROOT / "drkgc"
DEFAULT_OUT_DIR = DRKGC_ROOT / "data"

TRIPLES_DIR = "triples"
SPLITS_DIR = "splits"
CONTEXT_DIR = "context_graph"

# ---------------------------------------------------------------------------
# Data constants (verified against pyg_implementation/txgnn — see NOTES.md)
# ---------------------------------------------------------------------------

#: drug<->disease relations that are the *prediction target*.
#: PrimeKG also carries 'off-label use'; it is deliberately out of scope for
#: this step (see NOTES.md).
TARGET_RELATIONS = ("indication", "contraindication")

#: every drug<->disease relation TxGNN treats as a "dd" relation. Used only for
#: the leakage check, so that an accidental 'off-label use' edge in the
#: auxiliary graph would also be caught.
ALL_DD_RELATIONS = ("indication", "contraindication", "off-label use")

#: relations that make up the auxiliary / context graph.
#: drug-target, gene-disease association, protein-protein interaction.
AUX_RELATIONS = ("drug_protein", "disease_protein", "protein_protein")

#: node type whose degree we cap (PrimeKG spells it with a slash)
HUB_NODE_TYPE = "gene/protein"

DRUG_TYPE = "drug"
DISEASE_TYPE = "disease"

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

SEED = 42

#: train / valid / test fractions for the entity-safe random split
SPLIT_FRACS = (0.90, 0.05, 0.05)

#: fractions over *diseases* for the zero-shot disease-holdout split.
#: Same numbers TxGNN uses for its 'complex_disease' split (`utils.py:407`).
ZERO_SHOT_FRACS = (0.83125, 0.11875, 0.05)

#: share of the disease-area training pool held out for validation.
#: TxGNN uses random_fold(train_val, [0.875, 0.125, 0.0]) — `utils.py:395`.
AREA_VALID_FRAC = 0.125

#: the nine disease areas shipped in data/disease_files/ (TxData.prepare_split)
DISEASE_AREAS = (
    "adrenal_gland",
    "anemia",
    "autoimmune",
    "cardiovascular",
    "cell_proliferation",
    "diabetes",
    "mental_health",
    "metabolic_disorder",
    "neurodigenerative",
)

#: default degree cap = this percentile of the gene/protein degree distribution
DEGREE_CAP_PERCENTILE = 95.0
