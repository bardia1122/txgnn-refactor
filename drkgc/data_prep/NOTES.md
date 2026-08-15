# NOTES — what actually exists in `pyg_implementation/`

Findings from reading the repo *before* writing the extraction code, so the
DrKGC step-1 pipeline is grounded in the real data structures. Line numbers
refer to the state of the repo at the time of writing (commit `c244ad5`).

---

## 1. How the graph is loaded and represented

| Question | Answer |
|---|---|
| Graph object | PyG `HeteroData`, built by `create_pyg_graph(df_train, df)` — `pyg_implementation/txgnn/utils.py:1086` |
| Where it is constructed | `TxData.prepare_split()` — `pyg_implementation/txgnn/TxData.py:107` (`self.G = g`) |
| Is it cached? | **No.** The graph is rebuilt in memory on every run. What *is* cached on disk are CSVs (below). |
| Node features | `initialize_node_embedding(g, n_inp)` (`utils.py:1124`) writes a Xavier-uniform `nn.Parameter` to `data[ntype].inp`. Nothing else is stored on the graph except `edge_index` and `edge_id`. |
| `num_nodes` rule | `int(max index for that node type over the **full** KG) + 1` (`utils.py:1101-1112`), i.e. node index spaces are per node type and stay stable across splits. `effect/phenotype` gets a hardcoded floor of `0.0`. |

### The CSV pipeline (this is the real source of truth)

```
<data_folder>/kg.csv            raw PrimeKG (downloaded from Harvard Dataverse by TxData.__init__)
<data_folder>/node.csv          "
<data_folder>/edges.csv         "
        |  txgnn.utils.preprocess_kg()          utils.py:60
        v
<data_folder>/kg_directed.csv   x_type, x_id, relation, y_type, y_id, x_idx, y_idx
        |  txgnn.utils.create_split()           utils.py:400
        v
<data_folder>/<split>_<seed>/{train,valid,test}.csv
```

Facts that matter for us:

* `preprocess_kg` keeps **one direction per relation only** — for a heterogeneous
  relation it keeps the rows whose `x_type` equals the `x_type` of that
  relation's *first* row (`utils.py:117-118`). So which endpoint ends up in
  `x_*` is data-order dependent and **must not be hardcoded**. Our loader
  discovers it (`kg_loader.canonical_edge_type`) and `extract_triples` flips
  rows when needed.
* The `rev_*` relations you see all over `model.py` do **not** exist in
  `kg_directed.csv`. They are added per split by `reverse_rel_generation`
  (`utils.py:910`), which is called inside `create_split`. Our step reads
  `kg_directed.csv`, so we work with single-direction edges.
* `x_idx` / `y_idx` are contiguous per-node-type integer indices assigned in
  `preprocess_kg` (`utils.py:128-135`); they are stored as floats in the CSV.
* `x_id` / `y_id` are stringified through `convert2str` (`utils.py:1039`), which
  renders numeric ids as `'1234.0'`. Merged disease nodes carry underscore-joined
  ids such as `'1234.0_5678.0'` (handled at `utils.py:1064-1075`).

## 2. Edge type names (verified, not assumed)

Drug–disease relations, i.e. our prediction targets — `utils.py:168`, `195`,
`599-604`, `model.py:41-46`:

| Relation string | Canonical PyG edge type used by TxGNN |
|---|---|
| `indication` | `('drug', 'indication', 'disease')` |
| `contraindication` | `('drug', 'contraindication', 'disease')` |
| `off-label use` | `('drug', 'off-label use', 'disease')` |

> **Discrepancy vs. the task description:** PrimeKG has *three* drug–disease
> relations, not two. `off-label use` is out of scope for this step (we only
> extract `indication` and `contraindication`), but it *is* included in the
> leakage check's forbidden-relation list so it can never sneak into the
> auxiliary graph either.

Auxiliary / mechanism relations — `utils.py:987`, `995`, `model.py:85`, `108`:

| Task description | Actual relation string | Notes |
|---|---|---|
| drug target | `drug_protein` | drug ↔ `gene/protein` |
| gene–disease association | `disease_protein` | TxGNN reaches proteins from a disease via `rev_disease_protein` (`model.py:108`, `utils.py:987`), which implies the stored direction is `gene/protein -> disease`. **Confirmed against the data** (§7). We still do not hardcode it — the real signature is discovered at runtime and dumped to `data/kg_inventory.json`. |
| protein–protein interaction | `protein_protein` | homogeneous, `gene/protein` on both ends (`utils.py:995`) |

Node type spelling: **`gene/protein`** (with the slash) — `utils.py:985`,
`model.py:108`. Other node types referenced in the code include `drug`,
`disease`, `effect/phenotype` and `exposure`; the complete observed inventory is
written to `drkgc/data/kg_inventory.json` when the pipeline runs.

## 3. Node id ↔ human-readable name mapping

`TxData.retrieve_id_mapping()` — `TxData.py:117` — returns

```python
{'idx2id_drug', 'idx2id_disease',    # x_idx -> x_id, from kg_directed.csv
 'id2name_drug', 'id2name_disease'}  # x_id  -> x_name, re-read from raw kg.csv
```

Limitations that made us wrap rather than call it directly:

1. it only covers `drug` and `disease` — we also need `gene/protein` names for
   subgraph retrieval and prompts later;
2. it requires `prepare_split()` to have been run first (it reads `self.df`);
3. it mutates `self.df` in place (applies `convert2str` to the id columns).

`drkgc/data_prep/kg_loader.py` reimplements exactly the same two-step lookup
(`idx -> raw id` from `kg_directed.csv`, `raw id -> name` from `kg.csv`'s
`x_name`/`y_name`) for **any** node type, and adds merged-id handling.

Merged disease nodes carry underscore-joined ids — real examples from the data:
`'8450_15304'` and a 18-way merge
`'11764_11658_13625_8199_...'`. Verified: every row of kg.csv has a non-empty
`x_name`/`y_name`, **including the merged nodes**, so the direct lookup
resolves them and no name is left blank. The part-wise fallback in
`resolve_name` (which mirrors `process_disease_area_split`, `utils.py:1064-1075`)
tries both `'8450'` and `'8450.0'`, because `convert2str` floats unmerged
numeric ids but leaves merged ones untouched.

## 4. Where TxGNN's zero-shot / disease-area split logic lives

Noted for the follow-up task; **not used in this step.**

| What | Where |
|---|---|
| Split dispatcher | `utils.py:375` `create_fold()` / `utils.py:400` `create_split()` |
| Zero-shot "complex disease" split (holds whole diseases out of all drug–disease relations) | `utils.py:194` `complex_disease_fold()` |
| Single-disease holdout | `utils.py:162` `disease_eval_fold()` |
| Disease-**area** splits (ontology driven) | `utils.py:61-98` (the disease-area branch of `preprocess_kg`) + `pyg_implementation/txgnn/data_splits/datasplit.py` `DataSplitter.get_test_kg_for_disease()` (uses `HumanDO.obo`) |
| Post-filtering of the disease-area test set | `utils.py:1057` `process_disease_area_split()` |
| The nine disease-area node lists | `data/disease_files/{adrenal_gland,anemia,autoimmune,cardiovascular,cell_proliferation,diabetes,mental_health,metabolic_disorder,neurodigenerative}.csv` |
| Mirror copy (DGL original) | `dgl_implementation/txgnn/utils.py`, `dgl_implementation/txgnn/data_splits/` |

When we swap the split later, the new `get_disease_area_split(...)` only has to
return a `split_base.SplitResult` and register itself with
`@register_split('disease_area')`; `run_all.py --split disease_area` then picks
it up with no other change.

## 5. Output-format conventions we follow

* The repo stores every tabular artifact as **CSV via pandas** (splits, disease
  files, results). We do the same for triples, splits and the context edge list.
* There is no existing "data output directory" pattern other than *"write next
  to the input KG"* (`TxData` writes `kg_directed.csv` and `<split>_<seed>/`
  into its `data_folder`). We deliberately do **not** write into that folder;
  DrKGC artifacts go to `drkgc/data/` so the two pipelines never collide.
* The PyG graph object is saved with `torch.save` — `PYG_REFACTOR.md:88`
  explicitly replaces DGL's `save_graphs` with `torch.save`.

## 6. Verified against the real PrimeKG (`data/kg/kg.csv`)

The official PrimeKG release is present locally (`data/kg/`: `kg.csv` 981 MB,
`nodes.csv`, `edges.csv`, `kg_raw.csv`, `kg_giant.csv`, `kg_grouped.csv`, ...).
Note the release names the node table **`nodes.csv`**, while `TxData.__init__`
downloads it as `node.csv` — irrelevant here, since only `kg.csv` is ever read.

`kg.csv` was streamed once (8,100,498 raw rows) while emulating
`preprocess_kg`'s direction-dropping rule, to check the assumptions above
rather than trusting them. Results for the relations we touch:

| Relation | rows in kg.csv | kept by `preprocess_kg` | kept orientation |
|---|---|---|---|
| `indication` | 18,776 | **9,388** | `drug -> disease` |
| `contraindication` | 61,350 | **30,675** | `drug -> disease` |
| `off-label use` | 5,136 | 2,568 | `drug -> disease` (out of scope) |
| `drug_protein` | 51,306 | **25,653** | `drug -> gene/protein` |
| `disease_protein` | 160,822 | **80,411** | `gene/protein -> disease` |
| `protein_protein` | 642,150 | **321,075** | `gene/protein -> gene/protein` |

* Every drug–disease and drug/disease–protein relation appears in kg.csv in
  **both** orientations in equal numbers, and `preprocess_kg` keeps exactly one
  of them — so the "first row wins" rule really is what decides orientation.
  The predicted `gene/protein -> disease` direction for `disease_protein` is
  confirmed.
* `indication` gives **9,388** triples — exactly the count the DrKGC paper
  reports for its PrimeKG subset (appendix A.1), which is a good cross-check
  that we are extracting the same thing they did. Their 8,388/500/500 split
  differs from our 90/5/5 (≈8,449/469/470 before the entity-safety pass).
* Entity counts on the extracted triples: `indication` 1,801 drugs / 1,363
  diseases; `contraindication` 1,263 drugs / 1,195 diseases.
* Auxiliary graph: 25,653 + 80,411 + 321,075 = **427,139 edges** over 19,051
  `gene/protein` nodes that have at least one auxiliary edge.
* `gene/protein` incident degree in that auxiliary graph: min 1, median 20,
  mean 39.3, p90 89, **p95 134**, p99 316, **max 5,198** — a hard hub tail,
  which is exactly why DrKGC caps it. 950 nodes sit above the p95 cap.

## 7. Other things worth recording

* The KG lives under `data/kg/` here, not `data/`, and it is untracked (the root
  `.gitignore` ignores any `data/` directory). `config._detect_kg_folder()`
  therefore probes `data/kg` before `data`, and `$TXGNN_DATA_FOLDER` overrides
  both. If `kg_directed.csv` is absent, `kg_loader.load_kg_directed()` runs
  TxGNN's `preprocess_kg` (downloading `kg.csv` first only if it is missing).
* **`preprocess_kg` on the full PrimeKG is slow.** It loads all 8.1 M rows into
  pandas and, for every *homogeneous* relation, runs a row-wise
  `df.apply(..., axis=1)` to build a sorted-id key — `drug_drug` alone is
  2.67 M rows. Expect tens of minutes and several GB of RAM the first time;
  afterwards `kg_directed.csv` is reused. (This is TxGNN's own code path,
  untouched.)
* `drkgc/data/` is untracked for the same `.gitignore` reason. Intentional —
  every artifact is regenerable from `run_all.py`.
* The DrKGC paper (appendix A.1) builds its PrimeKG subset from exactly these
  9,388 `indication` triples and splits them 8,388/500/500 with the same
  entity-safety constraint. Our 90/5/5 (≈8,449/469/470) is the split the task
  asked for, so absolute sizes differ slightly from the paper's.
