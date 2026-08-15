# DrKGC — Step 2: Lightweight Structure Model (R-GCN)

Trains an R-GCN + DistMult link predictor on the step-1 artifacts and exports
the two things every later step needs:

* `global_embeddings.pt` — `E_global`, one vector per entity, which the GCN
  adapter (step 4) initialises from;
* `candidates/*.csv` — the top-k candidate drugs per query, which become the
  constrained answer set in the LLM prompt (step 5).

It also gives you the **R-GCN baseline row** of the paper's Table 1 (filtered
MRR / Hits@k), so you can see what the LLM has to beat.

---

## Running it

```bash
# 1. train (writes model.pt + metrics.json)
python -m drkgc.ranker.train --out-dir drkgc/data --model-dir drkgc/models/rgcn

# 2. export embeddings + top-20 candidates
python -m drkgc.ranker.rank --out-dir drkgc/data --model-dir drkgc/models/rgcn -k 20
```

`--out-dir` must be the step-1 directory the model was trained on. For a
zero-shot run, point both at the matching artifacts, e.g.
`--out-dir drkgc/data_cardiovascular --model-dir drkgc/models/rgcn_cardiovascular`.

Inspect the assembled tensors without training anything:

```bash
python -m drkgc.ranker.data --out-dir drkgc/data
```

### Key flags

| Flag | Default | Meaning |
|---|---|---|
| `--dim` / `--layers` | 200 / 2 | R-GCN hidden size and depth (paper's PrimeKG setting is 200) |
| `--lr` / `--batch-size` | 1e-3 / 4096 | paper uses lr 1e-3, batch 256 — see the deviation note below |
| `--num-negatives` | 64 | type-constrained corruptions per positive, per side |
| `--epochs` / `--patience` | 300 / 8 | early stopping on valid MRR, evaluated every `--eval-every` (5) |
| `--mode` | `head` | `head` ranks drugs for a disease; `both` also does tail prediction |
| `--no-aux-supervision` | off | auxiliary edges become message-passing structure only, not training targets |
| `--uncapped` | off | train on the uncapped context graph |
| `-k` (rank.py) | 20 | candidate set size, as in the paper |

**Deviation from the paper's hyperparameters, on purpose:** batch 256 with 512
negatives means ~1,800 optimiser steps per epoch, and each step needs a full
graph forward pass (R-GCN encodes the whole KG, not a subgraph). Batch 4096 with
64 negatives trains the same model in a fraction of the time. Pass
`--batch-size 256 --num-negatives 512` to match the paper exactly.

---

## What it builds

**Index spaces.** Step 1 works in PrimeKG's per-node-type index space
(`(node_type, node_idx)`), which is what TxGNN's `HeteroData` uses. The R-GCN
needs one flat entity index, so `data.py` builds an `EntityIndex` bridging the
two and writes it to `entities_global.csv`. Row *i* of `global_embeddings.pt`
is `global_id == i` in that table — every later step should join through it.

**The training graph** is the capped context graph plus the *training* target
triples, with an inverse relation added for every relation so messages flow both
ways (`num_relations` is therefore twice the number of named relations). Held-out
triples never enter the graph.

**Negative sampling is type-constrained**: a corrupted head for `indication` is
always another drug, never a gene. This matches how candidates are generated at
inference, where only drugs are ever ranked.

**Evaluation is filtered** (Bordes et al. 2013): other entities forming a known
true triple with the same query are removed before ranking, using
train+valid+test as the filter set. Ties are optimistic
(`rank = 1 + #{strictly higher}`).

Candidate generation uses the same filter by default (`--filter all`), so the
MRR `rank.py` prints matches the one `train.py` reports. `--filter train`
removes only *training* triples, leaving other held-out true drugs in the
ranking to compete with the gold — a harsher, deployment-like setting. The gap
between the two is large for relations with many true drugs per disease:
contraindication averages ~26 drugs per disease against indication's ~7, so it
loses far more when its sibling answers are left in.

### Outputs

```
drkgc/models/rgcn/
├── model.pt                 best checkpoint (by valid MRR) + its config
├── metrics.json             valid/test MRR + Hits@{1,3,10}, per relation, + history
├── entities_global.csv      global_id, node_type, node_idx, node_id, node_name
└── global_embeddings.pt     {'embeddings': [num_entities, dim], ...}

drkgc/data/candidates/
├── indication_valid_candidates.csv
├── indication_test_candidates.csv
├── contraindication_*.csv
└── candidates_report.json   recall@k and MRR per relation/split
```

Each candidate row carries the query (`query_*`), the gold answer, `gold_rank`
in the full filtered ranking, `gold_in_candidates`, and the top-k
`candidate_global_ids` / `candidate_names` as JSON lists — ready to drop into
the prompt template.

---

## Zero-shot: disease prototypes

A disease held out of training has no drug edges, so its embedding is untrained
noise and the ranker cannot place it. `proto.py` ports TxGNN's answer
(`pyg_implementation/txgnn/model.py`, `DistMultPredictor` proto-learning):
describe each disease by its neighbours in the context graph, find the most
similar *training* diseases, and mix their embeddings into the query's.

```bash
# try it post-hoc on a model you already trained - seconds, no retraining
python -m drkgc.ranker.rank --out-dir drkgc/data_holdout \
    --model-dir drkgc/models/rgcn_holdout -k 20 --proto rarity

# if it helps, train with it so the encoder learns around it
python -m drkgc.ranker.train --out-dir drkgc/data_holdout \
    --model-dir drkgc/models/rgcn_holdout_proto --proto rarity

# inspect the prototype context alone
python -m drkgc.ranker.proto --out-dir drkgc/data_holdout
```

`--proto rarity` is TxGNN's default (`proto_num=5`, `exp_lambda=0.7`):
`alpha = lambda*exp(-lambda*degree) + 0.2`, so a disease with **zero** training
edges takes 90% of its embedding from the prototype while well-connected
diseases barely move. Other modes: `avg` (50/50), `heuristics-0.8` (20%
prototype), `100proto` (prototype only). `rank.py` inherits whatever the
checkpoint was trained with; `--proto` overrides it for experiments.

Two deliberate deviations from TxGNN, both in `proto.py`'s docstring: the
context is built once over all target relations rather than per relation (with
a shared disease partition a held-out disease has no edges of *any* target
relation), and self-matches are masked explicitly rather than via TxGNN's
"drop column 0 during training" shape heuristic.

**The profile is only as good as the context graph.** With the default
auxiliary relations it is built from `disease_protein` alone — TxGNN's
`protein_profile` variant. TxGNN's own default (`all_nodes_profile`) also uses
`disease_disease`, which is usually the stronger similarity signal. To match it,
rebuild step 1 with that relation included:

```bash
python -m drkgc.data_prep.run_all --split disease_holdout --out-dir drkgc/data_holdout \
    --aux-relations drug_protein disease_protein protein_protein disease_disease --force
```

That helps twice over: better prototypes, and `disease_disease` edges also give
the R-GCN a path into held-out diseases during message passing. It needs a
retrain, and it does not leak — `disease_disease` is not a target relation, and
the leakage check still runs.

## What to expect

On the **random split**, this is the setting the paper's R-GCN number comes from
(PrimeKG, MRR 0.640 / Hits@1 0.569). Their numbers are on a 500-triple test set
with both directions; ours is a larger test set and head-only by default, so
treat theirs as a ballpark, not a target to reproduce exactly.

On the **zero-shot splits** (`disease_holdout`, `disease_area`) expect a large
drop, and that is the honest result rather than a bug. Held-out diseases have no
drug edges at all in the training graph, so the ranker has to reach them purely
through mechanism paths — gene associations and PPI. In the `disease_area`
setting TxGNN additionally removes a sample of the area's 2-hop neighbourhood,
thinning even those paths. Closing that gap is the entire point of the DrKGC
steps that follow; a weak zero-shot R-GCN is the baseline they improve on, and
`recall@20` matters more here than MRR, because it bounds everything downstream.
