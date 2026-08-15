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
