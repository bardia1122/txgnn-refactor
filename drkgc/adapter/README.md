# DrKGC — Step 4: The GCN Adapter

Turns a retrieved subgraph into the vectors that replace the `[Placeholder]`
tokens in the LLM prompt:

```
global embedding (step 2) ──▶ SubgraphGCN ──▶ local embedding
                             (low-dim, per query)
[global ; local] ──▶ StructureAdapter ──▶ LLM input width
```

**This component has no loss.** The paper trains it jointly with the LLM, letting
gradients flow through the GCN during LoRA fine-tuning (section 3.5). So step 4
delivers a module and its data pipeline; step 5 trains it.

---

## Verifying it

```bash
python -m drkgc.adapter.data --out-dir drkgc/data_holdout \
    --model-dir drkgc/models/rgcn_holdout_v2

python -m drkgc.adapter.precompute --out-dir drkgc/data_holdout \
    --model-dir drkgc/models/rgcn_holdout_v2
```

The first prints per-subgraph node/edge counts and the batched shapes. The second
runs a forward and backward pass and asserts:

- query vectors are `[B, llm_dim]`, candidates `[B, k, llm_dim]`;
- all outputs finite;
- **gradients reach the GCN** — if they did not, joint training in step 5 would
  silently be a no-op, and the local embeddings would stay at their random
  initialisation;
- it reports how many candidates were isolated in the batch.

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--hidden-dim` | 128 | GCN width. The paper searched {128, 256}. |
| `--llm-dim` | 4096 | LLM hidden size — 4096 for Llama-3-8B, 3072 for Llama-3.2-3B. |
| `--batch-size` | 8 | subgraphs per batch |
| `--export` | off | write frozen local embeddings (diagnostics only, see below) |

---

## Design notes

**Isolated candidates still get a vector.** A candidate the retriever could not
connect to the query has no edges in the subgraph, but the prompt will still list
it, so it must have an embedding. `data.py` therefore adds a node for the query
and every candidate before adding any triple, and `RGCNConv`'s root weight means
an isolated node passes through its own projected features. Concretely: an
unreachable candidate is represented by its global embedding alone, which is the
correct fallback — no local structure was found, so there is no local signal to
add.

**Batching is by disjoint union.** Several subgraphs are concatenated with their
local indices offset, so one GCN call handles a batch without messages crossing
between queries. This is PyG's standard convention.

**LayerNorm on the adapter output.** Injected vectors sit alongside the LLM's own
token embeddings; without normalisation they can dominate early in fine-tuning
simply by having a larger scale. Not specified in the paper — added here.

**Relation ids come from the ranker.** `data.py` maps relation names through
`RankerData.rel2id`, so the adapter's `num_relations` matches the R-GCN's. A
relation present in a subgraph but unknown to the ranker is skipped rather than
silently mapped to id 0.

**`--export` is for diagnostics, not results.** It writes local embeddings from an
*untrained* GCN, which carry little signal. It exists so you can inspect shapes
and run a "no joint training" ablation; it is a deviation from the paper, which
back-propagates into the GCN.

---

## What step 5 will need from this

```python
from drkgc.adapter.data import collate, load_global_embeddings, load_samples
from drkgc.adapter.model import GCNAdapter

samples = load_samples(out_dir, "indication", "valid", data.rel2id)
batch = collate(samples[:8])
features = global_embeddings[batch["node_ids"]]

out = adapter(features, batch["edge_index"], batch["edge_type"],
              batch["query_index"], batch["candidate_index"])
# out["query"]      -> [B, llm_dim]        splice at the query placeholder
# out["candidates"] -> [B, k, llm_dim]     splice at candidate placeholders
```

`batch["meta"]` carries the query name, gold id, `gold_in_candidates`, and the
candidate list in prompt order, so the prompt builder needs nothing else.
