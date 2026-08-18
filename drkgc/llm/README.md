# DrKGC — Step 5: Prompts and LLM Fine-Tuning

The LLM does one job: given a question, 20 candidates, and structural evidence,
**pick one candidate**. It never invents an entity and cannot recover an answer
the retriever missed — `recall@20` is a hard ceiling on its accuracy.

---

## Run it in this order

**1. Baseline first — no LLM, no GPU.** Establishes what the LLM has to beat:

```bash
python -m drkgc.llm.evaluate --out-dir drkgc/data_holdout \
    --model-dir drkgc/models/rgcn_holdout_v2 --split test --baseline-only
```

**2. Inspect one prompt** before spending GPU time on a malformed template:

```bash
python -m drkgc.llm.dataset --out-dir drkgc/data_holdout --split valid --evidence text
```

**3. Fine-tune** (the paper trains on the validation split, appendix A.5):

```bash
python -m drkgc.llm.train --out-dir drkgc/data_holdout \
    --model-dir drkgc/models/rgcn_holdout_v2 \
    --run-dir drkgc/models/llm_holdout
```

**4. Evaluate on test:**

```bash
python -m drkgc.llm.evaluate --out-dir drkgc/data_holdout \
    --model-dir drkgc/models/rgcn_holdout_v2 \
    --run-dir drkgc/models/llm_holdout --split test
```

For the random split, point the three paths at `drkgc/data`, `drkgc/models/rgcn`
and a new run dir. Nothing else changes.

### Key flags

| Flag | Default | Meaning |
|---|---|---|
| `--evidence` | `embedding` | `embedding` (adapter vectors), `text` (triples written out), `none`. The latter two are the paper's ablations. |
| `--model-id` | `meta-llama/Llama-3.2-3B-Instruct` | `llm_dim` is read from the model config, never assumed |
| `--no-dedup` | off | one prompt per triple (paper-faithful); default collapses to one per query entity |
| `--keep-unanswerable` | off | keep training examples whose gold is not in the candidates |
| `--batch-size` / `--grad-accum` | 1 / 8 | effective batch 8; raise only if VRAM allows |
| `--no-quantise` | off | bf16 LoRA instead of 4-bit QLoRA |

---

## How the embeddings get into the prompt

The prompt contains `<|reserved_special_token_0|>` — already in the Llama
vocabulary, exactly one token, never occurring in natural text. After
tokenisation the model:

1. embeds `input_ids` normally;
2. runs the GCN adapter on the query's retrieved subgraph;
3. scatters the adapter's `1 + k` vectors into the placeholder positions;
4. runs the LLM on `inputs_embeds`.

Gradients flow back through those positions into the adapter and the GCN, so the
graph encoder learns what to represent. This is why no off-the-shelf text trainer
is used.

`build_inputs_embeds` **asserts** the placeholder count matches the adapter
output. A mismatch means the prompt and the candidate set disagree, which would
otherwise mis-align every vector silently.

---

## Two policies that affect the numbers

**Unanswerable training examples are skipped.** When the retriever missed the
gold, no candidate is correct, so training on it teaches the model to emit an
entity outside its own instruction. On the zero-shot split this drops roughly
half the training prompts. Evaluation never skips — a missed gold counts as
wrong, which is what makes `recall@20` the ceiling.

**Training prompts are deduplicated per query entity by default.** The splits are
per triple, so a disease with 8 indications produces 8 near-identical prompts with
different single answers — contradictory supervision at token level and 8× the
cost. Dedup keeps the prompt whose gold ranks highest. **Evaluation is always per
triple**, so the reported metric is unaffected and stays comparable to the paper.
`--no-dedup` restores the faithful behaviour.

---

## Reading the result

`evaluate.py` reports both arms on identical queries:

```
recall_at_k        0.522     ← the ceiling; identical in both arms
ranker_hits@1      0.270     ← candidate list is in ranker order, so [0] is its pick
llm_hits@1         ?
delta_hits@1       ?
delta_ci_low/high  ?         ← bootstrap, resampled over QUERY ENTITIES
significant_at_95  ?
unmatched_rate     ?
```

**The CI is resampled over query entities, not triples.** Variance here is
disease-level — one disease contributes several triples that succeed or fail
together — so resampling triples would produce a misleadingly narrow interval.

**Decide the rule before looking:** if `significant_at_95` is false, the finding
is "no measurable improvement", not "a small improvement". The paper's own gain
was +2.3 Hits@1 points, and with ~65 query entities in indication/test an effect
that size is not resolvable. A null result here is a real possibility and should
be reported as one.

**Watch `unmatched_rate`.** It counts generations that matched no candidate. If
it is more than a few percent the prompt is not constraining the model, and the
Hits@1 number is measuring formatting rather than knowledge.
