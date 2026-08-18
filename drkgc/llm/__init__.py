"""Step 5 of the DrKGC pipeline: prompt construction and LLM fine-tuning.

The LLM does one job: given a question, a candidate set, and structural evidence,
pick the single best candidate. It never invents an entity, and it cannot recover
an answer the retriever missed - `recall@k` is a hard ceiling on its accuracy.

Modules
-------
prompts   question templates + prompt assembly (embedding / text / no evidence)
dataset   candidates + subgraphs -> training and evaluation examples
model     base LLM + LoRA + GCN adapter, with embedding splicing
train     QLoRA fine-tuning loop
evaluate  prediction, metrics, and the ranker-vs-LLM comparison
"""
