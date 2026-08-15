"""Step 1 of the DrKGC pipeline: data preparation.

Modules
-------
kg_loader           load PrimeKG through the existing TxGNN code + resolve names
extract_triples     (drug, indication|contraindication, disease) triple tables
split_base          the split contract shared by every splitting strategy
split_random        entity-safe random split (swappable for a disease-area one)
build_context_graph auxiliary graph (drug-target / gene-disease / PPI) + leakage check
degree_cap          gene/protein degree statistics + capped auxiliary graph
run_all             end-to-end driver
test_sanity         assert-based checks over the produced artifacts
"""
