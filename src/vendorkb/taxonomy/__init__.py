"""M4 taxonomy: grow it from extracted claims, load it, map claims to it, build capabilities.

``propose`` clusters the distinct normalized capability (or target) strings in
``raw_claims`` and asks the extraction model for one entry per cluster, written to
``taxonomy/taxonomy.yaml`` for human review. ``load`` loads the reviewed YAML into the v2
taxonomy tables. ``mapper`` maps each v3 claim to taxonomy v2 entries (exact alias, else LLM
adjudication of the nearest candidates). ``capabilities`` rebuilds the ``capabilities`` fact
table from the mapped claims and queries it with scope-labelled groups.
"""
