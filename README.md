# vendorkb

This is a knowledge system over backup vendors' technical documentation. It returns cited answers and
abstains when nothing supports an answer. Documents come from DocExtractor. They are split into section
chunks and searched by keyword (Postgres full-text) and by vector (pgvector). The two result lists are
merged with reciprocal rank fusion and reranked.

- Design and milestones: [`DESIGN.md`](DESIGN.md)
- Why this replaced the Graphiti graph:
  [`docs/superpowers/head-to-head-report-2026-11-04.md`](docs/superpowers/head-to-head-report-2026-11-04.md)
- Developer guide: [`CLAUDE.md`](CLAUDE.md)

```bash
uv sync --extra dev
cp .env.example .env   # fill in
docker compose up -d
uv run vendorkb search "How does AWS Backup Vault Lock work?" --k 10
```
