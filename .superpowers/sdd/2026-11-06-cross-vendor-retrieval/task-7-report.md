# Task 7 (steps 1-2): answer-quality runner

Files: eval/answer_eval.py, tests/unit/test_answer_eval.py. Gate: ruff, mypy, pytest all green (268 passed).

## Commands (controller, paid)
Smoke:
  uv run python eval/answer_eval.py run --out eval/runs/answers-2026-11-06.jsonl --ids sv-a-01 xv-01
Full (resumes; skips ids already in the file):
  uv run python eval/answer_eval.py run --out eval/runs/answers-2026-11-06.jsonl --max-usd 10 --max-rerank-tokens 3000000
Summarize:
  uv run python eval/answer_eval.py summarize --answers eval/runs/answers-2026-11-06.jsonl --grades eval/runs/answer-grades-2026-11-06.jsonl

## Notes
- Citations from answer() carry text/url/title but not vendor; vendor is looked up from documents by doc_id (one query per question).
- Resume skips every id present, including errored lines (an errored paid call is not silently repeated); delete the line to retry.
- The rerank cap counts tokens used in this process only; subtract Task 6 usage when passing --max-rerank-tokens (cap is 3M across Tasks 6-7).
- Caps are checked before each question, so the last one can overshoot slightly.
