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

## Fix round 1
- Caps (--max-usd, --max-rerank-tokens) are cumulative: seeded from rows already in --out (errored included; each row stores its `charge`).
- Charge per question: usage.cost; else $3/M prompt + $12/M completion tokens; else $0.05. Three no-information questions stop the run (exit 1, "STOPPED: unknown_cost").
- Exception rows (answer() or vendor lookup raising; lookup is now inside the per-question try) are written with the error and charged max($0.05, known cost); a lookup failure keeps the answer and usage.
- Also: torn last line no longer glues onto the next row; non-object JSON lines tolerated; error rows keep the yaml shape.
- Tests added for seeding, token estimate, $0.05 charge, 3-strike stop, vendor-lookup failure, torn line.
