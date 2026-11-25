# The distilled extractor on a Vast.ai serverless endpoint

`vendorkb refresh` (weekly CronJob, `docs/deploy/k3s.md` §7a) extracts claims for new chunks
with the Opus-distilled Qwen3.5-9B. This directory serves that model on a **Vast.ai serverless
endpoint that scales to zero**: no worker runs, and no GPU is billed, between refreshes. The
refresh wakes it, extracts through its OpenAI-compatible proxy, and replaces bad hosts
(`src/vendorkb/claims/vast.py`). It replaced the Azure Container Apps plan of 2026-11-16 (the
container never served there, and the corporate account blocked the CLI).

- **Model:** `carev01/vendorkb-qwen3.5-9b-r2` on Hugging Face, private: the r2 adapter merged into
  Qwen3.5-9B bf16 (19.3 GB) with `merge_lora.py`, the corpus run's own merge. Workers download
  it with a read-only token.
- **Image:** Vast's `vastai/vllm:v0.32.0-cuda-13.0` (vLLM 0.32.0, the corpus run's version),
  with the corpus run's serving flags (`onstart.sh.tmpl`): thinking off, greedy decoding,
  `--max-model-len 12288`. vLLM answers to the repository name and to
  `vendorkb-qwen3.5-9b-r2`.
- **Endpoint:** `vendorkb-extractor` (`endpoint.env`): at most one worker, none on standby, back
  to zero 300 s after the last request.
- **Address:** `https://openai.vast.ai/vendorkb-extractor/chat/completions`, with the Vast
  account key as the Bearer token. There is no `/v1`: that path answers 404, and a 404 never
  starts a worker.

## Deploy

```
uv tool install vastai
VAST_API_KEY=... HF_TOKEN_READ=... bash deploy/vast/deploy.sh
```

The script is idempotent:
- It creates a private template, or replaces the old one.
- It creates or updates the endpoint.
- It points the worker group at the new template, keeping its machine exclusions.
- The CLI echoes templates back, start-up script included, so the script discards its output.

`HF_TOKEN_READ` is a fine-grained token with read access to the model's contents only. Workers
run on third-party hosts whose operators can read a container's environment.

Then the cluster:

```
kubectl -n vendorkb patch secret vendorkb-secret --type merge -p \
  "{\"stringData\":{\"VAST_API_KEY\":\"$VAST_API_KEY\"}}"
helm upgrade vendorkb deploy/helm/vendorkb -n vendorkb --reset-then-reuse-values \
  --set claimsRefresh.enabled=true \
  --set claimsRefresh.extract.vast.endpoint=vendorkb-extractor \
  --set-string "claimsRefresh.extract.vast.searchParams=$(. deploy/vast/endpoint.env; echo "$SEARCH_PARAMS")"
```

## Rebuilding the model

You only need this for a new adapter. Rent a 48 GB instance; the corpus run's
`setup_instance.sh` installs the pinned training stack. Then:

```
SRC=/workspace/adapter-r2 DST=/workspace/merged python merge_lora.py
hf upload carev01/vendorkb-qwen3.5-9b-r2 /workspace/merged . --private
```

Check the merged model before relying on it: serve it with the flags in `onstart.sh.tmpl` and
re-extract a few chunks the corpus run extracted.

## How a refresh uses it

`claims.vast.wake` sends a one-token chat request until one is answered. With no worker ready,
the proxy answers `504` after about 60 s instead of holding the request. The same cutoff ends
an unstreamed reply that takes longer (a chunk with many claims, while 32 calls share the one
worker: 41 chunks failed on 2026-11-23), so the refresh streams the extraction calls
(`EXTRACT_STREAM`, set by Helm with the endpoint). The extractor also loops on about 2.5% of chunks
(it repeats one claim until `max_tokens`, on every attempt): the stream is stopped at the third
repeat (error `repetition`), and the refresh hands those chunks, and any it left at `length`, to
`claimsRefresh.extract.fallback.model` (Opus 5.5) in one Anthropic Message Batch. Between requests,
`wake` checks the endpoint's workers:

- **Failed:** a worker in `error` or `rebooting` is destroyed, and its machine is added to the
  worker group's `machine_id notin [...]`.
- **Too slow:** a worker not ready after `vast_worker_ready_s` (900 s) is destroyed, but its
  machine is not excluded.

This is needed because **Vast does not move off a bad host.** A worker whose model load fails
is rebooted, stopped and restarted on the same machine.

The template's watchdog turns a bad host into a failed load. It does this when the download
runs under 25 MB/s, or when the worker is not serving 25 minutes after start. The PyWorker also
counts vLLM's tracebacks as a failed load (`MODEL_ERROR_LOG_MSGS`); by default the OpenAI
worker does not.

## Measured on 2026-11-16

| | |
|---|---|
| cold start, good host | about 4 min from worker created to serving, model download included |
| first request answered | 9 min 44 s, about 5 of them lost to a bad host |
| 8 already-extracted chunks | 8/8, 4-18 s each, claims equal to the stored ones on 5, worded differently on 3 |
| 32 at once | 32/32 in 51 s, mean claim agreement 0.95, no proxy errors |
| idle to stopped | the 300 s timeout, then `stopping`, then `stopped` |
| cost of the whole test | $2.50: the merge instance, the upload, and an hour on a slow host |

Bad hosts seen:
- one downloading from Hugging Face at about 3 MB/s;
- one that could not reach Hugging Face at all (TLS errors). It was recruited twice, so it is
  now excluded (machine 155931).

Workers that never started at all (no container) cost nothing: GPU billing starts at loading.

## Cost

Billing is per second at the host's price (about $0.48-0.70/h for these GPUs) while a worker
loads or serves. Its disk is billed while it exists, stopped included. Vast does not destroy a
stopped worker even with no standby workers (`COLD_WORKERS=0`): one stayed stopped from 23 to
25 Nov 2026, billing about $0.03/h for its 60 GB disk. So `vendorkb refresh` destroys the
endpoint's workers as soon as its extraction ends, or fails (`claims.vast.release`; best
effort, a failure is logged). Each due run therefore starts from a fresh worker, a cold start
of about 4 min, and pays neither the 300 s idle timeout nor any storage between runs. A refresh
of a few hundred chunks is one cold start and a few minutes of extraction: well under $1.
9,555 chunks (24 Nov 2026) took 85 min.
