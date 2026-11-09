# The distilled extractor on Azure Container Apps

`vendorkb refresh` (weekly CronJob, `docs/deploy/k3s.md` §7a) extracts claims for new chunks
with the Opus-distilled Qwen3.5-9B. This directory serves that model on a **serverless A100
that scales to zero**: nothing runs or bills between DocExtractor refreshes (about monthly),
and the first request after a quiet period wakes it.

- Image: the stock `vllm/vllm-openai`. Nothing is built or pushed.
- Model: `Qwen/Qwen3.5-9B` plus the LoRA adapter (`adapter-r2`, 131 MB). Both live in an Azure
  Files share mounted at `/models`. The base model is downloaded there on the first start. The
  adapter is uploaded by `deploy.sh` and never published.
- Serving matches the corpus run: thinking off, greedy decoding for requests that send no
  temperature, `--max-model-len 12288`. Requests name the adapter (`vendorkb-qwen3.5-9b-r2`,
  the `EXTRACT_MODEL` the CronJob sends).
- Auth: vLLM requires `Authorization: Bearer <key>` on every request. The key is generated once,
  kept in `~/.config/vendorkb/extractor-api-key`, and stored as a Container Apps secret.

## Deploy

Prerequisites:
- the az CLI, logged in (`az login`);
- serverless GPU quota for `Consumption-GPU-NC24-A100` in the region (default `swedencentral`;
  override with `LOCATION`);
- the adapter directory (it is kept outside the repo, under `graph-rag-distill/.../adapter-r2`).

```
ADAPTER_DIR=/path/to/adapter-r2 bash deploy/azure/deploy.sh
```

The script is idempotent. It prints `endpoint: https://<app>.<region>.azurecontainerapps.io/v1`.
The first start downloads about 18 GB and can take up to about 20 minutes (the startup probe
allows 21). Later cold starts reuse the share.

Check it (this wakes the app):

```
KEY=$(cat ~/.config/vendorkb/extractor-api-key)
curl -s -H "Authorization: Bearer $KEY" https://<fqdn>/v1/models
```

Then wire the CronJob: put the key in the cluster secret as `EXTRACT_API_KEY`, and enable
`claimsRefresh` with `claimsRefresh.extract.baseUrl` set to the endpoint (`docs/deploy/k3s.md`
§7a).

## Cost

Billing is per second while a replica runs. A refresh run is one cold start (minutes), the
extraction (the corpus run did about 6 chunks/s on an A100), and the 600 s cooldown before
scaling back to zero. The storage account (about 20 GB) costs cents a month.

## Validation

Validated 2026-11-08 on a Vast A100 with vLLM 0.32.0 and these exact flags: base model plus the
LoRA adapter, compared with the merged model that extracted the corpus on the same 59 requests.

| | |
|---|---|
| identical reply text | 49 / 59 |
| mean text similarity | 0.991 |
| claims extracted | 152 vs 152 |
| claim sets (capability, target, support), mean Jaccard | 0.915 |

The differences are small numeric effects of applying the adapter at serving time rather than
merging it into the weights. The `deploy.sh` and `app.yaml` themselves have not run yet: they
need an Azure login with serverless GPU quota.
