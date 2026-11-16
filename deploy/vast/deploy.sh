#!/usr/bin/env bash
# The extractor's Vast.ai serverless endpoint (README.md): a private template, the endpoint and
# its worker group. Idempotent: a re-run replaces the template (new hash) and points the worker
# group at it, keeping its machine exclusions.
#
# usage: VAST_API_KEY=... HF_TOKEN_READ=... bash deploy/vast/deploy.sh
#   VAST_API_KEY   the Vast account key (the cluster secret holds it too, for the refresh)
#   HF_TOKEN_READ  a read-only Hugging Face token for MODEL_REPO (workers run on third-party
#                  hosts: never a token that can write)
# needs: the vastai CLI (`uv tool install vastai`), curl, python3.
set -euo pipefail
: "${VAST_API_KEY:?}" "${HF_TOKEN_READ:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=endpoint.env
. "$HERE/endpoint.env"
V=(vastai --api-key "$VAST_API_KEY")
quiet() { "$@" >/dev/null 2>&1; }   # the CLI echoes templates back, start-up script (token) included

ONSTART=$(sed -e "s|__HF__|$HF_TOKEN_READ|" -e "s|__MODEL__|$MODEL_REPO|g" "$HERE/onstart.sh.tmpl")
ENVS="-p 3000:3000 -e MODEL_NAME=$MODEL_REPO -e AUTO_PARALLEL=true -e SERVERLESS=true -e BACKEND=openai -e MODEL_LOG=/var/log/portal/vllm.log -e MODEL_SERVER_URL=http://127.0.0.1:18000 -e MODEL_HEALTH_ENDPOINT=http://127.0.0.1:18000/health"

template_ids() {  # the ids of our templates named TEMPLATE_NAME, oldest first
  "${V[@]}" search templates "name=\"$TEMPLATE_NAME\"" --raw | python3 -c "
import sys, json
print(' '.join(str(t['id']) for t in sorted(json.load(sys.stdin), key=lambda t: t['id'])))"
}
OLD=$(template_ids)
echo "== template $TEMPLATE_NAME ($IMAGE:$IMAGE_TAG, $MODEL_REPO)"
# --ssh --direct: Vast runs the start-up script as a script (without a run type it becomes the
# container's command and the container never starts).
quiet "${V[@]}" create template --name "$TEMPLATE_NAME" --image "$IMAGE" --image_tag "$IMAGE_TAG" \
  --env "$ENVS" --onstart-cmd "$ONSTART" --ssh --direct --disk_space "$DISK_GB" \
  --search_params "$SEARCH_PARAMS" --desc "vendorkb claims extractor (private)"
read -r NEW_ID NEW_HASH < <("${V[@]}" search templates "name=\"$TEMPLATE_NAME\"" --raw | python3 -c "
import sys, json
t = max(json.load(sys.stdin), key=lambda t: t['id'])
assert t.get('private'), 'the template must be private'
print(t['id'], t['hash_id'])")
echo "template $NEW_ID ($NEW_HASH)"

echo "== endpoint $ENDPOINT"
EP_ID=$("${V[@]}" show endpoints --raw | python3 -c "
import sys, json
d = json.load(sys.stdin); d = d.get('results', d) if isinstance(d, dict) else d
print(next((str(e['id']) for e in d if e['endpoint_name'] == '$ENDPOINT'), ''))")
ARGS=(--min_load "$MIN_LOAD" --min_cold_load 0 --cold_workers "$COLD_WORKERS"
      --max_workers "$MAX_WORKERS" --inactivity_timeout "$INACTIVITY_TIMEOUT"
      --max_queue_time "$MAX_QUEUE_TIME" --target_queue_time "$TARGET_QUEUE_TIME")
if [ -z "$EP_ID" ]; then
  quiet "${V[@]}" create endpoint --endpoint_name "$ENDPOINT" "${ARGS[@]}"
  EP_ID=$("${V[@]}" show endpoints --raw | python3 -c "
import sys, json
d = json.load(sys.stdin); d = d.get('results', d) if isinstance(d, dict) else d
print(next(str(e['id']) for e in d if e['endpoint_name'] == '$ENDPOINT'))")
else
  quiet "${V[@]}" update endpoint "$EP_ID" "${ARGS[@]}"
fi
echo "endpoint $EP_ID"

echo "== worker group"
read -r WG_ID EXCLUDED < <("${V[@]}" show workergroups --raw | python3 -c "
import sys, json
d = json.load(sys.stdin); d = d.get('results', d) if isinstance(d, dict) else d
g = next((g for g in d if g.get('endpoint_name') == '$ENDPOINT'), None)
m = ((g or {}).get('search_query') or {}).get('machine_id') or {}
print(g['id'] if g else '-', ','.join(map(str, m.get('notin') or [])) or '-')")
QUERY="$SEARCH_PARAMS"
[ "$EXCLUDED" != "-" ] && QUERY="$QUERY machine_id notin [$EXCLUDED]"
if [ "$WG_ID" = "-" ]; then
  quiet "${V[@]}" create workergroup --template_hash "$NEW_HASH" --endpoint_name "$ENDPOINT" \
    --gpu_ram 40 --search_params "$QUERY"
else
  quiet "${V[@]}" update workergroup "$WG_ID" --endpoint_id "$EP_ID" --template_hash "$NEW_HASH" \
    --gpu_ram 40 --search_params "$QUERY"
fi
echo "worker group ${WG_ID/-/created} (excluded machines: ${EXCLUDED/-/none})"

for id in $OLD; do   # the replaced templates
  [ "$id" != "$NEW_ID" ] && quiet "${V[@]}" delete template --template-id "$id" && echo "deleted template $id"
done
echo "done: https://openai.vast.ai/$ENDPOINT (EXTRACT_MODEL vendorkb-qwen3.5-9b-r2; Helm claimsRefresh.extract.vast)"
