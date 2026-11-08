#!/usr/bin/env bash
# The distilled claim extractor on Azure Container Apps: vLLM on a serverless A100 that scales to
# zero, woken by `vendorkb refresh` (claims/refresh.py) when the sync has brought new chunks.
#
# Nothing is built: the stock vLLM image serves Qwen3.5-9B (downloaded once into an Azure Files
# share) with the LoRA adapter, which is uploaded to the same share and never published.
#
# usage: ADAPTER_DIR=/path/to/adapter-r2 bash deploy/azure/deploy.sh
#   needs: az CLI logged in (az login), a subscription with serverless A100 GPU quota in
#   LOCATION. Prints the endpoint URL and keeps the API key in $KEY_FILE (mode 600, outside the
#   repo); put it in the cluster secret as EXTRACT_API_KEY (docs/deploy/k3s.md §7a).
#   Idempotent: a re-run updates the app in place and keeps the key.
set -euo pipefail

: "${ADAPTER_DIR:?set ADAPTER_DIR to the LoRA adapter directory (adapter_config.json inside)}"
LOCATION="${LOCATION:-swedencentral}"
RG="${RG:-vendorkb-extractor}"
ENV_NAME="${ENV_NAME:-vendorkb-extractor-env}"
APP="${APP:-vendorkb-extractor}"
SHARE="${SHARE:-models}"
STORAGE="${STORAGE:-vkbextract$(az account show --query id -o tsv | tr -d - | cut -c1-10)}"
VLLM_IMAGE="${VLLM_IMAGE:-docker.io/vllm/vllm-openai:v0.32.0}"
SERVED_NAME="${SERVED_NAME:-vendorkb-qwen3.5-9b-r2}"
HERE="$(cd "$(dirname "$0")" && pwd)"
KEY_FILE="${KEY_FILE:-$HOME/.config/vendorkb/extractor-api-key}"

test -f "$ADAPTER_DIR/adapter_config.json" || { echo "no adapter_config.json in $ADAPTER_DIR" >&2; exit 1; }

echo "== resource group, storage and share ($LOCATION)"
az group create -n "$RG" -l "$LOCATION" -o none
az storage account create -n "$STORAGE" -g "$RG" -l "$LOCATION" --sku Standard_LRS \
  --kind StorageV2 --min-tls-version TLS1_2 --allow-blob-public-access false -o none
KEY=$(az storage account keys list -n "$STORAGE" -g "$RG" --query "[0].value" -o tsv)
az storage share-rm create --storage-account "$STORAGE" -g "$RG" -n "$SHARE" --quota 100 -o none

echo "== adapter upload"
az storage directory create --account-name "$STORAGE" --account-key "$KEY" -s "$SHARE" \
  -n adapter -o none
for f in adapter_config.json adapter_model.safetensors chat_template.jinja tokenizer.json \
         tokenizer_config.json; do
  az storage file upload --account-name "$STORAGE" --account-key "$KEY" -s "$SHARE" \
    --source "$ADAPTER_DIR/$f" --path "adapter/$f" -o none
done

echo "== Container Apps environment with a serverless A100 profile"
if ! az containerapp env show -n "$ENV_NAME" -g "$RG" -o none 2>/dev/null; then
  az containerapp env create -n "$ENV_NAME" -g "$RG" -l "$LOCATION" \
    --enable-workload-profiles -o none
fi
if ! az containerapp env workload-profile show -n "$ENV_NAME" -g "$RG" \
     --workload-profile-name gpu -o none 2>/dev/null; then
  az containerapp env workload-profile add -n "$ENV_NAME" -g "$RG" \
    --workload-profile-name gpu --workload-profile-type Consumption-GPU-NC24-A100 -o none
fi
az containerapp env storage set -n "$ENV_NAME" -g "$RG" --storage-name models \
  --azure-file-account-name "$STORAGE" --azure-file-account-key "$KEY" \
  --azure-file-share-name "$SHARE" --access-mode ReadWrite -o none

echo "== app"
if [ -f "$KEY_FILE" ]; then
  API_KEY=$(cat "$KEY_FILE")
else
  API_KEY=$(openssl rand -hex 24)
  mkdir -p "$(dirname "$KEY_FILE")"
  (umask 077 && printf '%s' "$API_KEY" > "$KEY_FILE")
fi
ENV_ID=$(az containerapp env show -n "$ENV_NAME" -g "$RG" --query id -o tsv)
YAML=$(umask 077 && mktemp)
trap 'rm -f "$YAML"' EXIT
sed -e "s|__ENV_ID__|$ENV_ID|" -e "s|__IMAGE__|$VLLM_IMAGE|" -e "s|__SERVED__|$SERVED_NAME|" \
  -e "s|__LOCATION__|$LOCATION|" -e "s|__API_KEY__|$API_KEY|" "$HERE/app.yaml" > "$YAML"
if ! az containerapp show -n "$APP" -g "$RG" -o none 2>/dev/null; then
  az containerapp create -n "$APP" -g "$RG" --environment "$ENV_NAME" \
    --workload-profile-name gpu --image "$VLLM_IMAGE" --min-replicas 0 --max-replicas 1 \
    --ingress external --target-port 8000 -o none
fi
az containerapp update -n "$APP" -g "$RG" --yaml "$YAML" -o none
FQDN=$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
echo "endpoint: https://$FQDN/v1   (API key in $KEY_FILE)"
echo "first start downloads Qwen/Qwen3.5-9B (~18 GB) into the share; later cold starts reuse it"
