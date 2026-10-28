#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics}"
TAG="${2:-leaderboard-data}"
WORKFLOW_PATH=".github/workflows/docs.yml"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command '$1' is not available." >&2
    exit 2
  fi
}

need_cmd git
need_cmd curl
need_cmd jq

HEAD_SHA="$(git rev-parse HEAD)"
HEAD_SHORT="$(git rev-parse --short HEAD)"
SHA_ASSET="map_data-${HEAD_SHA}.tar.gz"

RELEASE_JSON="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/tags/${TAG}")"
ASSET_NAMES="$(printf '%s' "$RELEASE_JSON" | jq -r '.assets[].name')"

HAS_STABLE="false"
HAS_SHA="false"

if printf '%s\n' "$ASSET_NAMES" | grep -Fxq "map_data.tar.gz"; then
  HAS_STABLE="true"
fi
if printf '%s\n' "$ASSET_NAMES" | grep -Fxq "$SHA_ASSET"; then
  HAS_SHA="true"
fi

WORKFLOW_ID="$(curl -fsSL "https://api.github.com/repos/${REPO}/actions/workflows" \
  | jq -r --arg p "$WORKFLOW_PATH" '.workflows[] | select(.path==$p) | .id' | head -n1)"

RUN_STATUS="missing"
RUN_CONCLUSION="missing"
RUN_ID="missing"

if [ -n "$WORKFLOW_ID" ]; then
  RUN_JSON="$(curl -fsSL "https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW_ID}/runs?head_sha=${HEAD_SHA}&per_page=1")"
  RUN_ID="$(printf '%s' "$RUN_JSON" | jq -r '.workflow_runs[0].id // "none"')"
  RUN_STATUS="$(printf '%s' "$RUN_JSON" | jq -r '.workflow_runs[0].status // "none"')"
  RUN_CONCLUSION="$(printf '%s' "$RUN_JSON" | jq -r '.workflow_runs[0].conclusion // "none"')"
fi

echo "Repo: ${REPO}"
echo "Tag: ${TAG}"
echo "HEAD: ${HEAD_SHA} (${HEAD_SHORT})"
echo "Expected SHA asset: ${SHA_ASSET}"
echo "Release has stable asset: ${HAS_STABLE}"
echo "Release has HEAD SHA asset: ${HAS_SHA}"
echo "Latest workflow run for HEAD: id=${RUN_ID} status=${RUN_STATUS} conclusion=${RUN_CONCLUSION}"

OK=true
if [ "$HAS_STABLE" != "true" ] || [ "$HAS_SHA" != "true" ]; then
  OK=false
fi
if [ "$RUN_CONCLUSION" != "success" ]; then
  OK=false
fi

if [ "$OK" = true ]; then
  echo "RESULT: OK - release assets and workflow are aligned with HEAD."
  exit 0
fi

echo "RESULT: KO - missing release asset for HEAD and/or workflow not successful."
exit 1
