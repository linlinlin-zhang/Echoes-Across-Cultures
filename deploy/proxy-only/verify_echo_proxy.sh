#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-./echo-proxy.env}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

: "${ECHO_DOMAIN:?Set ECHO_DOMAIN in ${CONFIG_PATH}}"
: "${ECHO_UPSTREAM:?Set ECHO_UPSTREAM in ${CONFIG_PATH}}"

scheme="https"
if [[ "${ECHO_ENABLE_HTTPS:-true}" != "true" || "${ECHO_DOMAIN}" == "_" ]]; then
  scheme="http"
fi

echo "[1/4] Tailscale status"
tailscale status >/dev/null
tailscale ip -4 || true

echo "[2/4] Local Windows Echo through Tailscale"
curl --fail --show-error --max-time 20 "http://${ECHO_UPSTREAM}/api/health"
echo

echo "[3/4] Public proxy health"
curl --fail --show-error --max-time 30 "${scheme}://${ECHO_DOMAIN}/api/health"
echo

echo "[4/4] Public music page headers with gzip/cache"
curl \
  --fail \
  --show-error \
  --silent \
  --dump-header - \
  --output /dev/null \
  --header "Accept-Encoding: gzip" \
  --max-time 30 \
  "${scheme}://${ECHO_DOMAIN}/music.html"

echo "Proxy verification passed."
