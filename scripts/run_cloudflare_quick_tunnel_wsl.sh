#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-18012}"
HOST_NAME="${HOST_NAME:-}"
PROTOCOL="${PROTOCOL:-http2}"
BIN="${BIN:-tools/cloudflared-linux-amd64}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST_NAME="$2"
      shift 2
      ;;
    --protocol)
      PROTOCOL="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$HOST_NAME" ]]; then
  HOST_NAME="$(ip route | awk '/default/ {print $3; exit}')"
fi

if [[ ! -x "$BIN" ]]; then
  mkdir -p "$(dirname "$BIN")"
  echo "Downloading cloudflared for Linux amd64 to $BIN"
  curl -L --fail -o "$BIN" "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
  chmod +x "$BIN"
fi

TARGET="http://${HOST_NAME}:${PORT}"
echo "Starting Cloudflare Quick Tunnel for ${TARGET}"
echo "Copy the generated https://*.trycloudflare.com URL into ECHO_MAINLINE_WORKER_URL on the cloud server."
echo "Using protocol: ${PROTOCOL}"

exec "$BIN" tunnel --protocol "$PROTOCOL" --url "$TARGET"
