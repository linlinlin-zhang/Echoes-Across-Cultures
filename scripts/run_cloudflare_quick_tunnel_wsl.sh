#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-18012}"
HOST_NAME="${HOST_NAME:-}"
PROTOCOL="${PROTOCOL:-http2}"
BIN="${BIN:-tools/cloudflared-linux-amd64}"
SKIP_ORIGIN_CHECK="${SKIP_ORIGIN_CHECK:-false}"
WORKER_TOKEN="${ECHO_WORKER_SHARED_TOKEN:-${ECHO_MAINLINE_WORKER_TOKEN:-}}"

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

if [[ -z "$WORKER_TOKEN" && -f "configs/local_worker.env" ]]; then
  WORKER_TOKEN="$(awk -F= '/^ECHO_WORKER_SHARED_TOKEN=/{print $2; exit}' configs/local_worker.env)"
fi
if [[ -z "$WORKER_TOKEN" && -f "configs/local_worker.env" ]]; then
  WORKER_TOKEN="$(awk -F= '/^ECHO_MAINLINE_WORKER_TOKEN=/{print $2; exit}' configs/local_worker.env)"
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

if [[ "$SKIP_ORIGIN_CHECK" != "true" ]]; then
  auth_args=()
  if [[ -n "$WORKER_TOKEN" ]]; then
    auth_args=(-H "X-Echo-Worker-Token: ${WORKER_TOKEN}")
  fi
  if ! curl -fsS --max-time 5 "${auth_args[@]}" "${TARGET}/api/mainline/status" >/dev/null; then
    cat >&2 <<EOF

Cannot reach the local Echo worker at ${TARGET}.

Start the worker from Windows PowerShell first:
  cd E:\\Desktop\\Echo
  .\\scripts\\run_local_mainline_worker.ps1 -HostName 0.0.0.0 -Port ${PORT}

If the worker is already running:
  - confirm configs/local_worker.env contains ECHO_WORKER_SHARED_TOKEN
  - confirm the token matches the worker process you started
  - Windows Firewall or a VPN/TUN proxy may be blocking WSL -> Windows host traffic
EOF
    exit 1
  fi
fi

exec "$BIN" tunnel --protocol "$PROTOCOL" --url "$TARGET"
