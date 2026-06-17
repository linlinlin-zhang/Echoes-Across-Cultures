#!/usr/bin/env bash
set -euo pipefail

DOMAIN="${1:-}"
UPSTREAM="${2:-}"
EMAIL="${3:-}"

if [[ -z "${DOMAIN}" || -z "${UPSTREAM}" ]]; then
  echo "Usage: sudo bash setup_ubuntu_tailscale_nginx.sh <domain> <windows-tailscale-ip:8000> [email]" >&2
  echo "For the editable config-file flow, use install_echo_proxy.sh instead." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_CONFIG="$(mktemp)"
trap 'rm -f "${TMP_CONFIG}"' EXIT

cat >"${TMP_CONFIG}" <<ENV
ECHO_DOMAIN=${DOMAIN}
ECHO_EXTRA_DOMAINS=
ECHO_UPSTREAM=${UPSTREAM}
ECHO_EMAIL=${EMAIL}
ECHO_ENABLE_HTTPS=true
ECHO_CONFIGURE_UFW=false
ECHO_TAILSCALE_AUTHKEY=
ECHO_TAILSCALE_HOSTNAME=echo-proxy
ECHO_REQUIRE_UPSTREAM_HEALTH=true
ECHO_CLIENT_MAX_BODY_SIZE=500m
ENV

bash "${SCRIPT_DIR}/install_echo_proxy.sh" "${TMP_CONFIG}"
