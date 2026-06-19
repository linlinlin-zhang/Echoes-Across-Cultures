#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-./echo-proxy.env}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo: sudo bash install_echo_proxy.sh ./echo-proxy.env" >&2
  exit 2
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  echo "Copy echo-proxy.env.example to echo-proxy.env and edit it first." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

: "${ECHO_DOMAIN:?Set ECHO_DOMAIN in ${CONFIG_PATH}}"
: "${ECHO_UPSTREAM:?Set ECHO_UPSTREAM in ${CONFIG_PATH}}"

ECHO_EXTRA_DOMAINS="${ECHO_EXTRA_DOMAINS:-}"
ECHO_EMAIL="${ECHO_EMAIL:-}"
ECHO_ENABLE_HTTPS="${ECHO_ENABLE_HTTPS:-true}"
ECHO_CONFIGURE_UFW="${ECHO_CONFIGURE_UFW:-false}"
ECHO_TAILSCALE_AUTHKEY="${ECHO_TAILSCALE_AUTHKEY:-}"
ECHO_TAILSCALE_HOSTNAME="${ECHO_TAILSCALE_HOSTNAME:-echo-proxy}"
ECHO_REQUIRE_UPSTREAM_HEALTH="${ECHO_REQUIRE_UPSTREAM_HEALTH:-true}"
ECHO_CLIENT_MAX_BODY_SIZE="${ECHO_CLIENT_MAX_BODY_SIZE:-500m}"

server_names="${ECHO_DOMAIN}"
if [[ -n "${ECHO_EXTRA_DOMAINS}" ]]; then
  server_names="${server_names} ${ECHO_EXTRA_DOMAINS}"
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y ca-certificates curl nginx certbot python3-certbot-nginx

if ! command -v tailscale >/dev/null 2>&1; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi

if ! tailscale status >/dev/null 2>&1; then
  if [[ -n "${ECHO_TAILSCALE_AUTHKEY}" ]]; then
    tailscale up --authkey "${ECHO_TAILSCALE_AUTHKEY}" --hostname "${ECHO_TAILSCALE_HOSTNAME}"
  else
    echo "Tailscale is installed but not logged in." >&2
    echo "Run this on the cloud server, finish login, then rerun this installer:" >&2
    echo "  sudo tailscale up --hostname ${ECHO_TAILSCALE_HOSTNAME}" >&2
    exit 3
  fi
fi

if [[ "${ECHO_REQUIRE_UPSTREAM_HEALTH}" == "true" ]]; then
  echo "Checking local Echo upstream: http://${ECHO_UPSTREAM}/api/health"
  curl --fail --silent --show-error --max-time 20 "http://${ECHO_UPSTREAM}/api/health" >/dev/null
fi

cat >/etc/nginx/conf.d/echo-proxy-upgrade.conf <<'NGINX'
map $http_upgrade $connection_upgrade {
    default upgrade;
    '' close;
}
NGINX

cat >/etc/nginx/sites-available/echo-proxy <<NGINX
server {
    listen 80;
    server_name ${server_names};

    client_max_body_size ${ECHO_CLIENT_MAX_BODY_SIZE};

    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 5;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/javascript application/javascript application/json application/xml image/svg+xml;

    location = /api/mainline/upload_recommend {
        proxy_pass http://${ECHO_UPSTREAM};

        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Host \$host;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection \$connection_upgrade;

        proxy_buffering off;
        proxy_request_buffering off;
        proxy_connect_timeout 60s;
        proxy_read_timeout 900s;
        proxy_send_timeout 900s;
    }

    location = /api/ai/kimi/chat/stream {
        proxy_pass http://${ECHO_UPSTREAM};

        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Host \$host;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection \$connection_upgrade;

        proxy_buffering off;
        proxy_cache off;
        proxy_connect_timeout 60s;
        proxy_read_timeout 900s;
        proxy_send_timeout 900s;
    }

    location ^~ /api/mainline/audio/ {
        proxy_pass http://${ECHO_UPSTREAM};

        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Host \$host;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Range \$http_range;
        proxy_set_header If-Range \$http_if_range;

        proxy_buffering off;
        proxy_request_buffering on;
        proxy_connect_timeout 60s;
        proxy_read_timeout 900s;
        proxy_send_timeout 900s;
    }

    location / {
        proxy_pass http://${ECHO_UPSTREAM};

        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Host \$host;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection \$connection_upgrade;

        proxy_buffering on;
        proxy_buffer_size 16k;
        proxy_buffers 32 16k;
        proxy_busy_buffers_size 64k;
        proxy_request_buffering on;
        proxy_connect_timeout 60s;
        proxy_read_timeout 900s;
        proxy_send_timeout 900s;
    }
}
NGINX

ln -sf /etc/nginx/sites-available/echo-proxy /etc/nginx/sites-enabled/echo-proxy
rm -f /etc/nginx/sites-enabled/default

nginx -t
systemctl enable --now nginx
systemctl reload nginx

if [[ "${ECHO_CONFIGURE_UFW}" == "true" ]]; then
  apt-get install -y ufw
  ufw allow OpenSSH
  ufw allow 80/tcp
  ufw allow 443/tcp
  ufw --force enable
fi

if [[ "${ECHO_ENABLE_HTTPS}" == "true" && "${ECHO_DOMAIN}" != "_" ]]; then
  certbot_args=(--nginx --redirect --non-interactive --agree-tos)
  for domain in ${server_names}; do
    certbot_args+=(-d "${domain}")
  done
  if [[ -n "${ECHO_EMAIL}" ]]; then
    certbot_args+=(-m "${ECHO_EMAIL}")
  else
    certbot_args+=(--register-unsafely-without-email)
  fi
  certbot "${certbot_args[@]}"
fi

echo "Echo proxy-only server is configured."
echo "Public server names: ${server_names}"
echo "Private upstream: http://${ECHO_UPSTREAM}"
echo "Verify with: bash verify_echo_proxy.sh ${CONFIG_PATH}"
