#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${APP_NAME:-echo-web}"
APP_PORT="${APP_PORT:-3000}"
SERVER_IP="${SERVER_IP:-39.108.139.62}"
APP_ROOT="${APP_ROOT:-/var/www/echo}"
PROJECT_SUBDIR="${PROJECT_SUBDIR:-web}"
REPO_URL="${REPO_URL:-}"
BRANCH="${BRANCH:-main}"
NODE_MAJOR="${NODE_MAJOR:-20}"

log() {
  echo "[deploy] $*"
}

fail() {
  echo "[deploy][error] $*" >&2
  exit 1
}

if ! command -v sudo >/dev/null 2>&1; then
  fail "sudo is required"
fi

log "Installing system packages..."
sudo apt-get update -y
sudo apt-get install -y curl git nginx build-essential

install_node=true
if command -v node >/dev/null 2>&1; then
  current_major="$(node -v | sed 's/^v//' | cut -d. -f1)"
  if [[ "${current_major}" == "${NODE_MAJOR}" ]]; then
    install_node=false
  fi
fi

if [[ "${install_node}" == "true" ]]; then
  log "Installing Node.js ${NODE_MAJOR}.x..."
  curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | sudo -E bash -
  sudo apt-get install -y nodejs
fi

log "Installing PM2..."
sudo npm install -g pm2

log "Preparing application directory..."
sudo mkdir -p "${APP_ROOT}"
sudo chown -R "${USER}:${USER}" "${APP_ROOT}"

if [[ -d "${APP_ROOT}/.git" ]]; then
  log "Repository exists. Pulling latest ${BRANCH}..."
  git -C "${APP_ROOT}" fetch --all --prune
  git -C "${APP_ROOT}" checkout "${BRANCH}"
  git -C "${APP_ROOT}" pull --ff-only origin "${BRANCH}"
else
  if [[ -z "${REPO_URL}" ]]; then
    fail "REPO_URL is required when ${APP_ROOT} is not a git repository"
  fi
  log "Cloning repository..."
  git clone --branch "${BRANCH}" "${REPO_URL}" "${APP_ROOT}"
fi

if [[ "${PROJECT_SUBDIR}" == "." || "${PROJECT_SUBDIR}" == "/" ]]; then
  APP_DIR="${APP_ROOT}"
else
  APP_DIR="${APP_ROOT}/${PROJECT_SUBDIR}"
fi

if [[ ! -f "${APP_DIR}/package.json" && -f "${APP_ROOT}/package.json" ]]; then
  APP_DIR="${APP_ROOT}"
fi

if [[ ! -f "${APP_DIR}/package.json" ]]; then
  fail "package.json not found under ${APP_DIR}. Check PROJECT_SUBDIR."
fi

cd "${APP_DIR}"

log "Installing dependencies..."
npm ci

log "Building Next.js app..."
npm run build

if pm2 describe "${APP_NAME}" >/dev/null 2>&1; then
  log "Restarting existing PM2 app: ${APP_NAME}"
  pm2 restart "${APP_NAME}" --update-env
else
  log "Starting PM2 app: ${APP_NAME}"
  pm2 start npm --name "${APP_NAME}" -- start -- -H 127.0.0.1 -p "${APP_PORT}"
fi

pm2 save
sudo env PATH="$PATH" pm2 startup systemd -u "${USER}" --hp "${HOME}" >/dev/null 2>&1 || true

log "Configuring Nginx reverse proxy..."
NGINX_CONF="/etc/nginx/sites-available/${APP_NAME}"
sudo tee "${NGINX_CONF}" >/dev/null <<EOF
server {
    listen 80;
    listen [::]:80;
    server_name ${SERVER_IP};

    location / {
        proxy_pass http://127.0.0.1:${APP_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

sudo ln -sfn "${NGINX_CONF}" "/etc/nginx/sites-enabled/${APP_NAME}"
if [[ -f /etc/nginx/sites-enabled/default ]]; then
  sudo rm -f /etc/nginx/sites-enabled/default
fi

sudo nginx -t
sudo systemctl enable nginx
sudo systemctl reload nginx

if command -v ufw >/dev/null 2>&1; then
  sudo ufw allow OpenSSH >/dev/null 2>&1 || true
  sudo ufw allow "Nginx Full" >/dev/null 2>&1 || true
fi

log "Deployment finished."
log "Open: http://${SERVER_IP}"
pm2 status "${APP_NAME}"
