#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${APP_NAME:-echo-web}"
APP_ROOT="${APP_ROOT:-/var/www/echo}"
PROJECT_SUBDIR="${PROJECT_SUBDIR:-web}"
BRANCH="${BRANCH:-main}"

log() {
  echo "[update] $*"
}

if [[ "${PROJECT_SUBDIR}" == "." || "${PROJECT_SUBDIR}" == "/" ]]; then
  APP_DIR="${APP_ROOT}"
else
  APP_DIR="${APP_ROOT}/${PROJECT_SUBDIR}"
fi

if [[ ! -d "${APP_ROOT}/.git" ]]; then
  echo "[update][error] ${APP_ROOT} is not a git repository" >&2
  exit 1
fi

if [[ ! -f "${APP_DIR}/package.json" && -f "${APP_ROOT}/package.json" ]]; then
  APP_DIR="${APP_ROOT}"
fi

if [[ ! -f "${APP_DIR}/package.json" ]]; then
  echo "[update][error] package.json not found in ${APP_DIR}" >&2
  exit 1
fi

log "Pulling latest ${BRANCH}..."
git -C "${APP_ROOT}" fetch --all --prune
git -C "${APP_ROOT}" checkout "${BRANCH}"
git -C "${APP_ROOT}" pull --ff-only origin "${BRANCH}"

cd "${APP_DIR}"

log "Installing dependencies..."
npm ci

log "Building..."
npm run build

log "Restarting PM2 app: ${APP_NAME}"
pm2 restart "${APP_NAME}" --update-env
pm2 save

log "Done."
pm2 status "${APP_NAME}"
