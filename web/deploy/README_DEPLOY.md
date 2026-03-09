# Ubuntu 22.04 Deployment (No Domain)

This folder includes two scripts:

- `bootstrap_ubuntu.sh`: first-time setup and deploy
- `update_ubuntu.sh`: pull latest code and redeploy

## 1) Prepare your repository

Recommended: create a dedicated Git repository for this project and push source code.

Keep these files/folders in repo (from `web/`):

- `app/`, `components/`, `data/`, `hooks/`, `lib/`, `public/`, `styles/`
- `package.json`, `package-lock.json`
- `next.config.mjs`, `tsconfig.json`, `tailwind.config.ts`, `postcss.config.mjs`
- `deploy/` (this folder)

Do NOT upload build/runtime artifacts:

- `.next/`
- `node_modules/`
- `dist/` (legacy Vite output in this project)
- `.dev-logs/`

## 2) First-time deploy on server

Recommended flow (clone once, then deploy):

```bash
sudo mkdir -p /var/www
sudo chown -R $USER:$USER /var/www

git clone <YOUR_REPO_URL> /var/www/echo
cd /var/www/echo/web
chmod +x deploy/bootstrap_ubuntu.sh deploy/update_ubuntu.sh
APP_ROOT=/var/www/echo BRANCH=main SERVER_IP=39.108.139.62 bash deploy/bootstrap_ubuntu.sh
```

After completion, open:

- `http://39.108.139.62`

## 3) Update deployment after new commits

```bash
cd /var/www/echo/web
bash deploy/update_ubuntu.sh
```

## Optional variables

You can override defaults:

- `APP_NAME` (default: `echo-web`)
- `APP_PORT` (default: `3000`)
- `APP_ROOT` (default: `/var/www/echo`)
- `PROJECT_SUBDIR` (default: `web`)
- `BRANCH` (default: `main`)
- `NODE_MAJOR` (default: `20`)
- `REPO_URL` (required only when `APP_ROOT` is not already a git repository)
