# Echo Ubuntu 24.04 Deployment Notes

This deployment shape serves the static web app and FastAPI backend from the same origin. Visitors stay anonymous: the backend issues an `echo_anon_id` cookie and stores that visitor's favorites in SQLite under `ECHO_USER_DATA_DIR`.

## Server Layout

- App checkout: `/srv/echo`
- Python virtualenv: `/srv/echo/.venv`
- Runtime data: `/srv/echo/storage`
- Environment file: `/etc/echo/echo.env`
- Service: `echo.service`
- Reverse proxy: Nginx to `127.0.0.1:18010`

## First Setup

```bash
sudo apt update
sudo apt install -y python3.12-venv python3-pip ffmpeg nginx git
sudo useradd --system --home /srv/echo --shell /usr/sbin/nologin echo
sudo mkdir -p /srv/echo /etc/echo
sudo chown -R echo:echo /srv/echo
```

Clone or copy the repository into `/srv/echo`, then install Python dependencies:

```bash
cd /srv/echo
sudo -u echo python3 -m venv .venv
sudo -u echo .venv/bin/pip install --upgrade pip
sudo -u echo .venv/bin/pip install -r requirements.txt
sudo -u echo cp configs/server.env.example /etc/echo/echo.env
sudo editor /etc/echo/echo.env
```

Keep `ECHO_COOKIE_SECURE=false` while testing on plain HTTP. Change it to `true` after you enable HTTPS in Nginx.

The mainline recommender expects the model and public catalog artifacts under `storage/`. Put these files in place before starting the service:

- `storage/public/merged/tracks_culturemert.npz`
- `storage/public/merged/metadata_merged.csv`
- `storage/models/dcas_full_v4_main_culturemert_stage3.pt`

## Service And Proxy

```bash
sudo cp deploy/ubuntu24/echo.service /etc/systemd/system/echo.service
sudo systemctl daemon-reload
sudo systemctl enable --now echo.service

sudo cp deploy/ubuntu24/nginx.conf /etc/nginx/sites-available/echo
sudo ln -sf /etc/nginx/sites-available/echo /etc/nginx/sites-enabled/echo
sudo nginx -t
sudo systemctl reload nginx
```

## Anonymous Favorites

`GET /api/favorites` creates a visitor session if needed. On first access, it seeds 20 random tracks from the mainline catalog and stores them in:

```text
$ECHO_USER_DATA_DIR/echo.sqlite3
```

Deleting that SQLite file resets anonymous favorites for all visitors on that deployment.
