# Echo Ubuntu 24.04 Deployment Notes

This deployment shape serves the static web app and FastAPI backend from the same origin. Visitors stay anonymous: the backend issues an `echo_anon_id` cookie and stores that visitor's favorites in SQLite under `ECHO_USER_DATA_DIR`.

The cloud server can run in a lightweight mode. In that mode it reads the catalog from CSV for pages and default favorites, while CultureMERT/DCAS-heavy recommendation requests are forwarded to a local worker running on your own computer.

## Server Layout

- App checkout: `/srv/echo`
- Python virtualenv: `/srv/echo/.venv`
- Runtime data: `/srv/echo/storage`
- Compressed public audio copy: `/srv/echo/cloud_audio_10gb`
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
sudo -u echo .venv/bin/pip install -r requirements-cloud.txt
sudo -u echo cp configs/server.env.example /etc/echo/echo.env
sudo editor /etc/echo/echo.env
```

Use `requirements-cloud.txt` on the cloud web server. It intentionally excludes
Torch, Transformers, and other research-worker packages. Install the full
`requirements.txt` only on the local worker that runs CultureMERT/DCAS
inference.

Keep `ECHO_COOKIE_SECURE=false` while testing on plain HTTP. Change it to `true` after you enable HTTPS in Nginx.

The mainline recommender expects the model and public catalog artifacts under `storage/` by default. Put these files in place before starting the service:

- `storage/public/merged/tracks_culturemert.npz`
- `storage/public/merged/metadata_merged.csv`
- `storage/models/dcas_full_v4_main_culturemert_stage3.pt`

For a small cloud server, only the metadata CSV plus audio files are required for browsing, streaming, and first-visit favorites. If you upload the compressed audio bundle to `/srv/echo/cloud_audio_10gb`, set this in `/etc/echo/echo.env`:

```bash
ECHO_MAINLINE_METADATA_PATH=/srv/echo/cloud_audio_10gb/metadata_merged.csv
```

The frontend should keep using `/api/mainline/catalog`, `/api/mainline/random`, and `/api/mainline/audio/{track_id}`. Do not put `/srv/echo/cloud_audio_10gb` into the HTML; that path is only for the backend to resolve files on disk. The `.npz` and `.pt` files can stay on the local worker machine.

## Cloud Gemini Upload Embedding

If you want uploaded-audio recommendation to run fully on the cloud server without exposing a local worker, use Gemini Embedding 2 for the upload embedding step. Put the Gemini-built mainline artifacts on the server, then set:

```bash
ECHO_UPLOAD_EMBEDDING_PROVIDER=gemini
GEMINI_API_KEY=your-google-ai-studio-key
ECHO_GEMINI_EMBEDDING_MODEL=gemini-embedding-2
ECHO_GEMINI_EMBEDDING_DIM=768
ECHO_MAINLINE_TRACKS_PATH=/srv/echo/storage/public/research_dataset_v4/main/tracks_gemini_embedding2_mw3.npz
ECHO_MAINLINE_MODEL_PATH=/srv/echo/storage/models/dcas_full_v4_main_gemini_stage3.pt
ECHO_MAINLINE_WORKER_URL=
ECHO_MAINLINE_WORKER_TOKEN=
```

Keep `ECHO_MAINLINE_METADATA_PATH` pointed at the deployed metadata CSV. The important rule is that uploaded Gemini embeddings, `ECHO_MAINLINE_TRACKS_PATH`, and `ECHO_MAINLINE_MODEL_PATH` must all be from the same Gemini embedding space. Using Gemini upload embeddings with a CultureMERT-trained mainline model can run if dimensions match, but the recommendation geometry is not meaningful.

## Local Mainline Worker

On your own computer, keep the full `storage/` directory and run the worker with CultureMERT/DCAS available:

```powershell
Copy-Item configs/local_worker.env.example configs/local_worker.env
notepad configs/local_worker.env
.\scripts\run_local_mainline_worker.ps1 -Port 18011
```

Set `ECHO_WORKER_SHARED_TOKEN` in `configs/local_worker.env` to a long random value. Then expose `http://127.0.0.1:18011` to the cloud server through a tunnel or reverse proxy you control.

If your computer has no public address, use Cloudflare Tunnel. See `deploy/cloudflare-tunnel/README.md` for the temporary Quick Tunnel and stable named-tunnel workflows.

On the Ubuntu cloud server, set these values in `/etc/echo/echo.env`:

```bash
ECHO_MAINLINE_WORKER_URL=https://your-local-worker-public-url
ECHO_MAINLINE_WORKER_TOKEN=the-same-long-random-token
ECHO_MAINLINE_WORKER_TIMEOUT_SECONDS=900
```

After that, the cloud server handles pages, sessions, favorites, and catalog browsing locally, but forwards:

- `POST /api/mainline/recommend`
- `POST /api/mainline/upload_recommend`
- `GET /api/mainline/status`

to your local worker. If the worker is offline, catalog browsing and anonymous favorites still work, while recommendation calls return an explicit worker-unavailable error.

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
