# Echo Local Worker Tunnel

This project can keep the cloud server lightweight while your own computer runs the CultureMERT/DCAS-heavy worker. If your computer does not have a public IP or cannot open inbound ports, use Cloudflare Tunnel to publish the local worker through an outbound `cloudflared` connection.

Cloudflare Tunnel has two useful modes for Echo:

- Quick Tunnel: fastest temporary test URL, no domain required.
- Named Tunnel: stable production URL, requires a Cloudflare account and a domain on Cloudflare DNS.

## 1. Start The Local Worker

From the repository root on your computer:

```powershell
Copy-Item configs/local_worker.env.example configs/local_worker.env
notepad configs/local_worker.env
```

Set a long random shared token:

```env
ECHO_WORKER_REQUIRE_TOKEN=true
ECHO_WORKER_SHARED_TOKEN=replace-with-a-long-random-secret
ECHO_MAINLINE_WORKER_URL=
```

Then start the worker:

```powershell
.\scripts\run_local_mainline_worker.ps1 -Port 18011
```

When `ECHO_WORKER_REQUIRE_TOKEN=true`, the worker only exposes `/api/mainline/*` and requires the `X-Echo-Worker-Token` header. The public tunnel URL should not be used as a normal Echo website.

## 2. Temporary Quick Tunnel

Install `cloudflared`, then run:

```powershell
.\scripts\run_cloudflare_quick_tunnel.ps1 -Port 18011
```

or directly:

```powershell
cloudflared tunnel --url http://127.0.0.1:18011
```

Copy the generated `https://*.trycloudflare.com` URL into the cloud server env:

```bash
ECHO_MAINLINE_WORKER_URL=https://generated-name.trycloudflare.com
ECHO_MAINLINE_WORKER_TOKEN=the-same-long-random-secret
ECHO_MAINLINE_WORKER_TIMEOUT_SECONDS=900
```

Quick Tunnel URLs are meant for testing. They can change whenever the process restarts.

## 3. Stable Named Tunnel

Use this when the cloud server should always call a fixed URL such as `https://echo-worker.example.com`.

```powershell
cloudflared tunnel login
cloudflared tunnel create echo-mainline-worker
```

Copy `deploy/cloudflare-tunnel/config.example.yml` to your Cloudflare config directory and replace:

- `echo-worker.example.com`
- the tunnel UUID or name
- the credentials file path

Then create the DNS route and run the tunnel:

```powershell
cloudflared tunnel route dns echo-mainline-worker echo-worker.example.com
cloudflared tunnel --config "$env:USERPROFILE\.cloudflared\config.yml" run echo-mainline-worker
```

For an always-on Windows worker, install `cloudflared` as a Windows service after your tunnel works interactively. Keep the local Echo worker service running as well, because Cloudflare only forwards traffic to the local port.

## 4. Cloud Server Values

On Ubuntu, edit `/etc/echo/echo.env`:

```bash
ECHO_MAINLINE_WORKER_URL=https://echo-worker.example.com
ECHO_MAINLINE_WORKER_TOKEN=the-same-long-random-secret
ECHO_MAINLINE_WORKER_TIMEOUT_SECONDS=900
```

Restart the cloud service:

```bash
sudo systemctl restart echo
sudo journalctl -u echo -f
```

The cloud server will keep catalog browsing, anonymous sessions, favorites, and static pages local. It forwards recommendation and upload-analysis requests to your local worker only when needed.

## 5. Troubleshooting

- Cloud returns worker unavailable: check that the local worker and `cloudflared` are both running.
- Worker returns 401: the cloud `ECHO_MAINLINE_WORKER_TOKEN` does not match local `ECHO_WORKER_SHARED_TOKEN`.
- Quick Tunnel does not start: remove or rename an existing `.cloudflared/config.yml`, or use a named tunnel.
- Upload recommendations time out: increase `ECHO_MAINLINE_WORKER_TIMEOUT_SECONDS` on the cloud server.

## References

- Cloudflare Tunnel overview: https://developers.cloudflare.com/tunnel/
- Quick Tunnels: https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/
- Locally-managed named tunnels: https://developers.cloudflare.com/tunnel/advanced/local-management/create-local-tunnel/
- Windows service mode: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/local-management/as-a-service/windows/
