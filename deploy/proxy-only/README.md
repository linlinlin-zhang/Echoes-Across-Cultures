# Echo Proxy-Only Cloud Server

This deployment shape keeps the real Echo app on the Windows machine. The cloud
server only accepts public HTTP/HTTPS traffic and forwards it over Tailscale to
the Windows host.

```text
visitor -> domain -> cloud Nginx/HTTPS -> Tailscale -> Windows Echo :8000
```

The cloud server does not need Python, model files, frontend files, or audio
assets. It only needs the files in this directory.

## 1. Windows Host

Install Tailscale and log in:

```powershell
winget install --id Tailscale.Tailscale -e
& "$env:ProgramFiles\Tailscale\tailscale.exe" up
& "$env:ProgramFiles\Tailscale\tailscale.exe" ip -4
```

Create the local private env file:

```powershell
Copy-Item configs/public_host.env.example configs/public_host.env
notepad configs/public_host.env
```

Start Echo so it listens on the Tailscale interface:

```powershell
.\scripts\run_public_host.ps1 -HostName 0.0.0.0 -Port 8000
```

Install optional local persistence:

```powershell
.\scripts\install_public_host_task.ps1
.\scripts\install_public_host_firewall.ps1
```

If the firewall script reports an access error, rerun PowerShell as
Administrator and execute the same command.

## 2. Cloud Server

Install Tailscale and join the same tailnet:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
tailscale status
```

Copy this directory to the cloud server, then create a config file:

```bash
cp echo-proxy.env.example echo-proxy.env
nano echo-proxy.env
```

Set at least:

```bash
ECHO_DOMAIN=your-domain.com
ECHO_EXTRA_DOMAINS=www.your-domain.com
ECHO_UPSTREAM=100.99.177.67:8000
ECHO_EMAIL=you@example.com
```

Install or update the proxy:

```bash
sudo bash install_echo_proxy.sh ./echo-proxy.env
```

If Tailscale is not logged in yet, the installer will print the login command.
Run it, finish the browser login, then rerun the installer:

```bash
sudo tailscale up --hostname echo-proxy
sudo bash install_echo_proxy.sh ./echo-proxy.env
```

Open only ports 80 and 443 to the public Internet. The Echo app port stays
private inside Tailscale. If you want the script to configure `ufw`, set
`ECHO_CONFIGURE_UFW=true` after confirming your SSH access is standard.

## 3. DNS

Point the public domain to the cloud server IP:

```text
A     your-domain.com      CLOUD_SERVER_IPV4
AAAA  your-domain.com      CLOUD_SERVER_IPV6   # optional
```

Do not point the public domain to the Windows home/office machine.

## 4. Validation

From the cloud server:

```bash
curl -I http://100.x.y.z:8000/api/health
curl -I https://your-domain.com/api/health
bash verify_echo_proxy.sh ./echo-proxy.env
```

From any browser:

```text
https://your-domain.com/
https://your-domain.com/music.html
```

If the first curl works but the second fails, debug Nginx or DNS. If the first
curl fails, debug Tailscale or the Windows firewall.

## Legacy One-Line Installer

The older argument-based wrapper still works:

```bash
sudo bash setup_ubuntu_tailscale_nginx.sh your-domain.com 100.99.177.67:8000 you@example.com
```
