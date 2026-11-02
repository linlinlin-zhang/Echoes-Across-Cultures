"""Supervise Echo, two isolated reverse forwards, and health-based recovery."""

from __future__ import annotations
import argparse
import ctypes
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import urllib.request


def stop_tree(child):
    if child.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            [str(Path(os.environ["WINDIR"]) / "System32/taskkill.exe"), "/PID", str(child.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        child.terminate()
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def healthy(url, timeout=3):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Echo-Tunnel-Watchdog/2"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status == 200 and json.loads(response.read(2048)).get("ok") is True
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", default="root@47.250.196.219")
    parser.add_argument("--ssh-fallback-host", default="")
    parser.add_argument("--ssh-key", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--remote-port", type=int, default=18020)
    parser.add_argument("--backup-remote-port", type=int, default=18021)
    parser.add_argument("--env-file", default="configs/local_web.env")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    runtime = root / "tmp/intranet-web"
    runtime.mkdir(parents=True, exist_ok=True)
    lock = (runtime / "supervisor.lock").open("a+b")
    lock.seek(0)
    lock.write(b"0")
    lock.flush()
    lock.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("Echo intranet supervisor is already running")
    handler = RotatingFileHandler(runtime / "supervisor.log", maxBytes=2_000_000, backupCount=2)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log = logging.getLogger("echo-intranet")
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    env = os.environ.copy()
    for line in (root / args.env_file).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            env[name.strip()] = value.strip()
    if env.get("ECHO_WORKER_REQUIRE_TOKEN", "").lower() != "false":
        raise SystemExit("The full website requires ECHO_WORKER_REQUIRE_TOKEN=false")
    if env.get("ECHO_MAINLINE_WORKER_URL"):
        raise SystemExit("The local website must not proxy to another worker")
    key = Path(args.ssh_key).expanduser().resolve()
    if not key.is_file():
        raise SystemExit("SSH private key does not exist")
    ssh = str(Path(os.environ["WINDIR"]) / "System32/OpenSSH/ssh.exe") if os.name == "nt" else "ssh"
    fallback = args.ssh_fallback_host or env.get("ECHO_SSH_FALLBACK_HOST", "").strip()
    hosts = list(dict.fromkeys(h for h in [args.ssh_host, fallback] if h))
    host_index = 0
    common = [
        ssh,
        "-F",
        "none",
        "-N",
        "-T",
        "-i",
        str(key),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        f"HostKeyAlias={args.ssh_host.rsplit('@', 1)[-1]}",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=2",
        "-o",
        "TCPKeepAlive=yes",
    ]
    commands = {
        "web": [
            sys.executable,
            "-m",
            "uvicorn",
            "dcas_server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--proxy-headers",
            "--forwarded-allow-ips",
            "127.0.0.1",
            "--timeout-keep-alive",
            "15",
            "--limit-concurrency",
            "128",
        ],
        "tunnel": common + ["-R", f"127.0.0.1:{args.remote_port}:127.0.0.1:{args.port}", args.ssh_host],
    }
    if fallback:
        commands["tunnel-backup"] = common + [
            "-R",
            f"127.0.0.1:{args.backup_remote_port}:127.0.0.1:{args.port}",
            fallback,
        ]
    children, streams, failures, due, started, restarts = {}, {}, {}, {}, {}, {}
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    def atomic_status(name, value):
        target = runtime / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, target)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    (runtime / "supervisor.pid").write_text(str(os.getpid()))
    (runtime / "stop").unlink(missing_ok=True)
    awake = os.name == "nt" and env.get("ECHO_KEEP_AWAKE", "true").lower() in {"true", "1", "yes"}
    if awake:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    local_failures = public_failures = 0
    last_probe = 0
    next_public_probe = 0
    local_ok = public_ok = None
    last_success = None
    try:
        log.info("Starting Echo with %s reverse forwards; keep-awake=%s", len(commands) - 1, awake)
        while not stopping and not (runtime / "stop").exists():
            now = time.monotonic()
            for name, command in commands.items():
                child = children.get(name)
                if child is not None and child.poll() is None:
                    if now - started[name] > 60:
                        failures[name] = 0
                    continue
                if child is not None:
                    log.warning("%s exited with code %s", name, child.returncode)
                    children.pop(name, None)
                    streams.pop(name).close()
                    failures[name] = min(failures.get(name, 0) + 1, 5)
                    due[name] = now + min(60, 2 ** failures[name]) * random.uniform(0.8, 1.2)
                    if name == "tunnel":
                        host_index = (host_index + 1) % len(hosts)
                if now < due.get(name, 0):
                    continue
                output = runtime / f"{name}.log"
                if output.exists() and output.stat().st_size > 10_000_000:
                    output.replace(runtime / f"{name}.previous.log")
                stream = output.open("ab", buffering=0)
                launch = list(command)
                if name == "tunnel":
                    launch[-1] = hosts[host_index]
                try:
                    child = subprocess.Popen(
                        launch,
                        cwd=root,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                except OSError as exc:
                    stream.close()
                    due[name] = now + 15
                    log.error("Cannot start %s: %s", name, exc)
                    continue
                children[name], streams[name], started[name] = child, stream, now
                restarts[name] = restarts.get(name, -1) + 1
                log.info(
                    "Started %s pid=%s%s", name, child.pid, f" via {launch[-1]}" if name.startswith("tunnel") else ""
                )
            if now - last_probe >= 15:
                last_probe = now
                local_ok = healthy(f"http://127.0.0.1:{args.port}/api/health")
                local_failures = 0 if local_ok else local_failures + 1
                if local_failures >= 6 and "web" in children and now - started["web"] > 90:
                    log.warning("Local health check repeatedly failed; restarting web process tree")
                    stop_tree(children["web"])
                    local_failures = 0
                if local_ok and now >= next_public_probe:
                    next_public_probe = now + 30
                    public_ok = healthy("https://resonance.website/api/health", timeout=10)
                    public_failures = 0 if public_ok else public_failures + 1
                    if public_ok:
                        last_success = time.time()
                    if public_failures >= 3:
                        log.warning(
                            "Public origin health repeatedly failed while local app is healthy; renewing forwards"
                        )
                        for name in ("tunnel", "tunnel-backup"):
                            if name in children:
                                stop_tree(children[name])
                        public_failures = 0
                atomic_status(
                    "health.json",
                    json.dumps(
                        {
                            "time": time.time(),
                            "local_ok": local_ok,
                            "public_origin_ok": public_ok,
                            "last_public_success": last_success,
                            "restarts": restarts,
                            "pids": {n: c.pid for n, c in children.items()},
                        }
                    ),
                )
            atomic_status("children.pid", "\n".join(f"{name}={p.pid}" for name, p in children.items()))
            time.sleep(2)
    finally:
        for name, child in children.items():
            stop_tree(child)
            log.info("Stopped %s", name)
        for stream in streams.values():
            stream.close()
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        (runtime / "supervisor.pid").unlink(missing_ok=True)
        (runtime / "children.pid").unlink(missing_ok=True)
        lock.close()


if __name__ == "__main__":
    main()
