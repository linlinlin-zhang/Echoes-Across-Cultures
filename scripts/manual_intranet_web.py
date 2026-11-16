"""Start, supervise, or stop Echo on demand without enabling login startup."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "tmp/intranet-web"
PUBLIC_URL = "https://resonance.website/music.html"


def try_lock(path):
    """Return an open locked file, or None if another process holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if os.fstat(handle.fileno()).st_size == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def running():
    handle = try_lock(RUNTIME / "supervisor.lock")
    if handle is None:
        return True
    handle.close()
    return False


def contain_windows_process_tree():
    """Keep a kernel-owned job alive until exit, including console-window closure."""
    if os.name != "nt":
        return None
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                     "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        error = ctypes.get_last_error()
        kernel.CloseHandle(job)
        raise ctypes.WinError(error)
    if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
        error = ctypes.get_last_error()
        kernel.CloseHandle(job)
        raise ctypes.WinError(error)
    # Do not close this handle explicitly: it includes this process. The OS closes
    # it on exit and terminates every inherited web/tunnel child, even after X.
    return job


def request_stop():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    (RUNTIME / "stop").write_text("Manual stop requested\n", encoding="utf-8")


def stop(timeout=50):
    if not running():
        print("Echo 已停止。", flush=True)
        return 0
    request_stop()
    print("正在停止网页应用和内网穿透，请稍候……", flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not running():
            print("Echo 已停止；下次需要时双击启动脚本。", flush=True)
            return 0
        time.sleep(0.5)
    print(f"停止尚未完成，请查看 {RUNTIME / 'supervisor.log'}", flush=True)
    return 1


def start(args):
    launcher_lock = try_lock(RUNTIME / "manual-launcher.lock")
    if launcher_lock is None or running():
        if launcher_lock is not None:
            launcher_lock.close()
        print(f"Echo 已在运行，无需重复启动。\n{PUBLIC_URL}", flush=True)
        return 0
    job = None
    child = None
    try:
        python = ROOT / ".venv-gpu/Scripts/python.exe" if os.name == "nt" else Path(sys.executable)
        key = Path(args.ssh_key).expanduser().resolve()
        for required in [python, key, ROOT / "configs/local_web.env", ROOT / "scripts/run_intranet_web.py"]:
            if not required.is_file():
                raise FileNotFoundError(f"缺少文件：{required}")
        job = contain_windows_process_tree()
        started = time.time()
        child = subprocess.Popen([str(python), str(ROOT / "scripts/run_intranet_web.py"),
                                  "--ssh-host", args.ssh_host, "--ssh-fallback-host", args.ssh_fallback_host,
                                  "--ssh-key", str(key)], cwd=ROOT)
        print("Echo 手动维护窗口\n正在启动应用、预热模型并连接云服务器……\n"
              "运行期间请保留此窗口；关闭窗口或双击停止脚本即可停止。\n"
              "应用异常会自动重启，隧道断线会自动重连。\n"
              f"日志：{RUNTIME}\n网页：{PUBLIC_URL}", flush=True)
        previous = None
        opened = False
        while child.poll() is None:
            try:
                health = json.loads((RUNTIME / "health.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                health = {}
            if health.get("time", 0) >= started:
                state = (health.get("local_ok"), health.get("public_origin_ok"))
                if state != previous:
                    message = "连接已建立，网页可以访问；模型首次预热可能还需稍候。" if all(state) else (
                        "本机应用已启动，正在连接公网隧道……" if state[0] else "应用正在启动或恢复，请稍候……")
                    print(time.strftime("[%H:%M:%S] ") + message, flush=True)
                    previous = state
                if all(state) and not opened and not args.no_browser:
                    webbrowser.open(PUBLIC_URL)
                    opened = True
            time.sleep(1)
        code = child.returncode
        print("Echo 维护已结束。" if code == 0 else f"Echo 启动失败（退出码 {code}），请检查日志。", flush=True)
        return code
    except KeyboardInterrupt:
        print("\n正在结束维护……", flush=True)
        return 0
    finally:
        if child is not None and child.poll() is None:
            request_stop()
            try:
                child.wait(timeout=50)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    subprocess.run([str(Path(os.environ["WINDIR"]) / "System32/taskkill.exe"),
                                    "/PID", str(child.pid), "/T", "/F"], check=False,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    child.terminate()
                child.wait(timeout=10)
        launcher_lock.close()
        # `job` is intentionally retained until the process exits (see above).
        _ = job


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("--ssh-host", default="echo-tunnel@47.250.196.219")
    parser.add_argument("--ssh-fallback-host", default="echo-tunnel@100.120.12.57")
    parser.add_argument("--ssh-key", default=str(Path.home() / ".ssh/echo_exhibition_ed25519"))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.action == "start":
        return start(args)
    if args.action == "stop":
        return stop()
    print("Echo 正在运行。" if running() else "Echo 已停止。", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"Echo：{exc}", file=sys.stderr)
        raise SystemExit(1)
