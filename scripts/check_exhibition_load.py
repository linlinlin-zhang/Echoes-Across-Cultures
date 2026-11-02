"""Twenty independent visitors: pages, catalog, playback and private favorites; no paid AI."""

from concurrent.futures import ThreadPoolExecutor
import httpx
import json
import os
from pathlib import Path
import statistics
import threading
import time

BASE = os.environ.get("ECHO_TEST_URL", "https://resonance.website")
OUT = Path(os.environ.get("ECHO_CHECK_OUTPUT", "tmp/exhibition-20261102"))
barrier = threading.Barrier(20)


def visitor(index):
    client = httpx.Client(http2=True, timeout=45, limits=httpx.Limits(max_connections=1, max_keepalive_connections=1))
    timings, failures, routes = [], [], []

    def fetch(path, expected=200, audio=False):
        start = time.monotonic()
        response = client.get(BASE + path, headers={"Range": "bytes=0-8191"} if audio else {})
        body = response.content
        assert response.status_code == expected, (path, response.status_code)
        assert response.http_version == "HTTP/2", response.http_version
        if audio:
            assert len(body) == 8192 and response.headers.get("X-Echo-Edge") == "cloud-audio"
        if path.startswith("/api/mainline/") and not audio:
            assert response.headers.get("X-Echo-Edge") == "cloud"
        routes.append({"path": path, "ms": round((time.monotonic() - start) * 1000, 1)})
        timings.append(routes[-1]["ms"])
        return response.json() if "application/json" in response.headers.get("Content-Type", "") else body

    barrier.wait()
    session = None
    try:
        fetch("/music.html")
        session = fetch("/api/session")["session_id"]
        for iteration in range(2):
            result = fetch(f"/api/mainline/catalog?limit=8&random_seed={9000 + index * 2 + iteration}")
            track = result["items"][0]
            fetch("/api/mainline/tracks/" + track["track_id"])
            fetch("/api/mainline/audio/" + track["track_id"], 206, True)
        favorites = fetch("/api/favorites")
        assert favorites.get("items"), "New anonymous session should have its own initial favorites"
    except Exception as error:
        failures.append(str(error))
    client.close()
    return {"visitor": index, "session_id": session, "timings_ms": timings, "routes": routes, "failures": failures}


def main():
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=20) as pool:
        visitors = list(pool.map(visitor, range(20)))
    failures = [v for v in visitors if v["failures"]]
    times = sorted(t for v in visitors for t in v["timings_ms"])
    report = {
        "base": BASE,
        "concurrent_visitors": 20,
        "transport": "HTTP/2, one persistent connection per visitor",
        "requests_passed": len(times),
        "unique_private_sessions": len({v["session_id"] for v in visitors if v["session_id"]}),
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "response_ms": {
            "median": statistics.median(times) if times else None,
            "p95": times[int((len(times) - 1) * 0.95)] if times else None,
            "max": max(times) if times else None,
        },
        "failures": failures,
        "visitors": visitors,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "load-validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "visitors"}, ensure_ascii=False))
    assert not failures and report["unique_private_sessions"] == 20


if __name__ == "__main__":
    main()
