"""Offline public-policy, queue saturation and upload responsiveness checks; no upstream AI calls."""

from __future__ import annotations
import asyncio
import importlib
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def queue_test():
    from dcas_server.exhibition_security import ExhibitionGuard

    started, finish = asyncio.Event(), asyncio.Event()

    async def app(scope, receive, send):
        if scope["method"] == "POST":
            started.set()
            await finish.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    guard = ExhibitionGuard(app, enabled=True, queue_limit=1, queue_timeout=0.2)

    async def request(method="POST"):
        output = []

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(message):
            output.append(message)

        await guard(
            {
                "type": "http",
                "method": method,
                "scheme": "https",
                "path": "/api/mainline/recommend" if method == "POST" else "/api/health",
                "headers": [(b"host", b"resonance.website")],
            },
            receive,
            send,
        )
        return next(x["status"] for x in output if x["type"] == "http.response.start")

    first = asyncio.create_task(request())
    await started.wait()
    second = asyncio.create_task(request())
    await asyncio.sleep(0.02)
    assert await request() == 503
    assert await request("GET") == 200
    finish.set()
    assert await first == await second == 200
    assert await request() == 200
    assert guard.limits["compute"]._value == 1
    print("PASS bounded compute queue, overload response, independent health and slot release")


def main():
    from fastapi.testclient import TestClient

    with tempfile.TemporaryDirectory(prefix="echo-security-") as folder:
        with patch.dict(
            os.environ,
            {
                "ECHO_PUBLIC_MODE": "true",
                "ECHO_COOKIE_SECURE": "true",
                "ECHO_STORAGE_ROOT": folder,
                "ECHO_USER_DATA_DIR": folder + "/user_data",
                "ECHO_WORKER_REQUIRE_TOKEN": "false",
            },
        ):
            module = importlib.import_module("dcas_server.app")
            store = module.AnonymousFavoriteStore(Path(folder) / "batch-test.sqlite3")
            store.upsert_favorites("one", [{"track_id": "a", "title": "First"}, {"track_id": "b", "title": "Second"}])
            store.upsert_favorite("one", {"track_id": "a", "title": "Updated"})
            assert len(store.list_favorites("one")) == 2
            assert {x["track_id"]: x["title"] for x in store.list_favorites("one")} == {"a": "Updated", "b": "Second"}
            assert not store.list_favorites("two")
            try:
                store.upsert_favorites("two", [{"track_id": "valid"}, {"title": ""}])
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid batch should fail before writing")
            assert not store.list_favorites("two")
            print("PASS batch favorite upserts, session isolation and invalid-batch atomicity")
            with TestClient(module.create_app(), base_url="https://resonance.website") as client:
                assert client.get("/api/health").status_code == 200
                for route in [
                    "/api/train",
                    "/api/files?dir=models",
                    "/api/files/download?path=models/test.pt",
                    "/openapi.json",
                    "/api/prototype/state",
                    "/.env",
                ]:
                    assert client.get(route).status_code == 404, route
                assert client.get("/api/health", headers={"Host": "evil.example"}).status_code == 400
                assert (
                    client.post("/api/favorites", headers={"Origin": "https://evil.example"}, json={}).status_code
                    == 403
                )
                assert client.post("/api/ai/kimi/chat", content=b"x" * (1024 * 1024 + 1)).status_code == 413
                session = client.get("/api/session")
                assert session.status_code == 200 and "Secure" in session.headers["set-cookie"]
                assert session.headers["cache-control"] == "no-store"
                print("PASS public allowlist, host, cross-origin, body limit and secure private session")
                with patch.object(
                    module,
                    "_load_local_kimi_config",
                    return_value={
                        "api_key": "offline-dummy",
                        "model": "kimi-k2.6",
                        "endpoint": "https://api.moonshot.cn/v1/chat/completions",
                    },
                ):
                    result = client.post(
                        "/api/ai/kimi/chat",
                        json={"messages": [], "api_key": "dummy", "endpoint": "https://127.0.0.1/private"},
                    )
                    assert result.status_code == 400
                    result = client.post("/api/ai/kimi/chat", json={"messages": [], "model": "unapproved-model"})
                    assert result.status_code == 400
                    result = client.post(
                        "/api/ai/kimi/chat", json={"messages": [{"role": "user", "content": "x" * 24001}]}
                    )
                    assert result.status_code == 413
                print("PASS public AI endpoint/model restrictions and prompt bound, without API calls")
                event = threading.Event()

                def slow_proxy(**kwargs):
                    event.set()
                    time.sleep(0.7)
                    return {"ok": True}

                result = []
                with (
                    patch.object(module, "_mainline_worker_url", return_value="https://offline.invalid"),
                    patch.object(module, "_proxy_worker_upload_recommend", side_effect=slow_proxy),
                    patch.object(module, "_hydrate_worker_mainline_metadata", side_effect=lambda data, storage: data),
                ):
                    thread = threading.Thread(
                        target=lambda: result.append(
                            client.post(
                                "/api/mainline/upload_recommend",
                                files={"file": ("sample.wav", b"RIFF" + b"0" * 80, "audio/wav")},
                            )
                        )
                    )
                    thread.start()
                    assert event.wait(timeout=5)
                    before = time.monotonic()
                    response = client.get("/api/health")
                    elapsed = time.monotonic() - before
                    thread.join()
                    assert result[0].status_code == 200
                    assert response.status_code == 200 and elapsed < 0.4, elapsed
                    print(f"PASS health remains responsive during upload work ({elapsed:.3f}s)")
    asyncio.run(queue_test())


if __name__ == "__main__":
    main()
