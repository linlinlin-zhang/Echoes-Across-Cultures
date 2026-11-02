"""Validate server-managed AI credential destinations without contacting a provider."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import HTTPException
from dcas_server.app import _kimi_request_endpoint
from dcas_server.schemas import KimiChatRequest


class EndpointPolicy(unittest.TestCase):
    configured = "https://api.moonshot.cn/v1/chat/completions"

    def request(self, **kwargs):
        return KimiChatRequest(messages=[{"role": "user", "content": "Endpoint policy check"}], **kwargs)

    def test_server_key_uses_configured_endpoint(self):
        for endpoint in [" ", self.configured]:
            with self.subTest(endpoint=endpoint):
                result = _kimi_request_endpoint(self.request(endpoint=endpoint), {"endpoint": self.configured})
                self.assertEqual(result, self.configured)

    def test_server_key_cannot_be_redirected(self):
        for endpoint in [
            "https://example.invalid/chat",
            self.configured + "?override=1",
            "https://api.moonshot.cn@example.invalid/v1/chat/completions",
        ]:
            for supplied_key in [None, "", "  "]:
                with self.subTest(endpoint=endpoint, supplied_key=supplied_key):
                    with self.assertRaises(HTTPException) as rejected:
                        _kimi_request_endpoint(
                            self.request(endpoint=endpoint, api_key=supplied_key), {"endpoint": self.configured}
                        )
                    self.assertEqual(rejected.exception.status_code, 400)

    def test_caller_key_can_choose_https_endpoint(self):
        endpoint = "https://example.invalid/chat"
        self.assertEqual(
            _kimi_request_endpoint(
                self.request(endpoint=endpoint, api_key="self-check-placeholder"), {"endpoint": self.configured}
            ),
            endpoint,
        )

    def test_http_is_rejected_even_with_caller_key(self):
        with self.assertRaises(HTTPException) as rejected:
            _kimi_request_endpoint(
                self.request(endpoint="http://example.invalid/chat", api_key="self-check-placeholder"), {}
            )
        self.assertEqual(rejected.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
