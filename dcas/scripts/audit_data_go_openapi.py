from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests


def _parse_extra(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"invalid --extra_param '{item}', expected KEY=VALUE")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def audit_data_go_openapi(
    service_url: str,
    service_key: str,
    out_dir: str | Path,
    pages: int = 1,
    page_start: int = 1,
    page_param: str = "pageNo",
    size_param: str = "numOfRows",
    page_size: int = 10,
    service_key_param: str = "serviceKey",
    format_param: str | None = None,
    format_value: str | None = None,
    extra_params: dict[str, str] | None = None,
    timeout_s: int = 60,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    extra = extra_params or {}

    responses: list[dict[str, Any]] = []
    session = requests.Session()

    for page in range(page_start, page_start + pages):
        params: dict[str, Any] = {
            service_key_param: service_key,
            page_param: page,
            size_param: page_size,
            **extra,
        }
        if format_param and format_value:
            params[format_param] = format_value

        r = session.get(service_url, params=params, timeout=timeout_s)
        payload_path = out_path / f"page_{page}.txt"
        payload_path.write_text(r.text, encoding="utf-8")
        responses.append(
            {
                "page": page,
                "status_code": r.status_code,
                "url": r.url.replace(service_key, "***SERVICE_KEY***"),
                "payload_file": str(payload_path.resolve()),
                "payload_bytes": len(r.content),
                "content_type": r.headers.get("content-type"),
            }
        )

    summary = {
        "service_url": service_url,
        "pages": pages,
        "page_start": page_start,
        "page_param": page_param,
        "size_param": size_param,
        "page_size": page_size,
        "service_key_param": service_key_param,
        "format_param": format_param,
        "format_value": format_value,
        "extra_params": extra,
        "responses": responses,
    }
    summary_path = out_path / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit a data.go.kr-style OpenAPI by saving raw sample pages.")
    ap.add_argument("--service_url", required=True)
    ap.add_argument("--service_key", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pages", type=int, default=1)
    ap.add_argument("--page_start", type=int, default=1)
    ap.add_argument("--page_param", default="pageNo")
    ap.add_argument("--size_param", default="numOfRows")
    ap.add_argument("--page_size", type=int, default=10)
    ap.add_argument("--service_key_param", default="serviceKey")
    ap.add_argument("--format_param", default=None)
    ap.add_argument("--format_value", default=None)
    ap.add_argument(
        "--extra_param",
        action="append",
        default=[],
        help="Extra query param as KEY=VALUE",
    )
    ap.add_argument("--timeout_s", type=int, default=60)
    args = ap.parse_args()

    summary = audit_data_go_openapi(
        service_url=args.service_url,
        service_key=args.service_key,
        out_dir=args.out_dir,
        pages=args.pages,
        page_start=args.page_start,
        page_param=args.page_param,
        size_param=args.size_param,
        page_size=args.page_size,
        service_key_param=args.service_key_param,
        format_param=args.format_param,
        format_value=args.format_value,
        extra_params=_parse_extra(args.extra_param),
        timeout_s=args.timeout_s,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
