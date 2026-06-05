from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_MODEL_ID = "ntua-slp/CultureMERT-95M"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _bool_env(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify the CultureMERT model used by the local Echo worker."
    )
    parser.add_argument("--env-file", default="configs/local_worker.env")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--revision", default="")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    _load_env_file(Path(args.env_file))

    model_id = args.model_id or os.environ.get("ECHO_CULTUREMERT_MODEL_ID") or DEFAULT_MODEL_ID
    cache_dir = args.cache_dir or os.environ.get("ECHO_CULTUREMERT_CACHE_DIR") or None
    revision = args.revision or os.environ.get("ECHO_CULTUREMERT_REVISION") or None
    local_files_only = bool(args.local_files_only or _bool_env("ECHO_CULTUREMERT_LOCAL_FILES_ONLY", False))

    from transformers import AutoFeatureExtractor, AutoModel

    kwargs = {"trust_remote_code": True, "local_files_only": local_files_only}
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    if revision:
        kwargs["revision"] = str(revision)

    print(f"model_id={model_id}")
    print(f"cache_dir={cache_dir or '(default Hugging Face cache)'}")
    print(f"revision={revision or '(default)'}")
    print(f"local_files_only={local_files_only}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, **kwargs)
    print(f"feature_extractor={feature_extractor.__class__.__name__}")
    model = AutoModel.from_pretrained(model_id, **kwargs)
    print(f"model={model.__class__.__name__}")

    if not local_files_only:
        print("Verifying cached offline load...")
        offline_kwargs = dict(kwargs)
        offline_kwargs["local_files_only"] = True
        AutoFeatureExtractor.from_pretrained(model_id, **offline_kwargs)
        AutoModel.from_pretrained(model_id, **offline_kwargs)
        print("offline_load=ok")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
