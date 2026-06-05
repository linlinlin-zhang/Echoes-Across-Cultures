from __future__ import annotations

import argparse
import ast
import builtins
import csv
import json
import platform
import re
import subprocess
import sys
import symtable
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOTS = ["dcas", "dcas_server", "scripts"]
FIGURE_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
RESULT_EXTS = {".json", ".csv", ".log"}
SCALAR_TYPES = (str, int, float, bool, type(None))
STD_LIBS = set(getattr(sys, "stdlib_module_names", set()))
IMPORT_TO_PACKAGE = {
    "PIL": "Pillow",
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "huggingface_hub": "huggingface-hub",
    "multipart": "python-multipart",
    "sklearn": "scikit-learn",
    "soundfile": "soundfile",
    "yaml": "PyYAML",
}
PACKAGE_VERSION_IMPORTS = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "sklearn",
    "torch",
    "torchaudio",
    "transformers",
    "datasets",
    "huggingface_hub",
    "pyarrow",
    "fsspec",
    "xgboost",
    "lightgbm",
    "fastapi",
    "uvicorn",
]


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def _load_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8-sig", errors="ignore")
        except Exception:
            return None
    except Exception:
        return None


def _load_json(path: Path) -> Any:
    text = _load_text(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _git_tracked_files() -> set[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return {line.strip().replace("\\", "/") for line in out.stdout.splitlines() if line.strip()}
    except Exception:
        return set()


def _git_head() -> dict[str, str]:
    branch = ""
    commit = ""
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
    except Exception:
        pass
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
    except Exception:
        pass
    return {"branch": branch, "commit": commit}


def _flatten_json(obj: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_json(value, next_prefix)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]"
            yield from _flatten_json(value, next_prefix)
    else:
        yield prefix, obj


def _normalize_package_name(name: str) -> str:
    return name.lower().replace("_", "-")


def _requirements_declared_packages() -> set[str]:
    req = REPO_ROOT / "requirements.txt"
    declared: set[str] = set()
    text = _load_text(req) or ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pkg = re.split(r"[<>=!~\\[]", line, maxsplit=1)[0].strip()
        if pkg:
            declared.add(_normalize_package_name(pkg))
    return declared


def _top_level_dir_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(REPO_ROOT.iterdir(), key=lambda p: p.name.lower()):
        if path.name.startswith(".") and path.name not in {".gitignore"}:
            continue
        if path.is_dir():
            try:
                file_count = sum(1 for _ in path.rglob("*") if _.is_file())
            except Exception:
                file_count = None
            rows.append({"path": _rel(path), "kind": "dir", "file_count": file_count})
        else:
            rows.append({"path": _rel(path), "kind": "file", "file_count": 1})
    return rows


def _module_name_from_path(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    return ".".join(parts)


def _parse_python_imports(path: Path, module_name: str) -> dict[str, Any]:
    source = _load_text(path)
    if source is None:
        return {"imports": [], "external_roots": [], "syntax_ok": False}
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return {"imports": [], "external_roots": [], "syntax_ok": False}
    imports: list[str] = []
    external_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
                external_roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                package_parts = module_name.split(".")[:-1]
                ascend = max(node.level - 1, 0)
                base_parts = package_parts[: len(package_parts) - ascend] if ascend <= len(package_parts) else []
                if node.module:
                    full = ".".join(base_parts + node.module.split("."))
                    if full:
                        imports.append(full)
                        external_roots.add(full.split(".", 1)[0])
                else:
                    for alias in node.names:
                        full = ".".join(base_parts + [alias.name])
                        if full:
                            imports.append(full)
                            external_roots.add(full.split(".", 1)[0])
            elif node.module:
                imports.append(node.module)
                external_roots.add(node.module.split(".", 1)[0])
    return {
        "imports": imports,
        "external_roots": sorted(external_roots),
        "syntax_ok": True,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _collect_python_modules() -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    module_rows: list[dict[str, Any]] = []
    module_map: dict[str, Path] = {}
    for root in CODE_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if any(part in {"__pycache__", ".venv-gpu", "node_modules"} for part in path.parts):
                continue
            module_name = _module_name_from_path(path)
            module_map[module_name] = path
            parsed = _parse_python_imports(path, module_name)
            module_rows.append(
                {
                    "path": _rel(path),
                    "module": module_name,
                    "imports": parsed["imports"],
                    "external_roots": parsed["external_roots"],
                    "syntax_ok": parsed["syntax_ok"],
                }
            )

    def resolve_internal(import_name: str) -> str | None:
        parts = import_name.split(".")
        for i in range(len(parts), 0, -1):
            candidate = ".".join(parts[:i])
            if candidate in module_map:
                return candidate
        return None

    edges: list[dict[str, str]] = []
    incoming = Counter()
    for row in module_rows:
        seen: set[str] = set()
        for import_name in row["imports"]:
            if not import_name.startswith(("dcas", "dcas_server", "scripts")):
                continue
            target = resolve_internal(import_name)
            if not target or target in seen:
                continue
            seen.add(target)
            incoming[target] += 1
            edges.append({"source": row["module"], "target": target})

    graph_summary = {
        "n_python_modules": len(module_rows),
        "n_internal_edges": len(edges),
        "top_incoming_modules": [
            {"module": module, "incoming_edges": count} for module, count in incoming.most_common(15)
        ],
    }
    return module_rows, edges, graph_summary


def _collect_external_import_findings(
    module_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    tracked_modules = {row["module"].split(".", 1)[0] for row in module_rows}
    declared = _requirements_declared_packages()
    external_roots = Counter()
    missing = Counter()
    for row in module_rows:
        for root in row["external_roots"]:
            if root in tracked_modules or root in STD_LIBS or root.startswith("_"):
                continue
            external_roots[root] += 1
            pkg = _normalize_package_name(IMPORT_TO_PACKAGE.get(root, root))
            if pkg not in declared:
                missing[pkg] += 1
    return {
        "external_import_frequency": dict(sorted(external_roots.items())),
        "candidate_undeclared_packages": [{"package": pkg, "count": count} for pkg, count in missing.most_common()],
    }


def _compile_check() -> dict[str, Any]:
    checked: list[str] = []
    failed: list[str] = []
    for root in CODE_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if any(part in {"__pycache__", ".venv-gpu", "node_modules"} for part in path.parts):
                continue
            checked.append(_rel(path))
            try:
                source = _load_text(path)
                if source is None:
                    raise ValueError("unreadable")
                compile(source, str(path), "exec")
            except Exception:
                failed.append(_rel(path))
    return {"checked": len(checked), "failed": failed}


def _undefined_name_scan() -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    builtin_names = set(dir(builtins))
    for root in CODE_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if any(part in {"__pycache__", ".venv-gpu", "node_modules"} for part in path.parts):
                continue
            source = _load_text(path)
            if source is None:
                continue
            try:
                table = symtable.symtable(source, str(path), "exec")
            except Exception:
                continue
            stack = [table]
            while stack:
                cur = stack.pop()
                stack.extend(cur.get_children())
                for symbol in cur.get_symbols():
                    name = symbol.get_name()
                    if name in builtin_names:
                        continue
                    if symbol.is_referenced() and not (
                        symbol.is_parameter()
                        or symbol.is_local()
                        or symbol.is_global()
                        or symbol.is_nonlocal()
                        or symbol.is_imported()
                        or symbol.is_free()
                        or symbol.is_assigned()
                    ):
                        findings.append({"path": _rel(path), "scope": cur.get_name(), "name": name})
    return findings


def _hardcoded_path_scan() -> list[dict[str, Any]]:
    patterns = [
        re.compile(r"(?<![A-Za-z])[A-Za-z]:\\\\"),
        re.compile(r"(?<![A-Za-z])[A-Za-z]:/"),
    ]
    findings: list[dict[str, Any]] = []
    for root in CODE_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if any(part in {"__pycache__", ".venv-gpu", "node_modules"} for part in path.parts):
                continue
            text = _load_text(path)
            if not text:
                continue
            matches: list[str] = []
            for pattern in patterns:
                matches.extend(pattern.findall(text))
            if matches:
                findings.append({"path": _rel(path), "n_matches": len(matches)})
    return findings


def _tracked_run_status(rel_path: str, tracked: set[str]) -> str:
    return "tracked" if rel_path in tracked else "local_or_ignored"


def _config_category(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT / "configs").parts[0]
    except Exception:
        return "other"


def _config_flavor(name: str) -> str:
    if name.endswith(".run.json"):
        return "run"
    if name.endswith(".example.json"):
        return "example"
    if ".local." in name or name.endswith(".local.json"):
        return "local"
    if ".tmp." in name or name.endswith(".tmp.json"):
        return "tmp"
    return "other"


def _collect_configs(
    tracked_files: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config_rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    category_counter = Counter()
    run_status_counter = Counter()
    for path in sorted((REPO_ROOT / "configs").rglob("*.json")):
        rel_path = _rel(path)
        category = _config_category(path)
        flavor = _config_flavor(path.name)
        tracked_status = _tracked_run_status(rel_path, tracked_files)
        payload = _load_json(path)
        flat = list(_flatten_json(payload)) if payload is not None else []
        scalar_flat = [(k, v) for k, v in flat if isinstance(v, SCALAR_TYPES)]
        top_level_keys = sorted(payload.keys()) if isinstance(payload, dict) else []
        seed_keys = [k for k, _ in scalar_flat if "seed" in k.lower()]
        hardware_keys = [
            (k, v)
            for k, v in scalar_flat
            if any(token in k.lower() for token in ("cuda", "device", "vertex", "worker"))
        ]
        config_rows.append(
            {
                "path": rel_path,
                "category": category,
                "flavor": flavor,
                "tracked_status": tracked_status,
                "top_level_keys": ";".join(top_level_keys),
                "n_scalar_params": len(scalar_flat),
                "seed_keys": ";".join(seed_keys),
                "hardware_keys": ";".join(f"{k}={v}" for k, v in hardware_keys[:12]),
            }
        )
        category_counter[category] += 1
        run_status_counter[f"{category}:{tracked_status}"] += 1
        for key, value in scalar_flat:
            param_rows.append(
                {
                    "config_path": rel_path,
                    "category": category,
                    "flavor": flavor,
                    "param_key": key,
                    "param_value": json.dumps(value, ensure_ascii=False),
                }
            )
    return (
        config_rows,
        param_rows,
        {
            "config_count_by_category": dict(sorted(category_counter.items())),
            "tracked_status_by_category": dict(sorted(run_status_counter.items())),
        },
    )


def _experiment_settings_table(
    config_rows: list[dict[str, Any]], param_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_config: dict[str, dict[str, str]] = {}
    for row in param_rows:
        by_config.setdefault(row["config_path"], {})[row["param_key"]] = row["param_value"]
    selected_keys = [
        "suite_name",
        "data",
        "tracks",
        "interactions",
        "constraints",
        "out",
        "model_id",
        "window_count",
        "max_seconds",
        "output_dimensionality",
        "epochs",
        "batch_size",
        "lr",
        "seed",
        "bootstrap_seed",
        "prefer_cuda",
        "device",
        "latent_dim",
        "hidden_dim",
        "negative_samples",
        "recall_k",
        "lambda_constraints",
        "lambda_rank",
        "lambda_domain",
        "lambda_contrast",
        "lambda_cov",
        "lambda_tc",
        "lambda_hsic",
        "lambda_source",
        "ranking_negatives",
        "constraint_margin",
        "vertexai",
        "vertex_project",
        "vertex_location",
        "api_key_file",
    ]
    summary_rows: list[dict[str, Any]] = []
    for cfg in config_rows:
        params = by_config.get(cfg["path"], {})
        row: dict[str, Any] = {
            "config_path": cfg["path"],
            "category": cfg["category"],
            "flavor": cfg["flavor"],
            "tracked_status": cfg["tracked_status"],
        }
        for needle in selected_keys:
            value = ""
            for key, raw in params.items():
                if key == needle or key.endswith(f".{needle}"):
                    value = raw
                    break
            row[needle] = value
        summary_rows.append(row)
    return summary_rows


def _figure_type_from_name(name: str) -> str:
    low = name.lower()
    if "heatmap" in low:
        return "heatmap"
    if "hist" in low:
        return "histogram"
    if "pca" in low:
        return "embedding_visualization"
    if "frontier" in low or "metric_grid" in low or "coverage_vs" in low:
        return "benchmark_plot"
    if "flow" in low or "pipeline" in low:
        return "flow_diagram"
    if "counts" in low or "distribution" in low or "scale_overview" in low:
        return "distribution_plot"
    return "other"


def _collect_figures() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from PIL import Image
    except Exception:
        Image = None

    rows: list[dict[str, Any]] = []
    ext_counter = Counter()
    type_counter = Counter()
    for path in sorted((REPO_ROOT / "reports" / "figures").rglob("*")):
        if not path.is_file() or path.suffix.lower() not in FIGURE_EXTS:
            continue
        ext = path.suffix.lower().lstrip(".")
        ext_counter[ext] += 1
        figure_type = _figure_type_from_name(path.name)
        type_counter[figure_type] += 1
        width = None
        height = None
        issue_flags: list[str] = []
        if ext in {"png", "jpg", "jpeg"} and Image is not None:
            try:
                with Image.open(path) as img:
                    width, height = img.size
                if (width or 0) < 1200 or (height or 0) < 800:
                    issue_flags.append("low_raster_resolution")
                if ext in {"jpg", "jpeg"}:
                    issue_flags.append("lossy_plot_format")
            except Exception:
                issue_flags.append("unreadable_raster")
        rows.append(
            {
                "path": _rel(path),
                "suite": path.parent.name,
                "extension": ext,
                "figure_type": figure_type,
                "width": width,
                "height": height,
                "issue_flags": ";".join(issue_flags),
            }
        )
    return rows, {
        "n_figures": len(rows),
        "count_by_extension": dict(sorted(ext_counter.items())),
        "count_by_type": dict(sorted(type_counter.items())),
    }


def _choose_metadata_file(dataset_dir: Path) -> Path | None:
    priorities = [
        "metadata_release.csv",
        "metadata_release_culturemert_mw3.csv",
        "metadata_release_gemini_embedding2_mw3.csv",
        "metadata_v4_main.csv",
        "metadata_v3_main_harmonized.csv",
        "metadata_v3_main_harmonized_mw3.csv",
        "metadata_v2_main_clean.csv",
        "metadata_v2_main.csv",
        "metadata_merged.csv",
        "metadata.csv",
        "metadata_clean.csv",
        "metadata_harmonized.csv",
        "metadata_raw.csv",
    ]
    for name in priorities:
        candidate = dataset_dir / name
        if candidate.exists():
            return candidate
    all_candidates = sorted(dataset_dir.glob("metadata*.csv"))
    return all_candidates[0] if all_candidates else None


def _dataset_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    path = str(row.get("dataset_dir", ""))
    if "research_dataset_v4/main" in path:
        return (0, path)
    if "research_dataset_v4/routeA_small" in path:
        return (1, path)
    if "research_dataset_v3" in path:
        return (2, path)
    if "routeA_phase2_cn" in path:
        return (3, path)
    if "yambda_5b_subset" in path:
        return (4, path)
    return (9, path)


def _csv_profile(path: Path) -> dict[str, Any]:
    profile = {
        "metadata_file": _rel(path),
        "metadata_rows": 0,
        "n_columns": 0,
        "culture_unique": None,
        "source_unique": None,
        "top_cultures": "",
        "top_sources": "",
    }
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            culture_counter = Counter()
            source_counter = Counter()
            n_rows = 0
            fieldnames = list(reader.fieldnames or [])
            for row in reader:
                n_rows += 1
                culture = str(row.get("culture", "")).strip()
                source = str(row.get("source_dataset", "")).strip()
                if culture:
                    culture_counter[culture] += 1
                if source:
                    source_counter[source] += 1
        profile["metadata_rows"] = n_rows
        profile["n_columns"] = len(fieldnames)
        if culture_counter:
            profile["culture_unique"] = len(culture_counter)
            profile["top_cultures"] = ";".join(f"{k}:{v}" for k, v in culture_counter.most_common(5))
        if source_counter:
            profile["source_unique"] = len(source_counter)
            profile["top_sources"] = ";".join(f"{k}:{v}" for k, v in source_counter.most_common(5))
    except Exception:
        profile["metadata_rows"] = None
    return profile


def _collect_datasets() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    top_dataset_counter = Counter()
    for path in sorted((REPO_ROOT / "storage" / "public").rglob("*.manifest.json")):
        rel = _rel(path)
        if ".cache/" in rel or ".cache\\" in rel:
            continue
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        if not rel.endswith(".npz.manifest.json"):
            continue
        dataset_dir = path.parent
        metadata_file = _choose_metadata_file(dataset_dir)
        profile = (
            _csv_profile(metadata_file)
            if metadata_file
            else {
                "metadata_file": "",
                "metadata_rows": None,
                "n_columns": None,
                "culture_unique": None,
                "source_unique": None,
                "top_cultures": "",
                "top_sources": "",
            }
        )
        top_dataset = rel.split("/")[2] if rel.startswith("storage/public/") else dataset_dir.name
        top_dataset_counter[top_dataset] += 1
        rows.append(
            {
                "manifest_path": rel,
                "dataset_dir": _rel(dataset_dir),
                "top_dataset": top_dataset,
                "track_artifact": path.name.replace(".manifest.json", ""),
                "model_id": payload.get("model_id"),
                "n_tracks": payload.get("n_tracks"),
                "dim": payload.get("dim"),
                "max_seconds": payload.get("max_seconds"),
                "window_count": payload.get("window_count", 1),
                "n_errors": payload.get("n_errors", len(payload.get("errors", []))),
                **profile,
            }
        )
    return rows, {"manifest_rows_by_top_dataset": dict(sorted(top_dataset_counter.items()))}


def _classify_result_path(rel_path: str) -> tuple[str, str]:
    low = rel_path.lower()
    if "reports/benchmarks/v4_main_" in low and "stage3_lambdamart" in low:
        return "main_experiment", "complete_candidate"
    if "reports/benchmarks/v4_routea_small_" in low and "stage3_lambdamart" in low:
        return "main_experiment_small", "complete_candidate"
    if "reports/benchmarks/public_routea" in low or "reports/benchmarks/yambda" in low:
        return "external_or_public_benchmark", "complete_candidate"
    if "reports/benchmarks/" in low:
        return "benchmark_support", "complete_candidate"
    if "reports/ablation/" in low or "ablation_" in low:
        return "ablation", "complete_candidate"
    if "reports/baseline_comparison/" in low:
        return "baseline_comparison", "complete_candidate"
    if "reports/datasets/" in low:
        return "dataset_audit", "complete_candidate"
    if "reports/pal/" in low:
        return "pal", "complete_candidate"
    if any(token in low for token in ("smoke", "tmp", "probe", "cache")):
        return "smoke_or_partial", "partial_or_failed"
    return "other", "unknown"


def _load_json_failure_signals(path: Path) -> str:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return ""
    if int(payload.get("n_errors", 0) or 0) > 0:
        return f"n_errors={int(payload.get('n_errors', 0) or 0)}"
    errors = payload.get("errors")
    if isinstance(errors, list) and errors:
        return f"errors={len(errors)}"
    return ""


def _collect_results() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    category_counter = Counter()
    for path in sorted((REPO_ROOT / "reports").rglob("*")):
        if not path.is_file() or path.suffix.lower() not in RESULT_EXTS:
            continue
        rel_path = _rel(path)
        category, status_guess = _classify_result_path(rel_path)
        failure_signal = _load_json_failure_signals(path) if path.suffix.lower() == ".json" else ""
        rows.append(
            {
                "path": rel_path,
                "extension": path.suffix.lower().lstrip("."),
                "category": category,
                "status_guess": status_guess,
                "failure_signal": failure_signal,
            }
        )
        category_counter[category] += 1
    return rows, {"result_count_by_category": dict(sorted(category_counter.items()))}


def _collect_partial_failure_assets() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scan_roots = [REPO_ROOT / "reports", REPO_ROOT / "storage" / "public"]
    for root in scan_roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in RESULT_EXTS:
                continue
            rel_path = _rel(path)
            low = rel_path.lower()
            if ".cache/" in low or ".cache\\" in low:
                continue
            payload = _load_json(path) if path.suffix.lower() == ".json" else None
            n_errors = ""
            if isinstance(payload, dict) and int(payload.get("n_errors", 0) or 0) > 0:
                n_errors = str(int(payload.get("n_errors", 0) or 0))
            if any(token in low for token in ("smoke", "probe", "tmp")) or n_errors:
                rows.append(
                    {
                        "path": rel_path,
                        "root": "reports" if str(path).startswith(str(REPO_ROOT / "reports")) else "storage/public",
                        "extension": path.suffix.lower().lstrip("."),
                        "status_tag": "n_errors" if n_errors else "smoke_or_probe",
                        "n_errors": n_errors,
                    }
                )
    return rows


def _package_version(name: str) -> str | None:
    try:
        mod = __import__(name)
        return str(getattr(mod, "__version__", None))
    except Exception:
        return None


def _collect_environment(config_param_rows: list[dict[str, Any]]) -> dict[str, Any]:
    versions = {name: _package_version(name) for name in PACKAGE_VERSION_IMPORTS}
    seed_counter = Counter()
    for row in config_param_rows:
        if "seed" in row["param_key"].lower():
            seed_counter[row["param_value"]] += 1
    git_head = _git_head()
    cuda_available = False
    cuda_version = None
    device_name = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_branch": git_head["branch"],
        "git_commit": git_head["commit"],
        "package_versions": versions,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "cuda_device_0": device_name,
        "seed_frequency": dict(sorted(seed_counter.items())),
    }


def _core_file_inventory(tracked_files: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    curated = [
        (
            "dcas/models/dcas_vae.py",
            "core_method",
            "Method",
            "Main disentanglement / recommendation backbone implementation.",
        ),
        (
            "dcas/recommender.py",
            "core_method",
            "Method",
            "Core recommendation scoring and evaluation-facing inference entry.",
        ),
        (
            "dcas/embedding_recommenders.py",
            "baseline_family",
            "Method",
            "Industrial-style embedding, KNN, cosine, BPR and hybrid recommenders.",
        ),
        (
            "dcas/pipelines.py",
            "pipeline_orchestration",
            "Method",
            "Shared wiring between data, training, evaluation, and PAL stages.",
        ),
        (
            "dcas/scripts/build_research_dataset_v4.py",
            "dataset_pipeline",
            "Dataset",
            "End-to-end V4 dataset build orchestration.",
        ),
        (
            "dcas/scripts/harmonize_v4_metadata.py",
            "dataset_pipeline",
            "Dataset",
            "Metadata normalization and field alignment for V4.",
        ),
        (
            "dcas/scripts/build_tracks_from_audio.py",
            "embedding_pipeline",
            "Method",
            "Audio-to-embedding builder for CultureMERT and related backbones.",
        ),
        (
            "dcas/scripts/build_tracks_with_gemini.py",
            "embedding_pipeline",
            "Method",
            "Gemini embedding extraction pipeline with API/window controls.",
        ),
        (
            "dcas/scripts/synthesize_interactions.py",
            "dataset_pipeline",
            "Dataset",
            "Synthetic interaction generation used by released benchmark datasets.",
        ),
        (
            "dcas/scripts/run_train_from_json.py",
            "experiment_runner",
            "Appendix",
            "Reusable training entrypoint driven by JSON configs.",
        ),
        (
            "dcas/scripts/run_recommender_benchmarks.py",
            "experiment_runner",
            "Experiments",
            "Main benchmark runner used for V3/V4 result matrices.",
        ),
        (
            "dcas/scripts/evaluate_recommender.py",
            "experiment_runner",
            "Experiments",
            "Computes benchmark metrics and comparison outputs.",
        ),
        (
            "dcas/scripts/prepare_real_pal_bundle.py",
            "pal_human_loop",
            "Method",
            "Builds the real PAL task packet and candidate pool.",
        ),
        (
            "dcas/scripts/run_phase3_pal.py",
            "pal_human_loop",
            "Method",
            "Closes the PAL feedback loop from constraints to retraining.",
        ),
        (
            "dcas/scripts/build_pal_constraints_from_annotations.py",
            "pal_human_loop",
            "Method",
            "Transforms human annotation sheets into PAL constraints.",
        ),
        (
            "configs/dataset/research_dataset_v4_main_from_v3.json",
            "config_primary",
            "Appendix",
            "Primary V4 main dataset contract.",
        ),
        (
            "configs/dataset/research_dataset_v4_routeA_small.json",
            "config_primary",
            "Appendix",
            "Primary V4 small dataset contract.",
        ),
        (
            "configs/train/train_v4_main_culturemert_stage3.run.json",
            "config_primary",
            "Appendix",
            "Primary V4 main CultureMERT stage3 training setup.",
        ),
        (
            "configs/train/train_v4_main_gemini_stage3.run.json",
            "config_primary",
            "Appendix",
            "Primary V4 main Gemini stage3 training setup.",
        ),
        (
            "configs/benchmark/recommender_benchmark_v4_main_culturemert_stage3_lambdamart.run.json",
            "config_primary",
            "Appendix",
            "Primary V4 main CultureMERT benchmark setup.",
        ),
        (
            "configs/benchmark/recommender_benchmark_v4_main_gemini_stage3_lambdamart.run.json",
            "config_primary",
            "Appendix",
            "Primary V4 main Gemini benchmark setup.",
        ),
        (
            "configs/pal/pal_v4_main_culturemert_prepare.run.json",
            "config_primary",
            "Appendix",
            "Real PAL packet preparation setup.",
        ),
        (
            "configs/pal/pal_v4_main_culturemert_real.run.json",
            "config_primary",
            "Appendix",
            "Real PAL round ingestion and retraining setup.",
        ),
        (
            "paper/ismir2026_draft.tex",
            "paper_target",
            "Paper",
            "Draft paper still needs synchronization with current V4 evidence.",
        ),
        (
            "dcas_server/app.py",
            "platform_support",
            "Appendix",
            "Serving/demo layer, not a primary research contribution file.",
        ),
        (
            "web/package.json",
            "platform_support",
            "Appendix",
            "Web/demo dependency manifest, auxiliary to the research paper.",
        ),
    ]
    for rel_path, bucket, paper_section, rationale in curated:
        path = REPO_ROOT / rel_path
        rows.append(
            {
                "path": rel_path,
                "exists": str(path.exists()).lower(),
                "tracked": str(rel_path in tracked_files).lower(),
                "bucket": bucket,
                "paper_section": paper_section,
                "rationale": rationale,
            }
        )
    return rows


def _major_dataflow_mermaid() -> str:
    return "\n".join(
        [
            "flowchart LR",
            '  A["Raw Source Imports\\n(import_hf_*, import_europeana_*)"] --> B["Metadata Merge/Harmonize\\nmerge_metadata*, harmonize_v4_metadata.py"]',
            '  B --> C["Embedding Build\\nbuild_tracks_from_audio.py / build_tracks_with_gemini.py"]',
            '  B --> D["Interaction Synthesis\\nsynthesize_interactions.py"]',
            '  C --> E["Dataset Artifacts\\nstorage/public/research_dataset_v4/*"]',
            "  D --> E",
            '  E --> F["Training\\nrun_train_from_json.py"]',
            '  E --> G["Benchmarking\\nrun_recommender_benchmarks.py"]',
            "  F --> G",
            '  G --> H["Reports\\nreports/benchmarks/*"]',
            '  E --> I["PAL Packet Prep\\nprepare_real_pal_bundle.py"]',
            '  I --> J["Human Annotation CSV"]',
            '  J --> K["PAL Constraint Build\\nbuild_pal_constraints_from_annotations.py"]',
            '  K --> L["PAL Retraining\\nrun_phase3_pal.py"]',
            "  L --> H",
        ]
    )


def _code_dependency_mermaid(graph_summary: dict[str, Any]) -> str:
    lines = [
        "flowchart LR",
        '  M1["dcas.data.npz_tracks"] --> M2["run_recommender_benchmarks.py"]',
        '  M3["dcas.pipelines"] --> M4["prepare_real_pal_bundle.py"]',
        '  M3 --> M5["run_pal_platform.py"]',
        '  M6["dcas.recommender"] --> M2',
        '  M7["dcas.embedding_recommenders"] --> M2',
        '  M8["compare_recommender_runs.py"] --> M2',
        '  M9["build_research_dataset_v4.py"] --> M10["harmonize_v4_metadata.py"]',
        '  M9 --> M11["build_tracks_from_audio.py"]',
        '  M9 --> M12["build_tracks_with_gemini.py"]',
        '  M9 --> M13["synthesize_interactions.py"]',
    ]
    for idx, row in enumerate(graph_summary.get("top_incoming_modules", [])[:5], start=20):
        lines.append(f'  T{idx}["{row["module"]}\\n(in={row["incoming_edges"]})"]')
    return "\n".join(lines)


def _summary_markdown(report: dict[str, Any]) -> str:
    top_dirs = report["top_level_structure"]
    figures = report["figure_summary"]
    datasets = report["dataset_summary"]
    results = report["result_summary"]
    exec_checks = report["executability"]
    config_summary = report["config_summary"]
    env = report["environment"]
    lines: list[str] = []
    lines.append("# Phase 1 Workspace Diagnostic")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "- Goal: repository-wide diagnostic for file structure, executability, configs, figures, datasets, results, and reproducibility."
    )
    lines.append(f"- Branch: `{env.get('git_branch')}`")
    lines.append(f"- Commit: `{env.get('git_commit')}`")
    lines.append("")
    lines.append("## Workspace Structure")
    lines.append("")
    lines.append("| top-level path | kind | file_count |")
    lines.append("|---|---|---:|")
    for row in top_dirs:
        lines.append(
            f"| {row['path']} | {row['kind']} | {row['file_count'] if row['file_count'] is not None else ''} |"
        )
    lines.append("")
    lines.append("## Code Dependency Graph")
    lines.append("")
    lines.append("```mermaid")
    lines.append(report["code_dependency_mermaid"])
    lines.append("```")
    lines.append("")
    lines.append("## Data Flow Graph")
    lines.append("")
    lines.append("```mermaid")
    lines.append(report["dataflow_mermaid"])
    lines.append("```")
    lines.append("")
    lines.append("## Core Files vs Auxiliary Files")
    lines.append("")
    lines.append("| path | bucket | paper_section | tracked | rationale |")
    lines.append("|---|---|---|---|---|")
    for row in report["core_file_inventory"]:
        lines.append(
            f"| {row['path']} | {row['bucket']} | {row['paper_section']} | {row['tracked']} | {row['rationale']} |"
        )
    lines.append("")
    lines.append("## Executability Check")
    lines.append("")
    lines.append(
        f"- Python compile check: `{exec_checks['compile_check']['checked']}` files scanned, `{len(exec_checks['compile_check']['failed'])}` failures."
    )
    if exec_checks["compile_check"]["failed"]:
        lines.append(f"- Compile failures: `{';'.join(exec_checks['compile_check']['failed'])}`")
    lines.append(f"- Hard-coded absolute path findings in Python code: `{len(exec_checks['hardcoded_paths'])}`.")
    lines.append(f"- Undefined-name heuristic findings: `{len(exec_checks['undefined_name_findings'])}`.")
    missing = exec_checks["external_import_findings"]["candidate_undeclared_packages"]
    lines.append(f"- Candidate undeclared Python packages: `{len(missing)}`.")
    if missing:
        lines.append("")
        lines.append("| package | count |")
        lines.append("|---|---:|")
        for row in missing[:20]:
            lines.append(f"| {row['package']} | {row['count']} |")
    lines.append("")
    lines.append("## Config Inventory")
    lines.append("")
    lines.append("| category | count |")
    lines.append("|---|---:|")
    for key, value in config_summary["config_count_by_category"].items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("| category:status | count |")
    lines.append("|---|---:|")
    for key, value in config_summary["tracked_status_by_category"].items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("- Detailed settings table: `experiment_settings_table.csv`")
    lines.append("- Full flattened parameters: `config_parameters_long.csv`")
    lines.append("")
    lines.append("## Figure Inventory")
    lines.append("")
    lines.append(f"- Total figure assets under `reports/figures`: `{figures['n_figures']}`.")
    lines.append(f"- Count by extension: `{json.dumps(figures['count_by_extension'], ensure_ascii=False)}`")
    lines.append(f"- Count by figure type: `{json.dumps(figures['count_by_type'], ensure_ascii=False)}`")
    low_res = sum(1 for row in report["figure_inventory"] if "low_raster_resolution" in row["issue_flags"])
    lines.append(f"- Raster figures flagged for low resolution: `{low_res}`.")
    lines.append("")
    lines.append("## Dataset Inventory")
    lines.append("")
    lines.append(f"- Track manifest rows scanned: `{len(report['dataset_inventory'])}`.")
    lines.append(
        f"- Manifest rows by top dataset: `{json.dumps(datasets['manifest_rows_by_top_dataset'], ensure_ascii=False)}`"
    )
    lines.append("")
    lines.append(
        "| dataset_dir | track_artifact | n_tracks | dim | metadata_rows | culture_unique | source_unique | n_errors |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in sorted(report["dataset_inventory"], key=_dataset_sort_key)[:12]:
        lines.append(
            f"| {row['dataset_dir']} | {row['track_artifact']} | {row.get('n_tracks') or ''} | {row.get('dim') or ''} | {row.get('metadata_rows') or ''} | {row.get('culture_unique') or ''} | {row.get('source_unique') or ''} | {row.get('n_errors') or 0} |"
        )
    lines.append("")
    lines.append("## Result Inventory")
    lines.append("")
    lines.append(f"- Result files scanned (`.json/.csv/.log` under `reports/`): `{len(report['result_inventory'])}`.")
    lines.append(f"- Count by category: `{json.dumps(results['result_count_by_category'], ensure_ascii=False)}`")
    lines.append(
        f"- Partial/failure-side assets (`smoke/probe/tmp` or `n_errors>0`): `{len(report['partial_failure_inventory'])}`."
    )
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- Python executable: `{env['python_executable']}`")
    lines.append(f"- CUDA available: `{env['cuda_available']}`")
    lines.append(f"- CUDA version: `{env['cuda_version']}`")
    lines.append(f"- GPU[0]: `{env['cuda_device_0']}`")
    lines.append(f"- Seed frequency across configs: `{json.dumps(env['seed_frequency'], ensure_ascii=False)}`")
    lines.append("")
    lines.append("| package | version |")
    lines.append("|---|---|")
    for key, value in env["package_versions"].items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("## Primary Findings")
    lines.append("")
    lines.append(
        "- The codebase centers on `dcas` data, embedding, recommendation, and PAL pipelines; V4 build/benchmark scripts form the core research path."
    )
    lines.append(
        f"- `configs/` contains `{sum(config_summary['config_count_by_category'].values())}` JSON configs across dataset, embedding, training, benchmark, and PAL stages."
    )
    lines.append(
        "- Figure assets are concentrated in two overview bundles and currently rely almost entirely on PNG outputs."
    )
    lines.append(
        "- V4 benchmark and dataset artifacts are separated cleanly under `reports/benchmarks/v4_*` and `reports/datasets/research_dataset_v4/*`."
    )
    if missing:
        lines.append(
            "- There are candidate undeclared Python packages that should be cross-checked before claiming full one-command reproducibility."
        )
    if low_res > 0:
        lines.append(
            "- Some figure assets are below a conservative 1200x800 raster threshold and may need re-export before paper submission."
        )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class Args:
    out_dir: str


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 full-workspace diagnostic scan.")
    ap.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "reports" / "audits" / "phase1_workspace_scan_2026-03-21"),
        help="Directory for generated audit artifacts.",
    )
    ns = ap.parse_args()
    args = Args(out_dir=str(ns.out_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracked_files = _git_tracked_files()
    top_level_structure = _top_level_dir_inventory()
    module_rows, dependency_edges, dependency_graph = _collect_python_modules()
    config_rows, config_param_rows, config_summary = _collect_configs(tracked_files)
    experiment_settings_rows = _experiment_settings_table(config_rows, config_param_rows)
    figure_rows, figure_summary = _collect_figures()
    dataset_rows, dataset_summary = _collect_datasets()
    result_rows, result_summary = _collect_results()
    partial_failure_rows = _collect_partial_failure_assets()
    environment = _collect_environment(config_param_rows)
    executability = {
        "compile_check": _compile_check(),
        "hardcoded_paths": _hardcoded_path_scan(),
        "undefined_name_findings": _undefined_name_scan(),
        "external_import_findings": _collect_external_import_findings(module_rows),
    }
    core_files = _core_file_inventory(tracked_files)
    report = {
        "repo_root": str(REPO_ROOT),
        "top_level_structure": top_level_structure,
        "python_module_inventory": module_rows,
        "dependency_edges": dependency_edges,
        "dependency_graph": dependency_graph,
        "core_file_inventory": core_files,
        "config_inventory": config_rows,
        "config_parameters": config_param_rows,
        "config_summary": config_summary,
        "experiment_settings_table": experiment_settings_rows,
        "figure_inventory": figure_rows,
        "figure_summary": figure_summary,
        "dataset_inventory": dataset_rows,
        "dataset_summary": dataset_summary,
        "result_inventory": result_rows,
        "result_summary": result_summary,
        "partial_failure_inventory": partial_failure_rows,
        "environment": environment,
        "executability": executability,
        "code_dependency_mermaid": _code_dependency_mermaid(dependency_graph),
        "dataflow_mermaid": _major_dataflow_mermaid(),
    }
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(_summary_markdown(report), encoding="utf-8")
    (out_dir / "code_dependency_graph.mmd").write_text(report["code_dependency_mermaid"] + "\n", encoding="utf-8")
    (out_dir / "dataflow_graph.mmd").write_text(report["dataflow_mermaid"] + "\n", encoding="utf-8")
    (out_dir / "reproducibility_environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "python_module_inventory.csv", module_rows)
    _write_csv(out_dir / "dependency_edges.csv", dependency_edges)
    _write_csv(out_dir / "core_file_inventory.csv", core_files)
    _write_csv(out_dir / "config_inventory.csv", config_rows)
    _write_csv(out_dir / "experiment_settings_table.csv", experiment_settings_rows)
    _write_csv(out_dir / "config_parameters_long.csv", config_param_rows)
    _write_csv(out_dir / "figure_inventory.csv", figure_rows)
    _write_csv(out_dir / "dataset_inventory.csv", dataset_rows)
    _write_csv(out_dir / "result_inventory.csv", result_rows)
    _write_csv(out_dir / "partial_failure_inventory.csv", partial_failure_rows)
    _write_csv(out_dir / "hardcoded_path_findings.csv", executability["hardcoded_paths"])
    _write_csv(
        out_dir / "undefined_name_findings.csv",
        executability["undefined_name_findings"],
    )
    _write_csv(
        out_dir / "candidate_undeclared_packages.csv",
        executability["external_import_findings"]["candidate_undeclared_packages"],
    )
    print(str(out_dir))


if __name__ == "__main__":
    main()
