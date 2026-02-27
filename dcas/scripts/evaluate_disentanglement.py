from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from dcas.data.npz_tracks import load_tracks
from dcas.serialization import load_checkpoint


def _parse_seeds(raw: str) -> list[int]:
    out = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not out:
        raise ValueError("at least one seed is required")
    return out


def _mean_std_ci(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(float(arr.size))) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci95}


def _entropy_discrete(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts.astype(np.float64)
    p = p / max(1e-12, float(p.sum()))
    return float(-np.sum(p * np.log(p + 1e-12)))


def _mutual_info_discrete(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0 or x.shape[0] != y.shape[0]:
        return 0.0
    n = int(x.shape[0])
    if n == 0:
        return 0.0

    x_vals, x_inv = np.unique(x, return_inverse=True)
    y_vals, y_inv = np.unique(y, return_inverse=True)
    nx = int(x_vals.shape[0])
    ny = int(y_vals.shape[0])
    if nx <= 1 or ny <= 1:
        return 0.0

    joint = np.zeros((nx, ny), dtype=np.float64)
    for i in range(n):
        joint[x_inv[i], y_inv[i]] += 1.0
    joint = joint / max(1e-12, float(joint.sum()))
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    ratio = joint / np.maximum(1e-12, px @ py)
    return float(np.sum(joint * np.log(np.maximum(1e-12, ratio))))


def _discretize_quantile(x: np.ndarray, n_bins: int = 20) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0,), dtype=np.int64)
    n_bins = max(2, int(n_bins))
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x.astype(np.float64), qs)
    edges = np.unique(edges)
    if edges.shape[0] <= 2:
        return np.zeros_like(x, dtype=np.int64)
    return np.digitize(x, edges[1:-1], right=False).astype(np.int64)


def _anova_f_score(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        return 0.0
    classes = np.unique(y)
    n = int(x.shape[0])
    k = int(classes.shape[0])
    if n <= k or k <= 1:
        return 0.0

    mu = float(np.mean(x))
    ss_between = 0.0
    ss_within = 0.0
    for c in classes.tolist():
        mask = y == c
        if not np.any(mask):
            continue
        xc = x[mask]
        mu_c = float(np.mean(xc))
        ss_between += float(xc.shape[0]) * (mu_c - mu) ** 2
        ss_within += float(np.sum((xc - mu_c) ** 2))

    df_between = max(1, k - 1)
    df_within = max(1, n - k)
    ms_between = ss_between / float(df_between)
    ms_within = ss_within / float(df_within)
    if ms_within <= 1e-12:
        return 0.0
    return float(ms_between / ms_within)


def _dci_from_importance(importance: np.ndarray) -> tuple[float, float]:
    imp = np.maximum(importance.astype(np.float64), 0.0)
    total = float(imp.sum())
    if total <= 0:
        return 0.0, 0.0
    dims, factors = imp.shape

    dim_weights = imp.sum(axis=1) / total
    dis_scores: list[float] = []
    for d in range(dims):
        row = imp[d]
        row_sum = float(row.sum())
        if row_sum <= 0:
            dis_scores.append(0.0)
            continue
        p = row / row_sum
        h = float(-np.sum(p * np.log(p + 1e-12)))
        dis_scores.append(1.0 - h / np.log(max(2, factors)))
    disentanglement = float(np.sum(dim_weights * np.array(dis_scores, dtype=np.float64)))

    fac_weights = imp.sum(axis=0) / total
    comp_scores: list[float] = []
    for f in range(factors):
        col = imp[:, f]
        col_sum = float(col.sum())
        if col_sum <= 0:
            comp_scores.append(0.0)
            continue
        p = col / col_sum
        h = float(-np.sum(p * np.log(p + 1e-12)))
        comp_scores.append(1.0 - h / np.log(max(2, dims)))
    completeness = float(np.sum(fac_weights * np.array(comp_scores, dtype=np.float64)))

    return disentanglement, completeness


def _split_indices(n: int, seed: int, test_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(int(n))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    n_test = max(1, min(n - 1, int(round(float(n) * float(test_ratio)))))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def _linear_probe_accuracy(
    z: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> float:
    y_train = y[train_idx]
    classes = np.unique(y_train)
    if classes.shape[0] <= 1:
        return float("nan")

    class_to_new = {int(c): i for i, c in enumerate(classes.tolist())}
    y_mapped = np.array([class_to_new[int(c)] for c in y.tolist()], dtype=np.int64)

    x_train = z[train_idx].astype(np.float32)
    x_test = z[test_idx].astype(np.float32)
    y_train_t = torch.tensor(y_mapped[train_idx], dtype=torch.long)
    y_test_t = torch.tensor(y_mapped[test_idx], dtype=torch.long)

    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    x_train = (x_train - mu) / sd
    x_test = (x_test - mu) / sd

    torch.manual_seed(int(seed))
    model = nn.Linear(int(x_train.shape[1]), int(classes.shape[0]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)

    model.train()
    for _ in range(int(epochs)):
        logits = model(x_train_t)
        loss = nn.functional.cross_entropy(logits, y_train_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(x_test_t).argmax(dim=-1)
    return float((pred == y_test_t).float().mean().item())


def _load_metadata_map(path: str, key_col: str = "track_id") -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = str(row.get(key_col, "")).strip()
            if tid:
                out[tid] = {str(k): str(v) for k, v in row.items()}
    return out


def _encode_factors(
    track_ids: np.ndarray,
    tracks_culture: np.ndarray,
    metadata_map: dict[str, dict[str, str]],
    factor_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, list[str]]]:
    n = int(track_ids.shape[0])
    vals_by_factor: dict[str, np.ndarray] = {}
    mask = np.ones((n,), dtype=bool)
    factor_order: list[str] = []
    class_names: dict[str, list[str]] = {}

    for col in factor_cols:
        c = str(col).strip()
        if not c:
            continue
        if c == "culture":
            raw = tracks_culture.astype(str)
        else:
            raw = np.array([str(metadata_map.get(str(tid), {}).get(c, "")).strip() for tid in track_ids.tolist()], dtype=object)
        valid = np.array([bool(x) and x.lower() not in {"nan", "none"} for x in raw], dtype=bool)
        mask = mask & valid
        vals_by_factor[c] = raw
        factor_order.append(c)

    if not factor_order:
        raise ValueError("no valid factor columns")
    if not np.any(mask):
        raise ValueError("no samples remain after factor filtering")

    encoded: list[np.ndarray] = []
    for c in factor_order:
        raw = vals_by_factor[c][mask].astype(str)
        cls = sorted(set(raw.tolist()))
        if len(cls) <= 1:
            raise ValueError(f"factor '{c}' has <=1 class after filtering")
        class_names[c] = cls
        mapper = {v: i for i, v in enumerate(cls)}
        encoded.append(np.array([mapper[v] for v in raw.tolist()], dtype=np.int64))

    y = np.stack(encoded, axis=1)
    return mask, y, factor_order, class_names


def _evaluate_one_seed(
    z: np.ndarray,
    y: np.ndarray,
    factor_names: list[str],
    seed: int,
    n_bins: int,
    test_ratio: float,
    probe_epochs: int,
    probe_lr: float,
    probe_weight_decay: float,
) -> dict[str, Any]:
    n, d = z.shape
    f = y.shape[1]
    if n == 0 or d == 0 or f == 0:
        raise ValueError("empty space or factors")

    mi = np.zeros((d, f), dtype=np.float64)
    ent = np.zeros((f,), dtype=np.float64)
    for fi in range(f):
        ent[fi] = _entropy_discrete(y[:, fi])
    for di in range(d):
        z_disc = _discretize_quantile(z[:, di], n_bins=n_bins)
        for fi in range(f):
            mi[di, fi] = _mutual_info_discrete(z_disc, y[:, fi])

    mig_per_factor: dict[str, float] = {}
    for fi, name in enumerate(factor_names):
        col = np.sort(mi[:, fi])[::-1]
        top1 = float(col[0]) if col.size >= 1 else 0.0
        top2 = float(col[1]) if col.size >= 2 else 0.0
        denom = max(1e-12, float(ent[fi]))
        mig_per_factor[name] = float((top1 - top2) / denom)
    mig = float(np.mean(np.array(list(mig_per_factor.values()), dtype=np.float64))) if mig_per_factor else 0.0

    importance = np.zeros((d, f), dtype=np.float64)
    for di in range(d):
        x = z[:, di]
        for fi in range(f):
            importance[di, fi] = _anova_f_score(x, y[:, fi])
    dci_dis, dci_comp = _dci_from_importance(importance)

    train_idx, test_idx = _split_indices(n=n, seed=int(seed), test_ratio=float(test_ratio))
    dci_info_per_factor: dict[str, float] = {}
    for fi, name in enumerate(factor_names):
        acc = _linear_probe_accuracy(
            z=z,
            y=y[:, fi],
            train_idx=train_idx,
            test_idx=test_idx,
            seed=int(seed) + fi,
            epochs=int(probe_epochs),
            lr=float(probe_lr),
            weight_decay=float(probe_weight_decay),
        )
        dci_info_per_factor[name] = float(acc)
    valid = [v for v in dci_info_per_factor.values() if not np.isnan(v)]
    dci_info = float(np.mean(np.array(valid, dtype=np.float64))) if valid else float("nan")

    sap_per_factor: dict[str, float] = {}
    for fi, name in enumerate(factor_names):
        s = np.sort(importance[:, fi])[::-1]
        if s.size <= 1:
            sap_per_factor[name] = float(s[0]) if s.size == 1 else 0.0
        else:
            sap_per_factor[name] = float(s[0] - s[1])
    sap = float(np.mean(np.array(list(sap_per_factor.values()), dtype=np.float64))) if sap_per_factor else 0.0

    return {
        "seed": int(seed),
        "MIG": float(mig),
        "MIG_per_factor": mig_per_factor,
        "DCI_disentanglement": float(dci_dis),
        "DCI_completeness": float(dci_comp),
        "DCI_informativeness": float(dci_info),
        "DCI_informativeness_per_factor": dci_info_per_factor,
        "SAP": float(sap),
        "SAP_per_factor": sap_per_factor,
    }


def _aggregate_space(records: list[dict[str, Any]], factors: list[str]) -> dict[str, Any]:
    scalar_keys = [
        "MIG",
        "DCI_disentanglement",
        "DCI_completeness",
        "DCI_informativeness",
        "SAP",
    ]
    summary: dict[str, Any] = {}
    for k in scalar_keys:
        vals = [float(r[k]) for r in records]
        stat = _mean_std_ci(vals)
        summary[f"{k}_mean"] = stat["mean"]
        summary[f"{k}_std"] = stat["std"]
        summary[f"{k}_ci95"] = stat["ci95"]

    def _per_factor_stats(key: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for f in factors:
            vals = [float(r[key][f]) for r in records if f in r[key]]
            out[f] = _mean_std_ci(vals)
        return out

    summary["MIG_per_factor"] = _per_factor_stats("MIG_per_factor")
    summary["DCI_informativeness_per_factor"] = _per_factor_stats("DCI_informativeness_per_factor")
    summary["SAP_per_factor"] = _per_factor_stats("SAP_per_factor")
    return summary


def evaluate_disentanglement(
    model_path: str,
    tracks_path: str,
    metadata_csv: str,
    factor_cols: list[str],
    out_json: str | None = None,
    out_md: str | None = None,
    seeds: list[int] | None = None,
    n_bins: int = 20,
    test_ratio: float = 0.2,
    probe_epochs: int = 200,
    probe_lr: float = 1e-2,
    probe_weight_decay: float = 1e-4,
    prefer_cuda: bool = False,
) -> dict[str, Any]:
    device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(model_path, map_location=str(device))
    model.to(device)
    model.eval()

    seed_list = list(seeds or [42])
    tracks = load_tracks(tracks_path)
    metadata_map = _load_metadata_map(metadata_csv)
    mask, y, factors, class_names = _encode_factors(
        track_ids=tracks.track_id,
        tracks_culture=tracks.culture,
        metadata_map=metadata_map,
        factor_cols=factor_cols,
    )

    x = torch.from_numpy(tracks.embedding.astype(np.float32)).to(device)
    with torch.no_grad():
        zc, zs, za = model.encode(x)
    zc_np = zc.detach().cpu().numpy()[mask]
    zs_np = zs.detach().cpu().numpy()[mask]
    za_np = za.detach().cpu().numpy()[mask]

    spaces_raw = {"zc": zc_np, "zs": zs_np, "za": za_np}
    spaces: dict[str, Any] = {}
    for name, z in spaces_raw.items():
        per_seed = [
            _evaluate_one_seed(
                z=z,
                y=y,
                factor_names=factors,
                seed=int(s),
                n_bins=int(n_bins),
                test_ratio=float(test_ratio),
                probe_epochs=int(probe_epochs),
                probe_lr=float(probe_lr),
                probe_weight_decay=float(probe_weight_decay),
            )
            for s in seed_list
        ]
        summary = _aggregate_space(per_seed, factors=factors)
        first = per_seed[0]
        spaces[name] = {
            "n_samples": int(z.shape[0]),
            "dim": int(z.shape[1]),
            "factors": factors,
            "seeds": seed_list,
            "per_seed": per_seed,
            "summary": summary,
            # backward-compatible single-value fields (first seed)
            "MIG": float(first["MIG"]),
            "MIG_per_factor": first["MIG_per_factor"],
            "DCI_disentanglement": float(first["DCI_disentanglement"]),
            "DCI_completeness": float(first["DCI_completeness"]),
            "DCI_informativeness": float(first["DCI_informativeness"]),
            "DCI_informativeness_per_factor": first["DCI_informativeness_per_factor"],
            "SAP": float(first["SAP"]),
            "SAP_per_factor": first["SAP_per_factor"],
        }

    out: dict[str, Any] = {
        "config": {
            "model_path": str(model_path),
            "tracks_path": str(tracks_path),
            "metadata_csv": str(metadata_csv),
            "factors": factors,
            "seeds": seed_list,
            "n_bins": int(n_bins),
            "test_ratio": float(test_ratio),
            "probe_epochs": int(probe_epochs),
            "probe_lr": float(probe_lr),
            "probe_weight_decay": float(probe_weight_decay),
            "device": str(device),
        },
        "class_names": class_names,
        "n_samples_total": int(tracks.track_id.shape[0]),
        "n_samples_used": int(int(mask.sum())),
        "spaces": spaces,
    }

    if out_json:
        p = Path(out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    if out_md:
        lines = [
            "# Disentanglement Report (MIG / DCI / SAP, Multi-Seed)",
            "",
            f"- samples used: `{int(mask.sum())}` / total `{int(tracks.track_id.shape[0])}`",
            f"- factors: `{', '.join(factors)}`",
            f"- seeds: `{', '.join(str(s) for s in seed_list)}`",
            "",
            "| space | MIG(mean+/-std) | DCI_dis(mean+/-std) | DCI_comp(mean+/-std) | DCI_info(mean+/-std) | SAP(mean+/-std) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name in ["zc", "zs", "za"]:
            s = spaces[name]["summary"]
            lines.append(
                f"| {name} | {s['MIG_mean']:.6f}+/-{s['MIG_std']:.6f} | "
                f"{s['DCI_disentanglement_mean']:.6f}+/-{s['DCI_disentanglement_std']:.6f} | "
                f"{s['DCI_completeness_mean']:.6f}+/-{s['DCI_completeness_std']:.6f} | "
                f"{s['DCI_informativeness_mean']:.6f}+/-{s['DCI_informativeness_std']:.6f} | "
                f"{s['SAP_mean']:.6f}+/-{s['SAP_std']:.6f} |"
            )
        lines.append("")
        p = Path(out_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate disentanglement metrics (MIG/DCI/SAP) on zc/zs/za.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--factors", default="culture,label", help="comma-separated factor columns")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--seeds", default="42", help="comma-separated seeds, e.g. 42,43,44")
    ap.add_argument("--n_bins", type=int, default=20)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--probe_epochs", type=int, default=200)
    ap.add_argument("--probe_lr", type=float, default=1e-2)
    ap.add_argument("--probe_weight_decay", type=float, default=1e-4)
    ap.add_argument("--prefer_cuda", action="store_true")
    args = ap.parse_args()

    factor_cols = [x.strip() for x in str(args.factors).split(",") if x.strip()]
    seed_list = _parse_seeds(str(args.seeds))

    out = evaluate_disentanglement(
        model_path=str(args.model),
        tracks_path=str(args.tracks),
        metadata_csv=str(args.metadata),
        factor_cols=factor_cols,
        out_json=str(args.out_json) if args.out_json else None,
        out_md=str(args.out_md) if args.out_md else None,
        seeds=seed_list,
        n_bins=int(args.n_bins),
        test_ratio=float(args.test_ratio),
        probe_epochs=int(args.probe_epochs),
        probe_lr=float(args.probe_lr),
        probe_weight_decay=float(args.probe_weight_decay),
        prefer_cuda=bool(args.prefer_cuda),
    )

    tiny = {
        "n_samples_used": out["n_samples_used"],
        "spaces": {
            k: {
                "MIG_mean": v["summary"]["MIG_mean"],
                "MIG_ci95": v["summary"]["MIG_ci95"],
                "DCI_disentanglement_mean": v["summary"]["DCI_disentanglement_mean"],
                "DCI_completeness_mean": v["summary"]["DCI_completeness_mean"],
                "DCI_informativeness_mean": v["summary"]["DCI_informativeness_mean"],
                "SAP_mean": v["summary"]["SAP_mean"],
            }
            for k, v in out["spaces"].items()
        },
    }
    print(json.dumps(tiny, ensure_ascii=False))


if __name__ == "__main__":
    main()
