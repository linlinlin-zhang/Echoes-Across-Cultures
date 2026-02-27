from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dcas.data.npz_tracks import load_tracks
from dcas.serialization import load_checkpoint


def _entropy_discrete(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    _ = vals
    p = counts.astype(np.float64)
    p = p / max(1e-12, float(p.sum()))
    return float(-np.sum(p * np.log(p + 1e-12)))


def _mutual_info_discrete(x: np.ndarray, y: np.ndarray) -> float:
    # x and y are integer-coded vectors.
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
    mi = float(np.sum(joint * np.log(np.maximum(1e-12, ratio))))
    return mi


def _discretize_quantile(x: np.ndarray, n_bins: int = 20) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0,), dtype=np.int64)
    n_bins = max(2, int(n_bins))
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x.astype(np.float64), qs)
    edges = np.unique(edges)
    if edges.shape[0] <= 2:
        return np.zeros_like(x, dtype=np.int64)
    bins = np.digitize(x, edges[1:-1], right=False).astype(np.int64)
    return bins


def _dci_from_importance(importance: np.ndarray) -> tuple[float, float]:
    # importance shape: [dims, factors]
    imp = np.maximum(importance.astype(np.float64), 0.0)
    total = float(imp.sum())
    if total <= 0:
        return 0.0, 0.0
    dims, factors = imp.shape

    # disentanglement
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

    # completeness
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


def _centroid_accuracy(z: np.ndarray, y: np.ndarray, seed: int = 42, train_ratio: float = 0.8) -> float:
    n = int(z.shape[0])
    if n <= 2:
        return float("nan")
    idx = np.arange(n)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    n_train = max(1, min(n - 1, int(round(n * float(train_ratio)))))
    tr = idx[:n_train]
    te = idx[n_train:]
    if te.size == 0:
        return float("nan")

    labels = np.unique(y[tr])
    if labels.size <= 1:
        return float("nan")

    centroids: dict[int, np.ndarray] = {}
    for c in labels.tolist():
        mask = y[tr] == c
        if not np.any(mask):
            continue
        centroids[int(c)] = z[tr][mask].mean(axis=0)
    if len(centroids) <= 1:
        return float("nan")

    pred: list[int] = []
    for i in te.tolist():
        x = z[i]
        best_c = None
        best_d = None
        for c, mu in centroids.items():
            d = float(np.sum((x - mu) ** 2))
            if best_d is None or d < best_d:
                best_d = d
                best_c = c
        pred.append(int(best_c))
    y_true = y[te].astype(np.int64)
    y_pred = np.array(pred, dtype=np.int64)
    return float((y_true == y_pred).mean())


def _sap_score(z: np.ndarray, ys: np.ndarray, seed: int = 42, train_ratio: float = 0.8) -> tuple[float, dict[str, float]]:
    # ys shape: [N, F], each integer-coded factor label.
    n, d = z.shape
    f = ys.shape[1]
    if n <= 2 or d == 0 or f == 0:
        return 0.0, {}

    idx = np.arange(n)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    n_train = max(1, min(n - 1, int(round(n * float(train_ratio)))))
    tr = idx[:n_train]
    te = idx[n_train:]
    if te.size == 0:
        return 0.0, {}

    factor_gaps: dict[str, float] = {}
    for fi in range(f):
        y_tr = ys[tr, fi]
        y_te = ys[te, fi]
        classes = np.unique(y_tr)
        if classes.size <= 1:
            factor_gaps[str(fi)] = 0.0
            continue
        scores: list[float] = []
        for di in range(d):
            x_tr = z[tr, di]
            x_te = z[te, di]
            means: dict[int, float] = {}
            for c in classes.tolist():
                mask = y_tr == c
                if np.any(mask):
                    means[int(c)] = float(np.mean(x_tr[mask]))
            if len(means) <= 1:
                scores.append(0.0)
                continue
            pred: list[int] = []
            for val in x_te.tolist():
                best_c = None
                best_d = None
                for c, mu in means.items():
                    dist = abs(float(val) - float(mu))
                    if best_d is None or dist < best_d:
                        best_d = dist
                        best_c = c
                pred.append(int(best_c))
            acc = float((np.array(pred, dtype=np.int64) == y_te).mean())
            scores.append(acc)
        scores_sorted = sorted(scores, reverse=True)
        if len(scores_sorted) < 2:
            gap = float(scores_sorted[0]) if scores_sorted else 0.0
        else:
            gap = float(scores_sorted[0] - scores_sorted[1])
        factor_gaps[str(fi)] = gap
    sap = float(np.mean(np.array(list(factor_gaps.values()), dtype=np.float64))) if factor_gaps else 0.0
    return sap, factor_gaps


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


def _evaluate_one_space(
    z: np.ndarray,
    y: np.ndarray,
    factor_names: list[str],
    seed: int,
    n_bins: int,
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

    mig_factors: dict[str, float] = {}
    for fi, name in enumerate(factor_names):
        col = np.sort(mi[:, fi])[::-1]
        top1 = float(col[0]) if col.size >= 1 else 0.0
        top2 = float(col[1]) if col.size >= 2 else 0.0
        denom = max(1e-12, float(ent[fi]))
        mig_factors[name] = float((top1 - top2) / denom)
    mig = float(np.mean(np.array(list(mig_factors.values()), dtype=np.float64))) if mig_factors else 0.0

    dci_dis, dci_comp = _dci_from_importance(mi)
    info_scores: dict[str, float] = {}
    for fi, name in enumerate(factor_names):
        info_scores[name] = _centroid_accuracy(z, y[:, fi], seed=seed + fi, train_ratio=0.8)
    valid_info = [v for v in info_scores.values() if not np.isnan(v)]
    dci_info = float(np.mean(np.array(valid_info, dtype=np.float64))) if valid_info else float("nan")

    sap, sap_by_idx = _sap_score(z, y, seed=seed, train_ratio=0.8)
    sap_factors = {factor_names[int(k)]: float(v) for k, v in sap_by_idx.items()}

    return {
        "n_samples": int(n),
        "dim": int(d),
        "factors": factor_names,
        "MIG": float(mig),
        "MIG_per_factor": mig_factors,
        "DCI_disentanglement": float(dci_dis),
        "DCI_completeness": float(dci_comp),
        "DCI_informativeness": float(dci_info),
        "DCI_informativeness_per_factor": info_scores,
        "SAP": float(sap),
        "SAP_per_factor": sap_factors,
    }


def evaluate_disentanglement(
    model_path: str,
    tracks_path: str,
    metadata_csv: str,
    factor_cols: list[str],
    out_json: str | None = None,
    out_md: str | None = None,
    seed: int = 42,
    n_bins: int = 20,
    prefer_cuda: bool = False,
) -> dict[str, Any]:
    device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(model_path, map_location=str(device))
    model.to(device)
    model.eval()

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

    spaces = {
        "zc": _evaluate_one_space(z=zc_np, y=y, factor_names=factors, seed=seed, n_bins=n_bins),
        "zs": _evaluate_one_space(z=zs_np, y=y, factor_names=factors, seed=seed, n_bins=n_bins),
        "za": _evaluate_one_space(z=za_np, y=y, factor_names=factors, seed=seed, n_bins=n_bins),
    }

    out: dict[str, Any] = {
        "config": {
            "model_path": str(model_path),
            "tracks_path": str(tracks_path),
            "metadata_csv": str(metadata_csv),
            "factors": factors,
            "seed": int(seed),
            "n_bins": int(n_bins),
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
            "# Disentanglement Report (MIG / DCI / SAP)",
            "",
            f"- samples used: `{int(mask.sum())}` / total `{int(tracks.track_id.shape[0])}`",
            f"- factors: `{', '.join(factors)}`",
            "",
            "| space | MIG | DCI_dis | DCI_comp | DCI_info | SAP |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name in ["zc", "zs", "za"]:
            r = spaces[name]
            lines.append(
                f"| {name} | {r['MIG']:.6f} | {r['DCI_disentanglement']:.6f} | "
                f"{r['DCI_completeness']:.6f} | {r['DCI_informativeness']:.6f} | {r['SAP']:.6f} |"
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_bins", type=int, default=20)
    ap.add_argument("--prefer_cuda", action="store_true")
    args = ap.parse_args()

    factor_cols = [x.strip() for x in str(args.factors).split(",") if x.strip()]
    out = evaluate_disentanglement(
        model_path=str(args.model),
        tracks_path=str(args.tracks),
        metadata_csv=str(args.metadata),
        factor_cols=factor_cols,
        out_json=str(args.out_json) if args.out_json else None,
        out_md=str(args.out_md) if args.out_md else None,
        seed=int(args.seed),
        n_bins=int(args.n_bins),
        prefer_cuda=bool(args.prefer_cuda),
    )
    tiny = {
        "n_samples_used": out["n_samples_used"],
        "spaces": {
            k: {
                "MIG": v["MIG"],
                "DCI_disentanglement": v["DCI_disentanglement"],
                "DCI_completeness": v["DCI_completeness"],
                "DCI_informativeness": v["DCI_informativeness"],
                "SAP": v["SAP"],
            }
            for k, v in out["spaces"].items()
        },
    }
    print(json.dumps(tiny, ensure_ascii=False))


if __name__ == "__main__":
    main()

