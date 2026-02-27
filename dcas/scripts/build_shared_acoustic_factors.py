from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio


def _load_mono(path: str | Path, target_sr: int, max_seconds: float | None) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.float()
    if wav.ndim != 2:
        raise ValueError("waveform must be [channels, samples]")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if int(sr) != int(target_sr):
        wav = torchaudio.functional.resample(wav, orig_freq=int(sr), new_freq=int(target_sr))
    if max_seconds is not None and float(max_seconds) > 0:
        n = int(round(float(max_seconds) * int(target_sr)))
        wav = wav[..., :n]
    if int(wav.shape[-1]) <= 32:
        raise ValueError("audio too short")
    return wav


def _spectral_features(
    wav: torch.Tensor,
    sr: int,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> dict[str, float]:
    x = wav.squeeze(0)
    rms = float(torch.sqrt(torch.mean(x**2) + 1e-12).item())

    sign = torch.sign(x)
    zcr = float((torch.abs(sign[1:] - sign[:-1]) > 0).float().mean().item())

    window = torch.hann_window(int(win_length), dtype=x.dtype, device=x.device)
    spec = torch.stft(
        x,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(win_length),
        window=window,
        return_complex=True,
    )
    mag = torch.abs(spec).clamp_min(1e-8)

    freqs = torch.linspace(0.0, float(sr) / 2.0, steps=mag.shape[0], dtype=mag.dtype, device=mag.device)
    centroid = ((freqs[:, None] * mag).sum(dim=0) / mag.sum(dim=0).clamp_min(1e-8)).mean()

    flatness_frame = torch.exp(torch.mean(torch.log(mag), dim=0)) / torch.mean(mag, dim=0).clamp_min(1e-8)
    flatness = flatness_frame.mean()

    mag_norm = mag / mag.sum(dim=0, keepdim=True).clamp_min(1e-8)
    if mag_norm.shape[1] > 1:
        flux = torch.norm(mag_norm[:, 1:] - mag_norm[:, :-1], dim=0).mean()
    else:
        flux = torch.tensor(0.0, dtype=mag.dtype, device=mag.device)

    return {
        "rms": float(rms),
        "zcr": float(zcr),
        "centroid_hz": float(centroid.item()),
        "flatness": float(flatness.item()),
        "flux": float(flux.item()),
    }


def _quantile_bins(values: np.ndarray, n_bins: int = 3, labels: tuple[str, ...] | None = None) -> tuple[np.ndarray, list[float]]:
    vals = values.astype(np.float64)
    n_bins = max(2, int(n_bins))
    edges = np.quantile(vals, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.shape[0] <= 2:
        bins = np.zeros((vals.shape[0],), dtype=np.int64)
        used_edges = [float(edges[0]), float(edges[-1])]
    else:
        bins = np.digitize(vals, edges[1:-1], right=False).astype(np.int64)
        used_edges = [float(x) for x in edges.tolist()]

    if labels is None:
        labels = tuple(f"bin{i}" for i in range(n_bins))
    label_arr = np.array([labels[min(int(b), len(labels) - 1)] for b in bins.tolist()], dtype=object)
    return label_arr, used_edges


def build_shared_acoustic_factors(
    metadata_csv: str,
    out_csv: str,
    report_json: str | None = None,
    target_sr: int = 16000,
    max_seconds: float | None = 10.0,
    n_bins: int = 3,
) -> dict[str, Any]:
    meta_path = Path(metadata_csv)
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata not found: {meta_path}")

    rows: list[dict[str, str]] = []
    with open(meta_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for required in ("track_id", "culture", "audio_path"):
            if required not in cols:
                raise ValueError(f"missing required column: {required}")
        for r in reader:
            rows.append({str(k): str(v) for k, v in r.items()})

    feats: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for r in rows:
        tid = str(r["track_id"]).strip()
        cul = str(r["culture"]).strip()
        ap = str(r["audio_path"]).strip()
        if not tid or not ap:
            failures.append({"track_id": tid, "reason": "empty_track_or_audio_path"})
            continue
        p = Path(ap)
        if not p.exists():
            failures.append({"track_id": tid, "reason": "audio_not_found", "audio_path": ap})
            continue
        try:
            wav = _load_mono(path=p, target_sr=int(target_sr), max_seconds=max_seconds)
            fts = _spectral_features(wav=wav, sr=int(target_sr))
            feats.append(
                {
                    "track_id": tid,
                    "culture": cul,
                    "audio_path": ap,
                    **fts,
                }
            )
        except Exception as e:
            failures.append({"track_id": tid, "reason": str(e), "audio_path": ap})

    if not feats:
        raise RuntimeError("no usable audio rows for shared factor extraction")

    arr_rms = np.array([float(x["rms"]) for x in feats], dtype=np.float64)
    arr_zcr = np.array([float(x["zcr"]) for x in feats], dtype=np.float64)
    arr_cent = np.array([float(x["centroid_hz"]) for x in feats], dtype=np.float64)
    arr_flat = np.array([float(x["flatness"]) for x in feats], dtype=np.float64)
    arr_flux = np.array([float(x["flux"]) for x in feats], dtype=np.float64)

    # Shared categorical factors (cross-cultural).
    energy, e_edges = _quantile_bins(arr_rms, n_bins=n_bins, labels=("low", "mid", "high"))
    brightness, b_edges = _quantile_bins(arr_cent, n_bins=n_bins, labels=("dark", "mid", "bright"))
    texture, t_edges = _quantile_bins(arr_flat, n_bins=n_bins, labels=("tonal", "mixed", "noisy"))
    dynamics, d_edges = _quantile_bins(arr_flux, n_bins=n_bins, labels=("stable", "mixed", "dynamic"))
    percussive, p_edges = _quantile_bins(arr_zcr, n_bins=n_bins, labels=("smooth", "mixed", "percussive"))

    for i, row in enumerate(feats):
        row["factor_energy"] = str(energy[i])
        row["factor_brightness"] = str(brightness[i])
        row["factor_texture"] = str(texture[i])
        row["factor_dynamics"] = str(dynamics[i])
        row["factor_percussiveness"] = str(percussive[i])

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "track_id",
        "culture",
        "audio_path",
        "factor_energy",
        "factor_brightness",
        "factor_texture",
        "factor_dynamics",
        "factor_percussiveness",
        "rms",
        "zcr",
        "centroid_hz",
        "flatness",
        "flux",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in feats:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    def _dist(key: str) -> dict[str, int]:
        vals = [str(r[key]) for r in feats]
        out: dict[str, int] = {}
        for v in vals:
            out[v] = int(out.get(v, 0) + 1)
        return dict(sorted(out.items(), key=lambda kv: kv[0]))

    report = {
        "status": "pass" if len(failures) == 0 else "warn",
        "metadata_csv": str(meta_path),
        "out_csv": str(out_path),
        "n_rows_total": int(len(rows)),
        "n_rows_ok": int(len(feats)),
        "n_rows_failed": int(len(failures)),
        "target_sr": int(target_sr),
        "max_seconds": None if max_seconds is None else float(max_seconds),
        "factor_distributions": {
            "factor_energy": _dist("factor_energy"),
            "factor_brightness": _dist("factor_brightness"),
            "factor_texture": _dist("factor_texture"),
            "factor_dynamics": _dist("factor_dynamics"),
            "factor_percussiveness": _dist("factor_percussiveness"),
        },
        "bin_edges": {
            "factor_energy_rms": e_edges,
            "factor_brightness_centroid_hz": b_edges,
            "factor_texture_flatness": t_edges,
            "factor_dynamics_flux": d_edges,
            "factor_percussiveness_zcr": p_edges,
        },
        "failures_preview": failures[:50],
    }
    if report_json is not None:
        rp = Path(report_json)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Build cross-cultural shared acoustic factors from audio metadata.")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--report_json", default=None)
    ap.add_argument("--target_sr", type=int, default=16000)
    ap.add_argument("--max_seconds", type=float, default=10.0)
    ap.add_argument("--n_bins", type=int, default=3)
    args = ap.parse_args()

    out = build_shared_acoustic_factors(
        metadata_csv=str(args.metadata),
        out_csv=str(args.out_csv),
        report_json=str(args.report_json) if args.report_json else None,
        target_sr=int(args.target_sr),
        max_seconds=float(args.max_seconds) if args.max_seconds and float(args.max_seconds) > 0 else None,
        n_bins=int(args.n_bins),
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

