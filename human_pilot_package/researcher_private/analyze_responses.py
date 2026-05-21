from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CHOICE_COLUMNS = ["compatible_choice", "discovery_choice", "overall_choice"]


def load_responses(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["response_file"] = path.name
        frames.append(df)
    if not frames:
        raise SystemExit("No response CSV files found.")
    return pd.concat(frames, ignore_index=True)


def method_for_choice(row: pd.Series, choice: str) -> str:
    if choice == "A":
        return str(row.get("candidate_a_method", ""))
    if choice == "B":
        return str(row.get("candidate_b_method", ""))
    return "Tie"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CHOICE_COLUMNS:
        total = 0
        pal = 0
        baseline = 0
        tie = 0
        missing = 0
        for _, row in df.iterrows():
            choice = str(row.get(col, "")).strip()
            if not choice or choice.lower() == "nan":
                missing += 1
                continue
            total += 1
            method = method_for_choice(row, choice)
            if method == "Tie":
                tie += 1
            elif method.startswith("pal_"):
                pal += 1
            elif method.startswith("baseline_"):
                baseline += 1
        denom_no_tie = pal + baseline
        denom_half_tie = pal + baseline + tie
        rows.append(
            {
                "question": col,
                "answered": total,
                "pal_wins": pal,
                "baseline_wins": baseline,
                "ties": tie,
                "missing": missing,
                "pal_win_rate_excluding_ties": pal / denom_no_tie if denom_no_tie else float("nan"),
                "pal_win_rate_tie_as_half": (pal + 0.5 * tie) / denom_half_tie if denom_half_tie else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--responses",
        default="responses",
        help="Folder containing volunteer CSV exports, or a single CSV file.",
    )
    parser.add_argument(
        "--key",
        default="method_key_private_counterbalanced_all.csv",
        help="Private method key CSV. Use method_key_private.csv only for the shared non-counterbalanced package.",
    )
    parser.add_argument("--out", default="summary_human_pilot.csv", help="Output summary CSV.")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    response_path = (base / args.responses).resolve()
    key_path = (base / args.key).resolve()
    out_path = (base / args.out).resolve()

    if response_path.is_dir():
        paths = sorted(response_path.glob("*.csv"))
    else:
        paths = [response_path]

    responses = load_responses(paths)
    key = pd.read_csv(key_path, encoding="utf-8-sig")
    if "participant_id" in key.columns:
        merged = responses.merge(key, on=["participant_id", "task_id"], how="left", suffixes=("", "_key"))
    else:
        merged = responses.merge(key, on="task_id", how="left", suffixes=("", "_key"))
    summary = summarize(merged)
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print(f"Saved summary to: {out_path}")


if __name__ == "__main__":
    main()
