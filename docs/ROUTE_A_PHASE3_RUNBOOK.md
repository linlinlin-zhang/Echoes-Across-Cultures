# Route A Phase 3 Runbook

Date: 2026-02-27

Goal:
- run PAL feedback injection for 2 rounds
- run ablation (domain / constraints / OT)
- run MIG/DCI/SAP disentanglement evaluation

## 1) PAL two-round loop

```bash
python -m dcas.scripts.run_phase3_pal \
  --tracks ./storage/public/routeA_phase2_cn/tracks.npz \
  --interactions ./storage/public/routeA_phase2_cn/interactions.csv \
  --metadata ./storage/public/routeA_phase2_cn/metadata_merged.csv \
  --baseline_model ./storage/public/routeA_phase2_cn/model.pt \
  --out_dir ./reports/routeA_phase3_pal_cn \
  --artifacts_dir ./storage/pal/routeA_phase3_cn \
  --model_dir ./storage/public/routeA_phase3_cn \
  --rounds 2 \
  --tasks_per_round 120 \
  --label_col label \
  --epochs 10 \
  --batch_size 64 \
  --lr 2e-3 \
  --seed 42 \
  --lambda_constraints 0.2 \
  --constraint_margin 1.0 \
  --lambda_domain 0.5 \
  --lambda_contrast 0.2 \
  --lambda_cov 0.05 \
  --k 10
```

Outputs:
- `reports/routeA_phase3_pal_cn/phase3_pal_summary.{json,md}`
- `storage/pal/routeA_phase3_cn/constraints_upto_round2.jsonl`
- `storage/public/routeA_phase3_cn/round{1,2}_model.pt`

## 2) Ablation (domain / constraints / OT)

```bash
python -m dcas.scripts.run_ablation \
  --tracks ./storage/public/routeA_phase2_cn/tracks.npz \
  --interactions ./storage/public/routeA_phase2_cn/interactions.csv \
  --constraints ./storage/pal/routeA_phase3_cn/constraints_upto_round2.jsonl \
  --out_dir ./reports/routeA_phase3_ablation_cn \
  --model_dir ./storage/public/routeA_phase3_ablation_cn \
  --epochs 10 \
  --batch_size 64 \
  --lr 2e-3 \
  --seed 42 \
  --lambda_constraints 0.2 \
  --constraint_margin 1.0 \
  --lambda_domain 0.5 \
  --lambda_contrast 0.2 \
  --lambda_cov 0.05 \
  --k 10
```

Outputs:
- `reports/routeA_phase3_ablation_cn/ablation_summary.json`
- `reports/routeA_phase3_ablation_cn/ablation_table_draft.md`

## 3) MIG/DCI/SAP evaluation

Baseline:
```bash
python -m dcas.scripts.evaluate_disentanglement \
  --model ./storage/public/routeA_phase2_cn/model.pt \
  --tracks ./storage/public/routeA_phase2_cn/tracks.npz \
  --metadata ./storage/public/routeA_phase2_cn/metadata_merged.csv \
  --factors culture,label \
  --out_json ./reports/routeA_disentanglement_phase2_cn.json \
  --out_md ./reports/routeA_disentanglement_phase2_cn.md
```

PAL round2:
```bash
python -m dcas.scripts.evaluate_disentanglement \
  --model ./storage/public/routeA_phase3_cn/round2_model.pt \
  --tracks ./storage/public/routeA_phase2_cn/tracks.npz \
  --metadata ./storage/public/routeA_phase2_cn/metadata_merged.csv \
  --factors culture,label \
  --out_json ./reports/routeA_disentanglement_phase3_round2_cn.json \
  --out_md ./reports/routeA_disentanglement_phase3_round2_cn.md
```

Optional compare table:
- `reports/routeA_disentanglement_compare_cn.md`

## Notes
- Current PAL feedback is simulated by metadata label agreement (`same label => similar`).
- Replace with real expert annotation jsonl before final paper claims.
