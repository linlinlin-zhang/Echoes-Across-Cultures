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
  --k 10 \
  --bootstrap_samples 5000 \
  --permutation_samples 5000
```

Outputs:
- `reports/routeA_phase3_pal_cn/phase3_pal_summary.{json,md}`
- `reports/routeA_phase3_pal_cn/compare_baseline_vs_round*.{json,md}`
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
  --k 10 \
  --bootstrap_samples 5000 \
  --permutation_samples 5000
```

Outputs:
- `reports/routeA_phase3_ablation_cn/ablation_summary.json`
- `reports/routeA_phase3_ablation_cn/ablation_table_draft.md`
- `reports/routeA_phase3_ablation_cn/compare_full_vs_*.{json,md}`

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

## 4) Optional standalone significance test

```bash
python -m dcas.scripts.compare_recommender_runs \
  --base_json ./reports/routeA_phase2_cn_eval.json \
  --candidate_json ./reports/routeA_phase4_tc_hsic_eval.json \
  --metrics serendipity,cultural_calibration_kl \
  --bootstrap_samples 5000 \
  --permutation_samples 5000 \
  --out_json ./reports/routeA_compare_phase2_vs_phase4_significance.json \
  --out_md ./reports/routeA_compare_phase2_vs_phase4_significance.md
```

## Notes
- Current PAL feedback is simulated by metadata label agreement (`same label => similar`).
- Replace with real expert annotation jsonl before final paper claims.

## 5) Phase-2: TC/HSIC Sensitivity (Grid + Stability)

```bash
python -m dcas.scripts.run_tc_hsic_sensitivity \
  --tracks ./storage/public/routeA_phase2_cn/tracks.npz \
  --interactions ./storage/public/routeA_phase2_cn/interactions.csv \
  --out_dir ./reports/routeA_phase2_tc_hsic_sensitivity \
  --model_dir ./storage/public/routeA_phase2_tc_hsic_sensitivity \
  --tc_values 0.0,0.02,0.05,0.1 \
  --hsic_values 0.0,0.01,0.02,0.05 \
  --seeds 42,43,44 \
  --epochs 10 \
  --batch_size 64 \
  --lr 2e-3 \
  --k 10 \
  --bootstrap_samples 2000
```

Outputs:
- `reports/routeA_phase2_tc_hsic_sensitivity/tc_hsic_sensitivity_summary.json`
- `reports/routeA_phase2_tc_hsic_sensitivity/tc_hsic_sensitivity_summary.md`

## 6) Phase-3: Baseline Comparison (VAE/beta-VAE/FactorVAE)

```bash
python -m dcas.scripts.run_baseline_comparison \
  --tracks ./storage/public/routeA_phase2_cn/tracks.npz \
  --interactions ./storage/public/routeA_phase2_cn/interactions.csv \
  --out_dir ./reports/routeA_phase3_baselines \
  --model_dir ./storage/public/routeA_phase3_baselines \
  --seeds 42,43,44 \
  --epochs 10 \
  --batch_size 64 \
  --lr 2e-3 \
  --lambda_tc 0.05 \
  --lambda_hsic 0.02 \
  --beta_vae_beta 4.0 \
  --factorvae_lambda_tc 0.1 \
  --k 10 \
  --bootstrap_samples 2000 \
  --permutation_samples 2000
```

Outputs:
- `reports/routeA_phase3_baselines/baseline_comparison_summary.json`
- `reports/routeA_phase3_baselines/baseline_comparison_table_draft.md`
