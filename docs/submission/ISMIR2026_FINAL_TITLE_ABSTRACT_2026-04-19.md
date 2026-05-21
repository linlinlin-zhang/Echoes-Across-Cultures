# ISMIR 2026 Final Title and Abstract

Use this version for the CMT abstract submission.
Do not include the blind-listener pilot in the abstract; keep it as internal/rebuttal material unless the full paper later needs it.

## Final Title

ResonanceRec: Controllable Cross-Cultural Music Recommendation via Optimal-Transport Reranking and Human Feedback

## Final Abstract

Cross-cultural music recommendation is difficult because listeners often want music from unfamiliar cultures that still fits a target mood, style, function, or listening context. We present ResonanceRec, a backbone-agnostic framework that freezes pretrained audio backbones and learns a lightweight recommendation layer with factorized latent modeling, stage-wise training, optimal-transport (OT) retrieval, calibration-aware reranking, and participatory active learning (PAL). Candidates are retrieved by OT relevance and reranked with explicit controls for target-culture affinity, minority exposure at k, source balance, and novelty. On the V4 main dataset (1,122 tracks, 10 cultures, 8 sources), the CultureMERT target-calibrated operating point improves serendipity from 0.5558 to 0.8316 and minority exposure at k from 0.2681 to 0.4023 over the strongest hybrid baseline, while reducing cultural-calibration KL from 2.0966 to 2.0296. Similar trade-offs appear with Gemini embeddings, suggesting that the framework is not tied to a single backbone. We also incorporate a 200-pair human PAL round from the V4 annotation packet. After conflict handling, 188 valid pairwise constraints are used to warm-start retraining and calibration sweeps; the best PAL-calibrated settings further increase minority exposure by 0.0392-0.0616 and reduce KL by 0.0085-0.0098 relative to the non-PAL target-calibrated reference, while preserving or slightly improving serendipity. The resulting downstream checkpoints are only a few megabytes. ResonanceRec is thus a lightweight, controllable recommendation framework, though residual source confounding in V4 remains an important limitation.

## Recommended Subject Areas

Primary:

- music recommendation and playlist generation

Secondary:

- evaluation methodology
- evaluation metrics
- annotation protocols
- user-centered evaluation
- representation learning

## Claims To Keep Stable

- This is a frozen-backbone downstream recommendation framework, not a new foundation model.
- The main claim is a controllable calibration-exposure-serendipity frontier.
- The 200-pair PAL round is real human pairwise feedback, but it is not a large population-level user study.
- The blind-listener pilot is not included in the abstract.
- Source confound and synthetic interactions remain limitations.
