# ISMIR 2026 Abstract Packet / 摘要提交包

This file is a paste-ready abstract-submission companion for the current manuscript.
本文件用于明天在 CMT 中提交摘要，确保题目、摘要、主题方向和论文正文保持一致。

Official submission constraints checked on 2026-04-18:

- Abstract deadline: 2026-04-20 AoE.
- Full paper deadline: 2026-04-27 AoE.
- CMT metadata cannot be modified after the abstract deadline.
- The PDF title must exactly match the CMT title.
- ISMIR 2026 uses double-blind review.
- Full papers use a 6+N format: 6 pages of scientific content, with extra pages only for references, optional ethics statement, optional AI usage statement, and post-acceptance acknowledgements.
- AI usage does not need declaration for grammar/editing only, but must be declared if LLMs materially shape literature review, result analysis, methodology, or other scientific content.

Sources:

- https://ismir2026.ismir.net/authors/call-for-papers
- https://ismir2026.ismir.net/authors/author-guidelines
- https://ismir2026.ismir.net/ai-usage-policy

## Recommended CMT Title

ResonanceRec: Controllable Cross-Cultural Music Recommendation via Optimal-Transport Reranking and Human Feedback

## Paste-Ready Abstract

Cross-cultural music recommendation is difficult because listeners often want music from unfamiliar cultures that still fits a target mood, style, function, or listening context. We present ResonanceRec, a backbone-agnostic framework that freezes pretrained audio backbones and learns a lightweight recommendation layer with factorized latent modeling, stage-wise training, optimal-transport (OT) retrieval, calibration-aware reranking, and participatory active learning (PAL). Candidates are retrieved by OT relevance and reranked with explicit controls for target-culture affinity, minority exposure at k, source balance, and novelty. On the V4 main dataset (1,122 tracks, 10 cultures, 8 sources), the CultureMERT target-calibrated operating point improves serendipity from 0.5558 to 0.8316 and minority exposure at k from 0.2681 to 0.4023 over the strongest hybrid baseline, while reducing cultural-calibration KL from 2.0966 to 2.0296. Similar trade-offs appear with Gemini embeddings, suggesting that the framework is not tied to a single backbone. We also incorporate a 200-pair human PAL round from the V4 annotation packet. After conflict handling, 188 valid pairwise constraints are used to warm-start retraining and calibration sweeps; the best PAL-calibrated settings further increase minority exposure by 0.0392-0.0616 and reduce KL by 0.0085-0.0098 relative to the non-PAL target-calibrated reference, while preserving or slightly improving serendipity. The resulting downstream checkpoints are only a few megabytes. ResonanceRec is thus a lightweight, controllable recommendation framework, though residual source confounding in V4 remains an important limitation.

## Recommended Subject Areas

- Applications: music recommendation and playlist generation.
- Evaluation, datasets, and reproducibility: evaluation methodology, metrics, annotation protocols.
- Musical features and properties: representation learning, musical style and genre, musical affect, emotion and mood.
- Responsibility and Ethics in MIR: cultural coverage, exposure, and responsible recommendation.
- Cognitive and user-centered MIR: user-centered evaluation and human feedback.

## Claims To Keep

- The framework is backbone-agnostic because CultureMERT and Gemini use the same downstream interface.
- The system is lightweight at the modeling/deployment layer because the backbone is frozen and checkpoints are only a few MB.
- The main contribution is a controllable calibration-exposure-serendipity frontier, not a single universal winning score.
- PAL is a small human feedback pilot that helps tune the frontier, not a completed large-scale user study.
- Source confound remains a limitation and should be disclosed proactively.

## Claims To Avoid

- Do not claim that the system solves cross-cultural music understanding.
- Do not claim that PAL proves population-level listener preference.
- Do not claim that all PAL variants improve all metrics.
- Do not claim that routeA_small is as strong as V4 main evidence.
- Do not describe synthetic interactions as real user behavior.

## Rapid Human Evaluation Options

Recommendation: do Option A immediately, and only do Option B if there are enough trusted volunteers tonight.

Option A: strengthen the existing PAL analysis.

- Use the completed 200 PAL pairs.
- Report duplicate rate, conflict rate, usable constraint count, positive/negative balance, and culture/source coverage.
- Add 2-3 anonymized qualitative examples if they do not reveal authorship or copyrighted content.
- This is safest because it uses existing data and does not introduce a rushed user-study protocol.

Option B: tiny blind listening sanity check.

- Recruit 5-10 adult volunteers.
- Use 12-20 seed contexts.
- For each seed/context, show two anonymous recommendation candidates or two short ranked lists: one from the strongest baseline and one from DCAS/PAL.
- Ask two 5-point questions: "Which recommendation feels emotionally/style compatible?" and "Which recommendation feels culturally novel but still listenable?"
- Randomize order and hide method names.
- Collect optional one-sentence comments.
- Report only as an informal pilot or sanity check unless consent, data handling, and protocol are clean.

Option C: web-platform data collection.

- Mention as future work or ongoing infrastructure unless real users and cleaned logs are already collected.
- Do not use an unfinished web pilot as a central empirical claim for the ISMIR submission.

Not recommended: generating new interaction data with AI agents for the main claim.

- It may make the synthetic-interaction weakness worse because reviewers can question whether the agent preferences reflect human listening behavior.
- If used at all, agents should be clearly labeled as simulation/stress testing, not as replacement human evidence.
