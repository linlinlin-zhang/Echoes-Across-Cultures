# Reviewer Prebuttal Drill

Date: 2026-03-22

## Reviewer 1

### Concern
`The baseline suite seems unfair. The main draft compares against cosine retrieval and a shallow MLP, but omits stronger two-stage and listwise rerankers. This makes it hard to know whether the gains come from the proposed framework or from weak baselines.`

### Judgment
Yes, this is a real defect in the current compiled draft.

### Required revision
- Replace the old baseline subsection and main table in [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex):235-247 and 300-316.
- Use the actual V4 suite from [recommender_benchmark_v4_main_culturemert_stage3_lambdamart.run.json](E:/Desktop/Echo/configs/benchmark/recommender_benchmark_v4_main_culturemert_stage3_lambdamart.run.json):109-187.
- Present the strongest hybrid baseline explicitly, especially `bpr_lambdamart_hybrid`, and report paired bootstrap deltas against the target-calibrated operating point.

### Rebuttal stance
No rebuttal as-is. Fix in manuscript.

## Reviewer 2

### Concern
`The paper claims cross-cultural recommendation gains, but the dataset still appears heavily confounded by source. How do we know the reported effects are not mostly source artifacts?`

### Judgment
Partly yes. This is a valid limitation, but it does not invalidate the narrower claim if we state it correctly.

### Required revision
- Add an explicit limitation paragraph in the main manuscript using the current V4 evidence: `weighted_source_predictability_from_culture = 0.911765` for V4 main and `1.0` for routeA_small.
- Keep `routeA_small` labeled as a `sanity-check track`, not primary evidence.
- Add one sentence in Results explaining that the remaining asymmetry is interpreted as data-level confounding rather than universal model failure.

### Rebuttal draft
Thank you for raising this important point. We agree that source confound remains substantial and have revised the manuscript to state this explicitly. Our claim is therefore not that source bias has been solved, but that the proposed downstream stack yields a more controllable calibration-exposure trade-off under source-confound-aware evaluation. We also narrow the evidential status of `routeA_small` to a sanity-check track and reserve our primary empirical claim for `V4 main`.

## Reviewer 3

### Concern
`The PAL contribution is overclaimed. The manuscript mentions human feedback, but the current evidence does not yet establish a completed real human-in-the-loop study.`

### Judgment
Yes, the current compiled draft overreaches in places.

### Required revision
- Replace `pilot human-in-the-loop setting` and similar phrases with `PAL-ready` or `execution-ready` where appropriate.
- Move any discussion of real PAL benefit from the main result claim into limitations or future work unless and until a real round is reported.
- Keep the contribution framed as an operational workflow: uncertainty ranking, annotation packet generation, pairwise constraints, and warm-start retraining.

### Rebuttal draft
We appreciate this clarification request. We have revised the manuscript to narrow the PAL claim from a completed human-study contribution to an execution-ready workflow contribution. The revised text no longer treats PAL as a finished real-feedback evaluation, and instead presents it as a concrete mechanism for integrating targeted expert corrections into stage-3 retraining.

## Reviewer 4

### Concern
`The method section does not justify the computational cost of OT and uncertainty-guided PAL. The complexity discussion is missing, especially in the worst case.`

### Judgment
Yes, this is a missing explanation rather than a fatal defect.

### Required revision
- Add a short complexity paragraph to Method or Discussion based on the phase-2 archaeology notes.
- State the practical dependence on candidate-set size, latent dimension, and Sinkhorn iterations.
- Clarify that the reported system is designed for medium-scale cultural catalogs rather than web-scale deployment.

### Rebuttal draft
Thank you. We agree that the original draft underexplained computational cost. We have added a complexity paragraph that makes the practical scaling assumptions explicit and revised the manuscript to position the current implementation as a medium-scale cultural-catalog system rather than a web-scale solution.

## Reviewer 5

### Concern
`The paper seems to mix several contributions: factorized latent learning, calibration-aware reranking, and human feedback. It is unclear whether the gains really require the full stack or are driven mostly by the reranking weights.`

### Judgment
Partly yes. The concern is reasonable, but the current evidence can answer it if organized more clearly.

### Required revision
- Keep the module-removal ablation and calibration sensitivity study adjacent in the Results section.
- Explicitly distinguish two questions:
  - Which modules are necessary for the quality-exposure balance?
  - Which operating point should be chosen at inference time?
- Use the constraint-removal result to show that higher serendipity alone is not the whole story, and the calibration sweep to show that reranking is a controllable layer rather than the only source of improvement.

### Rebuttal draft
We agree that these roles should be separated more clearly. In the revised manuscript, we distinguish module necessity from operating-point selection. The module-removal ablation shows that constraints and domain shaping materially affect the exposure-quality balance, while the calibration sweep shows that inference-time reranking controls a smooth trade-off rather than replacing the need for the rest of the framework.

## Immediate Author Actions

1. Merge the new Introduction, Related Work, Method, and Results drafts into the main manuscript.
2. Replace all placeholder tables before further stylistic polishing.
3. Add a short complexity paragraph and a stronger limitations paragraph on source confound.
4. Keep all PAL claims at `PAL-ready` unless a real annotated round is actually reported.
5. Strengthen the baseline rationale with the current V4 benchmark suite.
