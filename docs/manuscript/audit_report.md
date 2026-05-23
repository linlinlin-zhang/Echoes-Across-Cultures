# Manuscript Audit Report

Date: 2026-03-22

Scope:
- [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex)
- [RELATED_WORK_DRAFT_2026-03-22_EN.tex](E:/Desktop/Echo/docs/manuscript/RELATED_WORK_DRAFT_2026-03-22_EN.tex)
- [INTRODUCTION_DRAFT_2026-03-21_EN.tex](E:/Desktop/Echo/docs/manuscript/INTRODUCTION_DRAFT_2026-03-21_EN.tex)
- [RESULTS_SECTION_DRAFT_2026-03-21_EN.tex](E:/Desktop/Echo/docs/manuscript/RESULTS_SECTION_DRAFT_2026-03-21_EN.tex)
- [METHOD_MODULE_A_PROBLEM_FORMULATION_2026-03-21.tex](E:/Desktop/Echo/docs/manuscript/METHOD_MODULE_A_PROBLEM_FORMULATION_2026-03-21.tex)
- [CODE_PAPER_TRACEABILITY_2026-03-21.csv](E:/Desktop/Echo/docs/manuscript/CODE_PAPER_TRACEABILITY_2026-03-21.csv)
- [CLAIM_EVIDENCE_MAP_2026-03-21.csv](E:/Desktop/Echo/docs/manuscript/CLAIM_EVIDENCE_MAP_2026-03-21.csv)
- [train_v4_main_culturemert_stage3.run.json](E:/Desktop/Echo/configs/train/train_v4_main_culturemert_stage3.run.json)
- [train_v4_main_gemini_stage3.run.json](E:/Desktop/Echo/configs/train/train_v4_main_gemini_stage3.run.json)
- [recommender_benchmark_v4_main_culturemert_stage3_lambdamart.run.json](E:/Desktop/Echo/configs/benchmark/recommender_benchmark_v4_main_culturemert_stage3_lambdamart.run.json)
- [README.md](E:/Desktop/Echo/docs/research_dataset_v4/README.md)

Overall verdict:
- Status: `mostly pass`
- Main improvement since last audit: the compiled manuscript is no longer a placeholder-only planning draft. Introduction, Related Work, Method, Experimental Setup, and Results now reflect the current V4 storyline and real evidence.
- Current blockers before submission:
  - LaTeX compilation was not verified because `pdflatex` is not installed in the current environment.
  - The manuscript is evidence-backed, but still light on figure polish and does not yet include item-level qualitative panels.
  - PAL is framed correctly as `PAL-ready`, but a real expert-feedback study is still future work.

## 1. Code-Paper Consistency

### Check 1.1 Method steps vs. implementation
- [x] The manuscript now reflects the full stage-3 objective hierarchy.
  - Main text includes `recon`, `KL`, `domain`, `contrast`, `cov`, `tc`, `hsic`, `affect`, `source`, plus delayed `pair` and `rank` terms in [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex#L189).
  - This matches the code-backed formulation in [METHOD_MODULE_A_PROBLEM_FORMULATION_2026-03-21.tex](E:/Desktop/Echo/docs/manuscript/METHOD_MODULE_A_PROBLEM_FORMULATION_2026-03-21.tex#L75) and [dcas_vae.py](E:/Desktop/Echo/dcas/models/dcas_vae.py#L194).
- [x] The inference description now matches the calibration-aware reranker.
  - The scoring function now includes relevance, novelty, target affinity, minority boost, and source boost in [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex#L258).
  - This aligns with [recommender.py](E:/Desktop/Echo/dcas/recommender.py#L354) and the benchmark config operating-point weights in [recommender_benchmark_v4_main_culturemert_stage3_lambdamart.run.json](E:/Desktop/Echo/configs/benchmark/recommender_benchmark_v4_main_culturemert_stage3_lambdamart.run.json#L165).

### Check 1.2 Claimed hyperparameters vs. config files
- [x] Training hyperparameters are now config-backed.
  - Main draft reports AdamW, `lr = 0.002`, `batch size = 128`, `10 epochs`, delayed constraints, delayed ranking, and seed 42 in [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex#L233).
  - These match [train_v4_main_culturemert_stage3.run.json](E:/Desktop/Echo/configs/train/train_v4_main_culturemert_stage3.run.json#L6), [train_v4_main_gemini_stage3.run.json](E:/Desktop/Echo/configs/train/train_v4_main_gemini_stage3.run.json#L6), and [cli/train.py](E:/Desktop/Echo/dcas/cli/train.py#L101).
- [x] Latent dimensions now match the actual model defaults.
  - Main draft uses `(32, 32, 16)` in [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex#L175).
  - This matches [dcas_vae.py](E:/Desktop/Echo/dcas/models/dcas_vae.py#L28).

### Check 1.3 Dataset scale claims vs. data artifacts
- [x] The manuscript no longer uses the outdated `1600 tracks / 4 cultures` draft setup.
  - Current dataset section describes V4 main and routeA_small in [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex#L286).
  - This matches the V4 audit outputs and dataset inventory.
- [x] Source confound is now explicitly disclosed in the main text.
  - Main draft reports `0.911765` for V4 main and `1.0` for routeA_small in [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex#L305).

### Check 1.4 Baseline definitions vs. benchmark suite
- [x] The main paper now uses a defensible compressed baseline story.
  - Instead of pretending the full suite is four methods, the paper explains that the full suite exists but the main table focuses on the strongest hybrid baseline plus DCAS variants in [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex#L317).
- [ ] Residual caution: the paper should still mention in camera-ready or appendix that the broader benchmark suite also includes `popularity`, `cosine`, `knn`, `lightfm_like`, `bpr_mf`, `bpr_two_stage_hybrid`, and `bpr_listwise_hybrid`.

## 2. Logic Chain Completeness

### Check 2.1 Intro -> Method -> Results loop
- [x] The main storyline is now closed.
  - Introduction states the gap and contributions in [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex#L41).
  - Method explains the DCAS stack and stage-3 curriculum in [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex#L130).
  - Results validate the calibration-exposure-serendipity trade-off in [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex#L327).

### Check 2.2 Figure/table usage
- [x] Major tables are now referenced in prose.
  - `Fig.~\ref{fig:pipeline}` is cited in the Introduction at [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex#L55).
  - `Table~\ref{tab:main}`, `Table~\ref{tab:ablation}`, and `Table~\ref{tab:sweep}` are all cited in the Results section.
- [x] `Table~\ref{tab:data}` is now also referenced explicitly in the Experimental Setup section.

## 3. Academic Norms

### Check 3.1 Statistical support
- [x] The paper no longer promises future significance testing as a placeholder.
  - It now states the current bootstrap/permutation protocol in [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex#L320).
  - Key pairwise contrasts report bootstrap CIs and selected permutation-supported p-values in [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex#L363).
- [ ] Residual caution: not every table cell carries its own CI, so the strongest claim framing should stay in the paired-comparison prose rather than in standalone cell-level interpretation.

### Check 3.2 Absolute numbers and metric definitions
- [x] Synthetic placeholder tables and fake PAL rows are gone from the compiled manuscript.
- [x] Metric definitions are aligned with the current locked terminology:
  - `serendipity`
  - `cultural calibration KL`
  - `minority exposure at k`
  - `source confound`

### Check 3.3 Honest disclosure of limitations
- [x] The manuscript is now honest about the present evidence boundary.
  - routeA_small is called a sanity-check track, not a main benchmark peer.
  - PAL is described as `PAL-ready`, not as a completed human study.
  - source confound is explicitly retained as a limitation.

## 4. Language and Format

### Check 4.1 Tense consistency
- [x] Most planning-language residue has been removed.
  - Old phrases such as `proof-of-concept`, `illustrative`, `final version will`, and `synthetic placeholder` are no longer present in the compiled manuscript.
- [ ] Residual caution: Results currently mix present-tense interpretation with past-tense experimental narration. This is acceptable, but a final language pass would still improve consistency.

### Check 4.2 Abbreviation discipline
- [x] `Music Information Retrieval (MIR)` is expanded on first in-body use in [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex#L39).
- [x] `optimal-transport (OT)` and `participatory active learning (PAL)` are introduced in the abstract.
- [ ] Residual caution: `DCAS` is used as a framework name without an expanded long form in-body. This is acceptable if intentional, but a short one-line introduction in camera-ready would improve readability.

### Check 4.3 Template / compilation status
- [x] Citation keys used in the manuscript exist in [refs.bib](E:/Desktop/Echo/paper/refs.bib).
- [ ] Compilation not verified.
  - `pdflatex` is not installed in the current environment, so no PDF build was run.

## Priority Fix List

1. Install a TeX engine and run a real compile pass on [ismir2026_draft.tex](E:/Desktop/Echo/paper/ismir2026_draft.tex).
2. Tighten one explicit in-text reference to [tab:data](E:/Desktop/Echo/paper/ismir2026_draft.tex#L294).
3. Optionally add a one-line long-form explanation for `DCAS` on first use in the Method section.
4. Add item-level qualitative figures or recommendation panels if the final page budget allows.
5. Keep PAL claims at `PAL-ready` unless a real expert round is completed and documented.

## Readiness Estimate

- For internal drafting: `ready`
- For advisor review: `ready`
- For external submission: `close, but compile + figure polish still needed`
