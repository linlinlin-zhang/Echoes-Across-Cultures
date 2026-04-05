# ISMIR2026 Section Drafts (2026-03-21)

## Introduction Draft Skeleton

Paragraph 1:

Cross-cultural music recommendation remains difficult even when strong audio foundation models are available. The central problem is not only retrieval accuracy, but the mismatch between what appears acoustically similar and what remains culturally meaningful, behaviorally relevant, and discovery-oriented for the listener.

Paragraph 2:

In practice, embedding-only retrieval tends to over-favor culturally dominant or stylistically familiar candidates. This is especially problematic in underrepresented repertoires, where cultural identity, source bias, and weak supervision can be entangled with the notion of relevance itself.

Paragraph 3:

We therefore frame cross-cultural recommendation as a structured downstream problem rather than a pure backbone problem. Instead of replacing the embedding model, we introduce a reusable pipeline that combines a unified dataset contract, a factorized downstream representation learner, calibration-aware reranking, and a PAL-ready human feedback loop.

Paragraph 4:

The framework is evaluated on two V4 dataset tracks, `V4 main` and `V4 routeA_small`, and on two heterogeneous embedding backbones, CultureMERT and Gemini. This setup allows us to test not only ranking quality, but also cultural calibration, minority-culture exposure, and the transferability of the downstream design across backbones.

Contributions:

1. We present a backbone-agnostic framework for culturally calibrated music recommendation that decouples representation learning, reranking, and human feedback under a unified V4 data contract.
2. We show that the calibration layer induces a controllable Pareto-style trade-off between serendipity, cultural calibration, and minority exposure across multiple V4 settings.
3. We operationalize a PAL-ready workflow that turns uncertainty-ranked candidate pairs into executable annotation packets and warm-start retraining hooks, enabling a realistic human-in-the-loop extension.

Validation route sentence:

We validate the framework through main benchmarks on `V4 main`, sanity-check benchmarks on `V4 routeA_small`, backbone transfer between CultureMERT and Gemini, structural ablations, and calibration hyperparameter sweeps.

## Related Work Draft Skeleton

### Foundation Audio Models And Cross-Cultural MIR

This section should position CultureMERT and related music foundation models as strong upstream front ends, while making clear that the paper does not propose a new foundation model. The gap is the downstream structure required for cross-cultural recommendation once embeddings already exist.

### Disentanglement, Invariance, And Modular Latent Learning

This section should connect the factorized DCAS design to representation disentanglement, domain-adversarial learning, and regularized latent geometry. The tone should remain pragmatic: the paper does not need to claim perfect semantic disentanglement, only task-useful factorization.

### Calibration, Diversity, And Reranking

This section should position the calibration layer as closer to responsible recommendation and reranking than to pure retrieval. The key idea is that cross-cultural recommendation quality cannot be reduced to relevance-only ranking.

### Human Feedback And Active Learning

This section should motivate PAL as a targeted latent-space repair mechanism. The emphasis is not large-scale annotation, but high-value expert intervention on uncertain boundary cases.

## Method Draft Skeleton

### Problem Definition

Let each track be represented by a frozen backbone embedding `e_i`. Given a user history `H_u` and a target culture `c*`, the task is to recommend tracks from `c*` that remain behaviorally relevant while improving cross-cultural discovery and calibrated exposure.

### Data

The paper should explicitly describe the V4 contract as the interface between heterogeneous backbone embeddings and a shared downstream pipeline. This subsection should mention unified metadata, interactions, audit outputs, and source-confound diagnostics.

### Model

Describe the downstream model as a factorized learner that maps each track into content, style, and affective-functional subspaces. Keep the notation simple and reserve most regularizer details for a compact objective paragraph.

### Training

Describe training as a curriculum:

1. representation stabilization
2. delayed constraint activation
3. delayed ranking activation

This is more defensible than presenting all losses as equally active from epoch one.

### Inference

Describe inference as a two-step process:

1. relevance estimation in the shared space
2. calibration-aware reranking for target affinity, minority exposure, source diversity, and stylistic novelty

### PAL

Describe PAL as four stages:

1. uncertainty scoring
2. expert annotation packet generation
3. pairwise constraint construction
4. warm-start retraining

## Pseudocode Plan

| Stage | Pseudocode block | Text explanation |
|---|---|---|
| Data contract | build V4 metadata, interactions, tracks, audits | explain that all backbones are normalized into the same downstream interface |
| Training | encode -> compute reconstruction/regularizers -> activate constraints -> activate ranking | highlight curriculum and delayed supervision |
| Inference | recommend with OT/shared-space relevance -> rerank with calibration weights | explain controllable fairness-quality trade-off |
| PAL | score uncertainty -> export tasks -> ingest annotations -> rebuild constraints | connect to practical expert workflow |

## Experiments Draft Skeleton

### Baseline Rationale

We compare against classic embedding retrieval baselines, learned ranking baselines, and internal ablations to answer three questions: whether gains come from the downstream stack rather than the backbone alone, whether calibration materially changes recommendation trade-offs, and whether the method remains competitive against stronger reranking families.

### Main Results Paragraph Template

On `V4 main`, the calibrated DCAS operating point improves minority exposure substantially relative to the uncalibrated OT variant while incurring only a modest serendipity reduction. This shows that the reranking layer is not merely cosmetic but acts as a controllable trade-off mechanism.

### Cross-Backbone Paragraph Template

The same qualitative trend also appears with Gemini embeddings, suggesting that the downstream calibration behavior is not tied to a single backbone. The strongest absolute metrics may differ by backbone, but the framework-level trade-off remains stable.

### Ablation Paragraph Template

Ablations show that removing constraints or domain shaping changes the balance between serendipity and minority exposure, indicating that the framework's value comes from the interaction of modules rather than a single isolated component.

### Hyperparameter Paragraph Template

Calibration sweep results form a smooth Pareto-style curve: as minority-oriented reranking weight increases, minority exposure rises monotonically, while serendipity and calibration move in predictable directions. This supports the claim that the operating point can be selected explicitly rather than tuned ad hoc.

### Failure Analysis Paragraph Template

The main remaining failure mode is not ranking collapse but data-level confounding. In particular, source predictability from culture remains high in `V4 main`, meaning that source bias is reduced but not eliminated.

## Discussion / Limitations Draft Skeleton

This section should explicitly acknowledge that:

- source confound remains substantial
- `routeA_small` is a sanity-check track, not a full-strength benchmark
- some labels such as `affect_label` are still incomplete
- the current implementation is optimized for medium-scale cultural catalogs rather than web-scale deployment

## Conclusion Draft Skeleton

The conclusion should emphasize reusable methodology rather than maximal empirical dominance. A safe closing sentence is:

The main value of the project lies in showing that cross-cultural recommendation can be treated as a modular, calibration-aware, and feedback-ready downstream problem on top of heterogeneous music embeddings.

Future work:

1. complete real PAL feedback rounds and quantify post-feedback gains
2. reduce source confound through broader source balancing and stronger audits
3. test the same pipeline on additional backbones or larger open cultural catalogs

