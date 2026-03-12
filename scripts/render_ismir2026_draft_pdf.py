from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs"
OUT_PDF = OUT_DIR / "ismir2026_draft_placeholder.pdf"


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontName="Times-Bold",
            fontSize=16,
            leading=18,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AuthorCenter",
            parent=styles["Normal"],
            alignment=TA_CENTER,
            fontName="Times-Roman",
            fontSize=10,
            leading=12,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJustify",
            parent=styles["BodyText"],
            alignment=TA_JUSTIFY,
            fontName="Times-Roman",
            fontSize=9.2,
            leading=12,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHead",
            parent=styles["Heading2"],
            fontName="Times-Bold",
            fontSize=11.5,
            leading=13,
            textColor=colors.black,
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallNote",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=8.2,
            leading=10,
            alignment=TA_JUSTIFY,
            textColor=colors.darkred,
        )
    )
    return styles


def add_paragraph(story, text, style):
    story.append(Paragraph(text, style))
    story.append(Spacer(1, 0.06 * inch))


def build_table(data, col_widths):
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9edf5")),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fb")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    styles = build_styles()
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=LETTER,
        leftMargin=0.72 * inch,
        rightMargin=0.72 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.72 * inch,
        title="ISMIR 2026 draft placeholder",
        author="Anonymous ISMIR 2026 Submission",
    )

    story = []
    story.append(
        Paragraph(
            "Disentangled Cross-Cultural Music Recommendation with Optimal Transport Alignment and Participatory Feedback",
            styles["TitleCenter"],
        )
    )
    story.append(Paragraph("Anonymous ISMIR 2026 Submission", styles["AuthorCenter"]))

    notice = Table(
        [[Paragraph(
            "<b>Internal drafting note.</b> This PDF is a rendered planning preview. "
            "All quantitative values marked with dagger symbols are synthetic placeholders "
            "introduced only to stabilize the manuscript structure before the final experiments are complete.",
            styles["SmallNote"],
        )]],
        colWidths=[6.6 * inch],
    )
    notice.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.8, colors.red),
                ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(notice)
    story.append(Spacer(1, 0.12 * inch))

    add_paragraph(
        story,
        "<b>Abstract.</b> Cross-cultural music recommendation remains difficult even with strong audio foundation models because stylistic and cultural variation is often entangled with affective or functional similarity. We present a proof-of-concept framework that combines disentangled downstream representation learning, domain-adversarial alignment, and optimal transport ranking on top of foundation embeddings. A pilot participatory active learning loop converts small amounts of expert feedback into pairwise constraints for retraining. This internal draft uses synthetic placeholder numbers in the quantitative sections, but the structure matches the intended ISMIR paper.",
        styles["BodyJustify"],
    )

    sections = [
        (
            "1. Introduction",
            [
                "Foundation audio models improve generic representation learning, but stronger embeddings alone do not solve cross-cultural recommendation. Users often want music that is unfamiliar in style yet meaningful in function or affect.",
                "We argue that this failure is partly representational: embedding-only retrieval often conflates stylistic unfamiliarity with irrelevance and therefore collapses to culturally dominant or overly safe targets.",
                "Our proposed response is to factor downstream representations into content, style, and relatively culture-agnostic affective-functional variables, then perform recommendation in the shared subspace rather than directly in the raw embedding space.",
            ],
        ),
        (
            "2. Method",
            [
                "Each track embedding is mapped into three latent variables: zc for content structure, zs for style or cultural variation, and za for relatively shared affective-functional relevance.",
                "A culture discriminator attached to za is trained adversarially so that the recommendation subspace becomes less culture-discriminative while still retaining task-relevant information.",
                "Recommendation is performed with entropic optimal transport between the user's preference distribution and the candidate distribution in the target culture.",
                "A PAL loop selects uncertain items for expert review and turns those judgments into pairwise constraints for retraining.",
            ],
        ),
        (
            "3. Experimental Setup",
            [
                "The current pipeline uses a four-domain public-audio dataset construction process spanning West, India, Turkey, and China. The internal dataset version contains 1,600 tracks with balanced culture coverage, fixed splits, and weak interaction logs.",
                "Baselines include CultureMERT cosine retrieval, CultureMERT with a shallow ranking head, DCAS without OT, and DCAS without domain-adversarial alignment.",
                "Metrics include nDCG@10, Recall@10, serendipity, cultural calibration KL, and disentanglement proxies.",
            ],
        ),
    ]

    for title, paragraphs in sections:
        story.append(Paragraph(title, styles["SectionHead"]))
        for p in paragraphs:
            add_paragraph(story, p, styles["BodyJustify"])

    story.append(Paragraph("4. Main Results", styles["SectionHead"]))
    add_paragraph(
        story,
        "The primary intended claim is that embedding-only retrieval is insufficient for cross-cultural recommendation. The full model should improve serendipity and reduce calibration error while preserving ranking quality.",
        styles["BodyJustify"],
    )

    main_table = build_table(
        [
            ["Model", "nDCG@10", "Recall@10", "Serendipity", "Calibration KL"],
            ["CultureMERT cosine", "0.401†", "0.512†", "0.171†", "0.91†"],
            ["CultureMERT + MLP", "0.417†", "0.528†", "0.180†", "0.87†"],
            ["DCAS w/o OT", "0.425†", "0.535†", "0.218†", "0.82†"],
            ["DCAS w/o domain adv.", "0.432†", "0.543†", "0.231†", "0.80†"],
            ["DCAS full", "0.439†", "0.547†", "0.252†", "0.69†"],
        ],
        [2.4 * inch, 0.75 * inch, 0.78 * inch, 0.95 * inch, 0.92 * inch],
    )
    story.append(main_table)
    story.append(Spacer(1, 0.12 * inch))
    add_paragraph(
        story,
        "All values above are synthetic placeholders for internal drafting only. They encode the target narrative pattern rather than actual final results.",
        styles["SmallNote"],
    )

    story.append(Paragraph("5. Ablation and PAL Pilot", styles["SectionHead"]))
    ablation_table = build_table(
        [
            ["Variant", "Serendipity", "Calibration KL"],
            ["Full model", "0.252†", "0.69†"],
            ["No OT", "0.218†", "0.82†"],
            ["No domain adv.", "0.231†", "0.80†"],
            ["No constraints", "0.240†", "0.74†"],
            ["After real PAL pilot", "0.268†", "0.62†"],
        ],
        [2.9 * inch, 1.2 * inch, 1.2 * inch],
    )
    story.append(ablation_table)
    story.append(Spacer(1, 0.12 * inch))
    add_paragraph(
        story,
        "The intended interpretation is modest: the structured latent representation improves the geometry of the recommendation space, while small amounts of expert feedback can repair especially ambiguous cross-cultural boundaries.",
        styles["BodyJustify"],
    )

    story.append(PageBreak())
    story.append(Paragraph("6. Discussion and Ethics", styles["SectionHead"]))
    discussion = [
        "This draft should not be read as a claim that disentanglement fully captures cultural meaning. Instead, the model provides a task-driven approximation for separating stylistic variation from affective-functional similarity.",
        "The contribution is therefore not a new foundation model, but a structured downstream alignment layer on top of foundation embeddings for a culturally sensitive recommendation problem.",
        "Any future publication should disclose dataset composition, annotation provenance, cultural coverage limits, and the distinction between simulated and real feedback.",
    ]
    for p in discussion:
        add_paragraph(story, p, styles["BodyJustify"])

    story.append(Paragraph("7. Conclusion", styles["SectionHead"]))
    add_paragraph(
        story,
        "This manuscript draft frames the project as an ISMIR-style proof-of-concept paper: a cross-cultural recommendation system built on foundation embeddings, disentangled downstream alignment, optimal transport, and a pilot participatory feedback loop. The current PDF is a planning artifact, while the accompanying LaTeX source is intended for later replacement with final experimental values.",
        styles["BodyJustify"],
    )

    story.append(Paragraph("References included in the LaTeX source", styles["SectionHead"]))
    refs = [
        "Li et al. (2023). MERT: Acoustic Music Understanding Model with Large-Scale Self-Supervised Training.",
        "Kanatas et al. (2025). CultureMERT: Continual Pre-Training for Cross-Cultural Music Representation Learning.",
        "Lee et al. (2025). GlobalMood: A Cross-Cultural Benchmark for Music Emotion Recognition.",
        "Ganin et al. (2016). Domain-Adversarial Training of Neural Networks.",
        "Cuturi (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport.",
        "Settles (2009). Active Learning Literature Survey.",
        "Zhang et al. (2012). Auralist: Introducing Serendipity into Music Recommendation.",
    ]
    for ref in refs:
        add_paragraph(story, ref, styles["BodyJustify"])

    doc.build(story)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
