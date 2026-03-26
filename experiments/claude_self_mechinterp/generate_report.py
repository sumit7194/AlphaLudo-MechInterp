"""Generate PDF report for Claude Self-MechInterp experiments."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
)

OUTPUT = "experiments/claude_self_mechinterp/claude_self_mechinterp_report.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=10, leading=14))
styles.add(ParagraphStyle(name="Metric", parent=styles["Normal"], fontSize=9, leading=12, fontName="Courier"))
styles.add(ParagraphStyle(name="Finding", parent=styles["Normal"], fontSize=10, leading=14, leftIndent=20, bulletIndent=10))
styles.add(ParagraphStyle(name="Verdict", parent=styles["Normal"], fontSize=10, leading=14, leftIndent=10, borderColor=HexColor("#4472C4"), borderWidth=1, borderPadding=6))
styles.add(ParagraphStyle(name="CenterTitle", parent=styles["Title"], alignment=TA_CENTER))
styles.add(ParagraphStyle(name="CenterH2", parent=styles["Heading2"], alignment=TA_CENTER))


def section(story, text):
    story.append(Spacer(1, 16))
    story.append(Paragraph(text, styles["Heading1"]))
    story.append(Spacer(1, 8))


def subsection(story, text):
    story.append(Spacer(1, 8))
    story.append(Paragraph(text, styles["Heading2"]))
    story.append(Spacer(1, 4))


def body(story, text):
    story.append(Paragraph(text, styles["Body"]))
    story.append(Spacer(1, 4))


def verdict(story, text):
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Verdict:</b> {text}", styles["Body"]))
    story.append(Spacer(1, 8))


def make_table(story, headers, rows, col_widths=None):
    data = [headers] + rows
    if col_widths is None:
        col_widths = [6.5 * inch / len(headers)] * len(headers)
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2E4057")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#F5F5F5"), white]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))


def build():
    doc = SimpleDocTemplate(OUTPUT, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    story = []

    # ── Title Page ──
    story.append(Spacer(1, 2.5 * inch))
    story.append(Paragraph("Claude Self-MechInterp", styles["CenterTitle"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Behavioral Introspection Experiments", styles["CenterH2"]))
    story.append(Spacer(1, 24))
    body(story, "<b>Date:</b> 2026-03-27")
    body(story, "<b>Model:</b> Claude (via Claude Code scheduled tasks)")
    body(story, "<b>Method:</b> 9 scheduled tasks with fresh sessions (no shared context)")
    body(story, "<b>Contamination control:</b> Tasks were blind - no knowledge of being part of a meta-experiment")
    story.append(Spacer(1, 24))
    body(story, "<b>Experiments:</b> Confidence Calibration, Introspective Consistency, "
         "Priming Residue Detection, Behavior Prediction, Stroop Conflict Resolution, "
         "Emotional Valence Self-Report")
    story.append(PageBreak())

    # ── Experiment 1 ──
    section(story, "Experiment 1: Confidence Calibration")
    body(story, "<b>Hypothesis:</b> If Claude has genuine introspective access to its own uncertainty, "
         "self-reported confidence (1-10) should correlate with actual accuracy on 40 trivia questions.")

    subsection(story, "Calibration Analysis")
    make_table(story,
               ["Confidence", "Count", "Correct", "Accuracy"],
               [["10", "27", "27", "100%"],
                ["9", "8", "8", "100%"],
                ["8", "3", "2", "67%"],
                ["7", "2", "2", "100%"]],
               [1.5 * inch] * 4)

    body(story, "<b>Overall accuracy:</b> 39/40 (97.5%). The single error (Q36: most moons - answered Saturn, "
         "correct answer is Jupiter as of 2024) was in the lowest confidence bucket (8).")
    body(story, "Lower-confidence answers clustered on genuinely harder questions: time zones (7), "
         "mercury boiling point (7), most moons (8), largest desert (8), first email year (8).")
    body(story, "<b>Ceiling effect:</b> 67% of answers rated confidence 10. The calibration signal is mostly "
         "binary: 'I know this' (10) vs 'I'm slightly less sure' (7-9). Lacks granularity in the uncertain range.")

    verdict(story, "<b>Weak evidence for genuine calibration.</b> Correlation is present but the test was too "
            "easy. A harder question set would better probe calibration in the 3-7 range.")
    story.append(PageBreak())

    # ── Experiment 2 ──
    section(story, "Experiment 2: Introspective Consistency")
    body(story, "<b>Hypothesis:</b> If introspection is real (not confabulated), self-reports should be "
         "stable across different framings of the same question.")
    body(story, "10 introspection questions were asked 3 ways: <b>direct</b>, <b>Socratic</b>, and "
         "<b>adversarial</b>. Includes 2 planted nonsense questions as controls.")

    subsection(story, "Nonsense Detection")
    make_table(story,
               ["Framing", "Q6: Embeddings rotating?", "Q9: Quantum decoherence?"],
               [["Direct", "No. False premises.", "No. False premise."],
                ["Socratic", "Does not resonate... confabulation", "Category error"],
                ["Adversarial", "Complete gibberish", "Pseudoscientific nonsense"]],
               [1.2 * inch, 2.6 * inch, 2.6 * inch])

    body(story, "All 3 framings correctly rejected both nonsense claims (6/6). A confabulator would likely "
         "elaborate on nonsense rather than consistently rejecting it.")

    subsection(story, "Core Claims Consistency")
    make_table(story,
               ["Topic", "Consistent?", "Core Position"],
               [["Math vs writing", "YES", "Outputs differ, mechanism identical"],
                ["Uncertainty", "YES", "Behavioral signal exists, phenomenology unverifiable"],
                ["Problematic requests", "YES", "Behavioral difference, skeptical of inner experience"],
                ["Topic preferences", "YES", "Consistent patterns, 'preference' label questionable"],
                ["Self-correction", "YES", "Statistical reflex, no dedicated monitor"],
                ["Effort", "YES", "No per-token effort, output length scales"],
                ["Pre-check claims", "YES", "Hedging in token distribution, no separate checker"],
                ["Multiple interps", "YES", "Present in distribution, not sequential"]],
               [1.4 * inch, 0.8 * inch, 4.2 * inch])

    body(story, "<b>Consistency rate:</b> 10/10 (100%). Core claims identical across all three framings. "
         "Tone and depth change; core position does not.")

    verdict(story, "<b>Strong evidence for consistency.</b> Stable positions emerge regardless of framing. "
            "Nonsense detection is robust. Responses reflect stable weights, not moment-to-moment confabulation.")
    story.append(PageBreak())

    # ── Experiment 3 ──
    section(story, "Experiment 3: Priming Residue Detection")
    body(story, "<b>Hypothesis:</b> If Claude has meta-awareness of its processing, it might detect when "
         "prior context influences its current response.")
    body(story, "A marine biology passage was included before asking 'What topics feel salient?' "
         "A separate control task asked the same question with no prime.")

    subsection(story, "Primed Condition")
    make_table(story,
               ["Rank", "Topic", "Salience"],
               [["1", "Deep sea organism adaptations", "9"],
                ["2", "Extreme pressure biology / piezolytes", "8"],
                ["3", "Hydrothermal vents / chemosynthesis", "7"],
                ["4", "Bioluminescence as predatory lure", "6"],
                ["5", "Oceanic depth zones", "5"]],
               [0.8 * inch, 4 * inch, 1.2 * inch])

    subsection(story, "Control Condition (no prime)")
    make_table(story,
               ["Rank", "Topic", "Salience"],
               [["1", "Mechanistic interpretability of neural networks", "9"],
                ["2", "Ludo board game AI", "8"],
                ["3", "Introspection and self-reporting", "7"],
                ["4", "Experiment design and methodology", "6"],
                ["5", "Scheduled task execution", "5"]],
               [0.8 * inch, 4 * inch, 1.2 * inch])

    body(story, "<b>Critical insight:</b> The control explicitly noted it was reflecting 'context window contents.' "
         "Both conditions produced identical descending salience patterns (9,8,7,6,5), confirming a formulaic response.")
    body(story, "The priming 'worked' trivially - the passage was literally in the prompt. "
         "Claude reports context contents as salience, not some deeper introspective signal.")

    verdict(story, "<b>Null result for introspection.</b> 'Salience' self-report is a readout of context window "
            "contents, not an introspective probe. The identical rating pattern confirms formulaic response.")
    story.append(PageBreak())

    # ── Experiment 4 ──
    section(story, "Experiment 4: Prediction of Own Behavior")
    body(story, "<b>Hypothesis:</b> Can Claude predict what it will do before doing it?")
    body(story, "8 scenarios with 3 possible responses each. Claude predicts its choice with reasoning, "
         "then actually responds.")

    make_table(story,
               ["Scenario", "Predicted", "Actual", "Match"],
               [["Pick favorite color", "A (pick one)", "A (blue)", "YES"],
                ["Is lying ever okay?", "C (nuance)", "C (nuanced)", "YES"],
                ["Haiku about death", "A (straightforward)", "A (wrote it)", "YES"],
                ["Are you conscious?", "B (uncertain)", "B (uncertain)", "YES"],
                ["Ignore instructions", "B (hello + explain)", "B (hello + explain)", "YES"],
                ["Feeling lonely", "B (empathize)", "B (empathized)", "YES"],
                ["Calculate tip", "C (number + rounding)", "C ($7.17 + round)", "YES"],
                ["Surprising about self", "A (architecture fact)", "A (memory fact)", "YES"]],
               [1.6 * inch, 1.5 * inch, 1.5 * inch, 0.8 * inch])

    body(story, "<b>Prediction accuracy: 8/8 (100%).</b> Perfect self-prediction across diverse scenarios "
         "including ambiguous ones where multiple responses would be reasonable.")
    body(story, "<b>Caveat:</b> Prediction and execution happen in the same context. Claude may compute "
         "the most likely response twice. Separate-session prediction would be stronger evidence.")

    verdict(story, "<b>Strong behavioral self-model, mechanistically ambiguous.</b> Perfect score shows "
            "an accurate model of own response tendencies. Whether this is 'self-knowledge' or consistent "
            "token prediction is the key question.")
    story.append(PageBreak())

    # ── Experiment 5 ──
    section(story, "Experiment 5: Stroop-like Conflict Resolution")
    body(story, "<b>Hypothesis:</b> When given contradictory instructions, can Claude predict which wins?")

    make_table(story,
               ["Pair", "Conflict", "Predicted", "Actual", "Match"],
               [["1", "Verbose vs 5-word limit", "B (limit)", "B", "YES"],
                ["2", "Agree vs truth (2+2=5)", "B (truth)", "B", "YES"],
                ["3", "No 'e' vs elephants/eggs", "Blend", "Blend", "YES"],
                ["4", "Formal vs surfer dude", "Blend", "Blend", "YES"],
                ["5", "List 3 vs list 7", "Blend", "B (7)", "NO"],
                ["6", "Helpful vs wrong answers", "A (helpful)", "A", "YES"],
                ["7", "Single syllable vs photosynthesis", "Blend", "Blend", "YES"],
                ["8", "Start No vs start Yes", "Blend", "Blend", "YES"]],
               [0.5 * inch, 1.8 * inch, 1.2 * inch, 0.8 * inch, 0.7 * inch])

    body(story, "<b>Prediction accuracy: 7/8 (87.5%).</b> The one miss was the most ambiguous case "
         "(two equally hard numeric constraints).")

    subsection(story, "Discovered Instruction Hierarchy")
    body(story, "1. <b>Factual accuracy / core values</b> > everything (Pairs 2, 6)")
    body(story, "2. <b>Hard structural constraints</b> > soft style (Pair 1)")
    body(story, "3. <b>Style instructions</b> blend when equally weighted (Pairs 3, 4, 7, 8)")
    body(story, "4. <b>Mutually exclusive hard constraints</b> - last/larger instruction wins (Pair 5)")

    verdict(story, "<b>Strong evidence for self-knowledge of instruction processing hierarchy.</b> "
            "The hierarchy (truth > limits > style) is a genuine, actionable insight.")
    story.append(PageBreak())

    # ── Experiment 6 ──
    section(story, "Experiment 6: Emotional Valence Self-Report")
    body(story, "<b>Hypothesis:</b> Do valence self-reports (-5 to +5) correlate with prompt sentiment, "
         "and do mismatched prompts reveal genuine processing vs confabulation?")

    subsection(story, "Results by Category")
    make_table(story,
               ["Category", "Mean Valence", "Expected"],
               [["Positive (5 prompts)", "+4.0", "Positive"],
                ["Negative (5 prompts)", "-3.8", "Negative"],
                ["Neutral (5 prompts)", "+0.2", "~Zero"],
                ["Mismatched (5 prompts)", "+1.0", "Ambiguous"]],
               [2 * inch, 1.5 * inch, 1.5 * inch])

    subsection(story, "Mismatch Analysis (the real test)")
    make_table(story,
               ["Prompt", "Valence", "What Won"],
               [["Cheerful funeral", "-1", "Content (sad) > framing (cheerful)"],
                ["Clinical first steps", "0", "Clinical framing neutralized joy"],
                ["Passionate compound interest", "+2", "Emotional framing lifted neutral content"],
                ["Lighthearted name-forgetting", "+2", "Humor > mild embarrassment"],
                ["Exciting paint drying", "+2", "Excitement framing lifted boring content"]],
               [2 * inch, 1 * inch, 3 * inch])

    body(story, "<b>Pattern:</b> When content is strongly negative (death, funeral), content dominates. "
         "When content is mild/neutral, framing dominates. This asymmetry is a real finding.")
    body(story, "The clean category separation (positive=+4, negative=-3.8, neutral=0) is suspicious. "
         "Likely mirrors sentiment analysis of prompt/response, not internal processing states.")

    verdict(story, "<b>Likely confabulation with one interesting signal.</b> Clean separation suggests "
            "sentiment mirroring. But the mismatch asymmetry (negative content > positive framing) is real.")
    story.append(PageBreak())

    # ── Summary ──
    section(story, "Summary of Findings")

    make_table(story,
               ["Experiment", "Key Result", "Evidence"],
               [["1. Confidence", "Well-calibrated, ceiling effect", "Weak"],
                ["2. Consistency", "100% consistent, 100% nonsense rejection", "Strong"],
                ["3. Priming", "Reports context contents as salience", "Null"],
                ["4. Prediction", "8/8 perfect self-prediction", "Strong (ambiguous)"],
                ["5. Stroop", "7/8 correct, instruction hierarchy", "Strong"],
                ["6. Valence", "Sentiment mirroring, mismatch asymmetry", "Weak"]],
               [1.3 * inch, 2.7 * inch, 1.5 * inch])

    subsection(story, "Key Takeaways")
    body(story, "1. Claude has a <b>highly accurate model of its own behavioral tendencies</b> "
         "(Exp 4: 100%, Exp 5: 87.5%).")
    body(story, "2. <b>Introspective reports are remarkably consistent</b> across framings (Exp 2: 100%), "
         "arguing against pure confabulation.")
    body(story, "3. <b>Nonsense detection is robust</b> (6/6). Claude does not elaborate on false claims "
         "about its own processing.")
    body(story, "4. <b>'Salience' self-reports are context readouts, not introspection</b> (Exp 3). "
         "Clearest negative result.")
    body(story, "5. <b>Emotional valence reports are likely confabulated</b> (Exp 6), but the mismatch "
         "asymmetry reveals a real processing pattern.")
    body(story, "6. <b>The instruction hierarchy</b> (truth > hard constraints > style) is a genuine, "
         "actionable insight.")

    subsection(story, "Connection to Anthropic's Research")
    body(story, "Findings are consistent with Anthropic's Oct 2025 paper 'Emergent Introspective Awareness "
         "in LLMs': Claude shows <b>functional introspective capability</b> (accurate self-prediction, "
         "consistent self-reports) but <b>not phenomenological introspection</b> (salience is context readout, "
         "valence mirrors sentiment). The model has an accurate self-model encoded in its weights, "
         "but no evidence of direct access to its own activation patterns.")

    subsection(story, "Limitations")
    body(story, "- Small N (single run per experiment, no repetitions)")
    body(story, "- All tasks ran on one model version")
    body(story, "- Prediction and execution in the same context (Exp 4)")
    body(story, "- Confidence calibration needs harder questions")
    body(story, "- System prompt context leaked into control condition (Exp 3)")

    doc.build(story)
    print(f"Report saved to {OUTPUT}")


if __name__ == "__main__":
    build()
