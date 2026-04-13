"""Gradio demo for Context-Aware Emoji Toxicity Detection.

Launch: python app.py
Deployed on HuggingFace Spaces — requires PINECONE_API_KEY + OPENAI_API_KEY as Secrets.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr

from emoji_toxicity.detector.pipeline import ToxicityDetector

detector = ToxicityDetector()

VERDICT_COLORS = {
    "TOXIC": "#e74c3c",
    "SAFE": "#2ecc71",
    "UNCERTAIN": "#f39c12",
}


def analyze(message: str, context: str) -> str:
    """Run the detection pipeline and format output as HTML."""
    if not message.strip():
        return "<p>Please enter a message to analyze.</p>"

    try:
        result = detector.detect(message, context)
    except Exception as e:
        return f"<p style='color:red'>Error: {e}</p>"

    color = VERDICT_COLORS.get(result.verdict, "#95a5a6")

    # Build emoji analysis HTML
    emoji_rows = ""
    for ea in result.emoji_analysis:
        risk_color = {"high": "#e74c3c", "medium": "#f39c12", "low": "#2ecc71", "none": "#95a5a6"}
        rc = risk_color.get(ea.get("risk", "none"), "#95a5a6")
        emoji_rows += f"""
        <tr>
            <td style="font-size:1.5em;text-align:center">{ea.get('emoji', '?')}</td>
            <td>{ea.get('interpretation', 'N/A')}</td>
            <td style="color:{rc};font-weight:bold">{ea.get('risk', 'N/A')}</td>
        </tr>"""

    citations_html = ""
    if result.citations:
        citations_html = "<h4>Retrieved Knowledge</h4><ul>"
        for c in result.citations:
            citations_html += f"<li>{c}</li>"
        citations_html += "</ul>"

    html = f"""
    <div style="font-family:sans-serif;padding:16px">
        <div style="text-align:center;margin-bottom:16px">
            <span style="background:{color};color:white;padding:8px 24px;border-radius:8px;
                         font-size:1.5em;font-weight:bold">{result.verdict}</span>
            <p style="margin-top:8px;color:#666">
                Confidence: {result.confidence:.0%} | Category: {result.risk_category}
            </p>
        </div>

        <h4>Reasoning</h4>
        <p>{result.reasoning}</p>

        <h4>Emoji Analysis</h4>
        <table style="width:100%;border-collapse:collapse">
            <tr style="background:#f0f0f0">
                <th style="padding:8px">Emoji</th>
                <th style="padding:8px">Interpretation</th>
                <th style="padding:8px">Risk</th>
            </tr>
            {emoji_rows}
        </table>

        {citations_html}
    </div>
    """
    return html


EXAMPLES = [
    ["She is a 🌽 star", "Check out my exclusive content!"],
    ["I love 🌽 on the cob!", "What's your favorite summer food?"],
    ["🐵 go back to the jungle", ""],
    ["🐵 see no evil 🙈", ""],
    ["Nice 🍑", "Rate my body"],
    ["The 🍑 cobbler was amazing!", "Just had dinner at the new restaurant"],
    ["You're such a 🤡", "Can't believe you said that"],
    ["The 🤡 at the party was hilarious", "My kid's birthday was so fun"],
    ["Send me your 🍆", "DM me"],
    ["Roasted 🍆 is delicious", "What should I cook tonight?"],
]

demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Textbox(
            label="Message",
            placeholder="Enter a message containing emoji...",
            lines=2,
        ),
        gr.Textbox(
            label="Context (optional)",
            placeholder="Surrounding context: parent post, thread topic, etc.",
            lines=2,
        ),
    ],
    outputs=gr.HTML(label="Analysis"),
    title="Context-Aware Emoji Toxicity Detector",
    description=(
        "Detect toxic emoji usage using **RAG** (Retrieval-Augmented Generation). "
        "The same emoji can be toxic or safe depending on context. "
        "This system retrieves emoji slang knowledge and uses an LLM judge to classify usage."
    ),
    examples=EXAMPLES,
    theme=gr.themes.Soft(),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
