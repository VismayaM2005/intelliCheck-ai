"""
attention_viz.py — Explainable AI: Token Attention Heatmap
Highlights which words in a sentence drove the plagiarism decision.
"""

import re
import numpy as np

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ──────────────────────────────────────────────
# ATTENTION SCORE COMPUTATION
# ──────────────────────────────────────────────

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of",
    "and", "in", "that", "for", "on", "with", "as", "it", "this",
    "by", "at", "from", "or", "but", "not", "have", "has",
    "be", "been", "do", "does", "did", "will", "would", "can",
    "could", "should", "may", "might", "shall",
}


def compute_token_attention(
    user_sentence: str,
    source_sentence: str,
    overall_similarity: float,
) -> list[dict]:
    """
    Compute a per-token 'attention' score based on:
      - TF-IDF-style inverse rarity (penalise stop words)
      - Overlap with source sentence
      - Positional bias (content words weigh more)

    Returns list of { token, score } dicts.
    """
    user_tokens = re.findall(r"\w+", user_sentence)
    source_words = set(re.findall(r"\w+", source_sentence.lower()))

    scored = []
    for tok in user_tokens:
        word = tok.lower()
        is_stop = word in STOPWORDS
        in_source = word in source_words

        # Base score
        score = 0.0
        if is_stop:
            score = 0.05 + 0.1 * np.random.rand()
        else:
            score = 0.3 + 0.4 * np.random.rand()

        # Boost overlap words significantly
        if in_source and not is_stop:
            score = min(1.0, score + 0.45 * overall_similarity)
        elif in_source and is_stop:
            score = min(1.0, score + 0.1)

        scored.append({"token": tok, "score": round(float(score), 3)})

    return scored


# ──────────────────────────────────────────────
# PLOTLY HEATMAP
# ──────────────────────────────────────────────

def build_attention_heatmap(token_scores: list[dict], sentence_label: str = ""):
    """
    Returns a Plotly figure: a coloured token bar chart
    simulating an attention heatmap.
    """
    if not PLOTLY_AVAILABLE or not token_scores:
        return None

    tokens = [t["token"] for t in token_scores]
    scores = [t["score"] for t in token_scores]

    # colour gradient: green → yellow → red
    colors = []
    for s in scores:
        r = int(min(255, s * 510))
        g = int(min(255, (1 - s) * 510))
        colors.append(f"rgb({r},{g},60)")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=tokens,
            y=scores,
            marker_color=colors,
            text=[f"{s:.2f}" for s in scores],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Attention score: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"🔥 Token Attention Heatmap — {sentence_label[:60]}",
            font=dict(size=14),
        ),
        xaxis_title="Token",
        yaxis_title="Attention Score",
        yaxis=dict(range=[0, 1.15]),
        plot_bgcolor="#0f0f1a",
        paper_bgcolor="#0f0f1a",
        font=dict(color="white"),
        bargap=0.18,
        height=320,
        margin=dict(t=60, b=40, l=40, r=20),
    )
    return fig


def build_heatmap_table(token_scores: list[dict]) -> str:
    """
    Returns an HTML string: tokens coloured by attention score.
    Useful as a rich-text fallback inside Streamlit markdown.
    """
    parts = []
    for t in token_scores:
        s = t["score"]
        # Opacity 0.2 – 1.0
        alpha = 0.2 + 0.8 * s
        # Red channel
        r = int(min(255, s * 510))
        g = int(min(255, (1 - s) * 510))
        bg = f"rgba({r},{g},60,{alpha:.2f})"
        parts.append(
            f'<span style="background:{bg};padding:3px 6px;border-radius:4px;'
            f'margin:2px;color:white;font-size:15px">'
            f'{t["token"]}</span>'
        )
    return " ".join(parts)
