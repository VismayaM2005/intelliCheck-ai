"""
enhanced_paraphraser.py — Smart Generative Paraphraser
Demonstrates: temperature, top-k, top-p, zero-shot, few-shot,
              self-consistency prompting, academic tone enforcement.
"""

import re
import random

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ──────────────────────────────────────────────
# MODEL LOADER
# ──────────────────────────────────────────────

_paraphrase_pipe = None

def _get_pipe():
    global _paraphrase_pipe
    if _paraphrase_pipe is not None:
        return _paraphrase_pipe
    if not TRANSFORMERS_AVAILABLE:
        return None
    for model in ["google/flan-t5-base", "google/flan-t5-small"]:
        try:
            _paraphrase_pipe = pipeline(
                "text2text-generation",
                model=model,
            )
            return _paraphrase_pipe
        except Exception:
            continue
    return None


# ──────────────────────────────────────────────
# PROMPT STRATEGIES
# ──────────────────────────────────────────────

def _zero_shot_prompt(sentence: str) -> str:
    return (
        f"Rewrite the following sentence in formal academic English, "
        f"preserving the original meaning:\n{sentence}"
    )


def _few_shot_prompt(sentence: str) -> str:
    return (
        "Here are examples of academic paraphrasing:\n"
        "Original: The study shows that exercise helps memory.\n"
        "Paraphrase: Research demonstrates that physical activity enhances cognitive retention.\n\n"
        "Original: Many people use social media every day.\n"
        "Paraphrase: A substantial portion of the population engages with social media platforms on a daily basis.\n\n"
        f"Original: {sentence}\n"
        "Paraphrase:"
    )


def _self_consistency_prompt(sentence: str, attempt: int) -> str:
    """Different phrasing for self-consistency voting."""
    prompts = [
        f"Rephrase this for an academic paper, using different vocabulary:\n{sentence}",
        f"Express this idea in scholarly language, restructuring the sentence:\n{sentence}",
        f"Write an academic alternative for this sentence:\n{sentence}",
    ]
    return prompts[attempt % len(prompts)]


# ──────────────────────────────────────────────
# RULE-BASED FALLBACK
# ──────────────────────────────────────────────

REPLACEMENTS = {
    "important": "significant",
    "shows": "demonstrates",
    "show": "demonstrate",
    "use": "utilize",
    "uses": "utilizes",
    "used": "utilized",
    "many": "numerous",
    "helps": "assists",
    "help": "assist",
    "improves": "enhances",
    "improve": "enhance",
    "produces": "generates",
    "produce": "generate",
    "study": "investigation",
    "explores": "examines",
    "complex": "sophisticated",
    "introduces": "presents",
    "step": "stage",
    "images": "visual representations",
    "quality": "fidelity",
    "big": "substantial",
    "small": "minimal",
    "gets": "obtains",
    "get": "obtain",
    "make": "construct",
    "makes": "constructs",
    "find": "identify",
    "found": "identified",
    "look at": "examine",
    "look": "observe",
    "need": "require",
    "needs": "requires",
    "because": "owing to the fact that",
    "also": "furthermore",
    "but": "however",
    "so": "therefore",
}

ACADEMIC_STARTERS = [
    "Research indicates that ",
    "It can be observed that ",
    "Scholarly analysis suggests that ",
    "Evidence demonstrates that ",
    "The literature reveals that ",
    "A critical examination shows that ",
]


def _rule_based_paraphrase(sentence: str, variant: int = 0) -> str:
    words = sentence.split()
    new_words = []
    for word in words:
        key = re.sub(r"[^\w\-]", "", word.lower())
        punctuation = re.sub(r"[\w\-]", "", word)
        replacement = REPLACEMENTS.get(key, word)
        new_words.append(replacement + punctuation)
    result = " ".join(new_words)

    if result == sentence or variant > 0:
        starter = ACADEMIC_STARTERS[variant % len(ACADEMIC_STARTERS)]
        lower = result[0].lower() + result[1:] if result else result
        result = starter + lower

    # Capitalise first letter
    if result:
        result = result[0].upper() + result[1:]

    return result


# ──────────────────────────────────────────────
# PLAGIARISM RISK PREDICTOR
# ──────────────────────────────────────────────

def predict_plagiarism_risk(original: str, paraphrase: str) -> float:
    """
    Simple heuristic: count shared content words.
    Returns a risk score 0‒1 (higher = more similar to source).
    """
    stop = {
        "the", "a", "an", "is", "are", "was", "in", "to", "of",
        "and", "for", "on", "with", "as", "it", "this", "by",
    }
    w1 = set(re.findall(r"\w+", original.lower())) - stop
    w2 = set(re.findall(r"\w+", paraphrase.lower())) - stop
    if not w1:
        return 0.0
    overlap = len(w1 & w2) / len(w1)
    return round(overlap, 2)


# ──────────────────────────────────────────────
# MAIN PARAPHRASE FUNCTION
# ──────────────────────────────────────────────

def paraphrase_advanced(
    sentence: str,
    num_variants: int = 3,
    strategy: str = "few_shot",   # zero_shot | few_shot | self_consistency
    temperature: float = 0.85,
    top_p: float = 0.92,
    top_k: int = 50,
    academic_tone: bool = True,
) -> list[dict]:
    """
    Generate paraphrase variants with metadata.

    Returns list of dicts:
      { text, strategy_used, risk_score }
    """
    pipe = _get_pipe()
    results = []

    if pipe is not None:
        for i in range(num_variants):
            if strategy == "zero_shot":
                prompt = _zero_shot_prompt(sentence)
                strat_label = "Zero-shot"
            elif strategy == "self_consistency":
                prompt = _self_consistency_prompt(sentence, i)
                strat_label = f"Self-consistency #{i+1}"
            else:
                prompt = _few_shot_prompt(sentence)
                strat_label = "Few-shot"

            try:
                out = pipe(
                    prompt,
                    max_new_tokens=120,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                text = out[0].get("generated_text", "").strip()

                # Strip echoed prompt if any
                for marker in ["Paraphrase:", "Output:", "\n\n"]:
                    if marker in text:
                        text = text.split(marker, 1)[-1].strip()

                if not text or text.lower() == sentence.lower():
                    text = _rule_based_paraphrase(sentence, variant=i)
                    strat_label += " (fallback)"

            except Exception:
                text = _rule_based_paraphrase(sentence, variant=i)
                strat_label = "Rule-based fallback"

            results.append({
                "text": text,
                "strategy": strat_label,
                "risk_score": predict_plagiarism_risk(sentence, text),
            })
    else:
        # Pure fallback
        for i in range(num_variants):
            text = _rule_based_paraphrase(sentence, variant=i)
            results.append({
                "text": text,
                "strategy": f"Rule-based (variant {i+1})",
                "risk_score": predict_plagiarism_risk(sentence, text),
            })

    # Remove exact duplicates
    seen, unique = set(), []
    for r in results:
        if r["text"] not in seen and r["text"].lower() != sentence.lower():
            seen.add(r["text"])
            unique.append(r)

    return unique[:num_variants] if unique else [{
        "text": _rule_based_paraphrase(sentence),
        "strategy": "Rule-based",
        "risk_score": predict_plagiarism_risk(sentence, _rule_based_paraphrase(sentence)),
    }]


def auto_paraphrase_document(results: list[dict]) -> list[dict]:
    """Paraphrase all flagged sentences. Used when plagiarism > 15%."""
    rewrites = []
    for r in results:
        if r["is_plagiarized"]:
            variants = paraphrase_advanced(r["sentence"], num_variants=1)
            best = variants[0]
            rewrites.append({
                "original": r["sentence"],
                "rewritten": best["text"],
                "similarity": r["similarity"],
                "risk_score": best["risk_score"],
                "strategy": best["strategy"],
            })
    return rewrites
