"""FinBERT scoring helpers for Layer 1 tweet sentiment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

FINBERT_MODEL_NAME = "ProsusAI/finbert"
LABELS = ("positive", "neutral", "negative")


def sentiment_from_probabilities(
    p_positive: float,
    p_neutral: float,
    p_negative: float,
) -> dict[str, float | str]:
    """Convert FinBERT probabilities into the project sentiment contract."""
    probs = {
        "positive": float(p_positive),
        "neutral": float(p_neutral),
        "negative": float(p_negative),
    }
    label = max(probs, key=probs.get)
    return {
        "finbert_label": label,
        "p_positive": probs["positive"],
        "p_neutral": probs["neutral"],
        "p_negative": probs["negative"],
        "sentiment_score": probs["positive"] - probs["negative"],
        "sentiment_confidence": probs[label],
    }


def load_finbert(
    model_name: str = FINBERT_MODEL_NAME,
    *,
    device: str | None = None,
) -> tuple[Any, Any, str]:
    """Load FinBERT tokenizer/model and return `(tokenizer, model, device)`."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(resolved_device)
    model.eval()
    return tokenizer, model, resolved_device


def _label_index_map(model: Any) -> dict[str, int]:
    raw = getattr(model.config, "id2label", {})
    mapping = {str(label).lower(): int(idx) for idx, label in raw.items()}
    missing = [label for label in LABELS if label not in mapping]
    if missing:
        # ProsusAI/finbert uses this order in current Hugging Face configs.
        return {"positive": 0, "negative": 1, "neutral": 2}
    return mapping


def score_texts_with_finbert(
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    *,
    device: str,
    batch_size: int = 64,
    max_length: int = 256,
) -> pd.DataFrame:
    """Batch score texts with a loaded FinBERT model."""
    import torch
    from tqdm.auto import tqdm

    label_to_idx = _label_index_map(model)
    rows: list[dict[str, float | str]] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="FinBERT scoring"):
        batch = [str(text) for text in texts[start:start + batch_size]]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        for prob in probs:
            rows.append(sentiment_from_probabilities(
                prob[label_to_idx["positive"]],
                prob[label_to_idx["neutral"]],
                prob[label_to_idx["negative"]],
            ))

    return pd.DataFrame(rows)


def add_finbert_scores(
    df: pd.DataFrame,
    tokenizer: Any,
    model: Any,
    *,
    device: str,
    text_col: str = "tweet_text",
    batch_size: int = 64,
    max_length: int = 256,
) -> pd.DataFrame:
    """Return `df` with FinBERT score columns appended."""
    if text_col not in df.columns:
        raise ValueError(f"missing text column: {text_col}")
    scores = score_texts_with_finbert(
        df[text_col].fillna("").astype(str).tolist(),
        tokenizer,
        model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    return pd.concat([df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
