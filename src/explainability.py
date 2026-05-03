"""
Model explainability for PhishGuard AI.

Provides two explanation strategies:

1. **Rule-based** — directly exposes the triggered rule list from
   :mod:`src.rule_based`, formatted for human consumption and UI rendering.

2. **ML model (LIME)** — uses LIME (Local Interpretable Model-Agnostic
   Explanations) to identify the top token features that pushed the
   TF-IDF + Logistic Regression model toward its prediction.

3. **Token highlighting** — returns a list of (token, direction) pairs
   that can be used by the frontend to colour-code phishing indicators
   inside the email body.

Dependencies
------------
    pip install lime
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-import LIME so the module works even when lime is not installed
# ---------------------------------------------------------------------------

def _require_lime() -> Any:
    try:
        from lime.lime_text import LimeTextExplainer
        return LimeTextExplainer
    except ImportError as exc:
        raise ImportError(
            "LIME is required for ML explainability.\n"
            "Install with:  pip install lime"
        ) from exc


# ---------------------------------------------------------------------------
# Rule-based explanation
# ---------------------------------------------------------------------------

def explain_rule_based(rule_result: dict[str, Any]) -> dict[str, Any]:
    """
    Format the rule-based classifier output into a structured explanation.

    Parameters
    ----------
    rule_result:
        The dict returned by :func:`src.rule_based.analyze`.

    Returns
    -------
    dict with keys:
        - ``summary``      : human-readable one-liner
        - ``verdict``      : "phishing" | "safe"
        - ``confidence``   : float 0-1
        - ``rules_fired``  : list of {rule, detail, weight, severity}
        - ``rule_count``   : int
        - ``top_rule``     : the highest-weight triggered rule or None
    """
    triggered = rule_result.get("triggered_rules", [])
    label = rule_result.get("label", "safe")
    confidence = rule_result.get("confidence", 0.0)

    # Assign human-readable severity tiers
    def _severity(weight: float) -> str:
        if weight >= 0.40:
            return "critical"
        if weight >= 0.25:
            return "high"
        if weight >= 0.10:
            return "medium"
        return "low"

    enriched: list[dict[str, Any]] = []
    for rule in triggered:
        enriched.append({
            "rule": rule.get("rule", "unknown"),
            "detail": rule.get("detail", ""),
            "weight": round(rule.get("weight", 0.0), 3),
            "severity": _severity(rule.get("weight", 0.0)),
        })

    # Sort by weight descending
    enriched.sort(key=lambda r: r["weight"], reverse=True)
    top_rule = enriched[0] if enriched else None

    if label == "phishing":
        summary = (
            f"Flagged as PHISHING with {confidence * 100:.1f}% confidence. "
            f"{len(enriched)} rule(s) triggered."
        )
    else:
        summary = (
            f"Classified as SAFE with {(1 - confidence) * 100:.1f}% confidence. "
            f"{'No rules triggered.' if not enriched else f'{len(enriched)} low-weight rule(s) did not exceed threshold.'}"
        )

    return {
        "summary": summary,
        "verdict": label,
        "confidence": confidence,
        "rules_fired": enriched,
        "rule_count": len(enriched),
        "top_rule": top_rule,
    }


# ---------------------------------------------------------------------------
# LIME explanation for TF-IDF + LR model
# ---------------------------------------------------------------------------

def explain_ml_model(
    subject: str,
    body: str,
    ml_model_instance: Any,
    num_features: int = 15,
    num_samples: int = 1000,
) -> dict[str, Any]:
    """
    Generate a LIME explanation for the TF-IDF + LR model prediction.
    Falls back to TF-IDF coefficients if LIME is unavailable or fails.

    Parameters
    ----------
    subject:
        Email subject.
    body:
        Email body.
    ml_model_instance:
        An :class:`src.ml_model.MLModel` instance (already loaded).
    num_features:
        Number of top features to include in the explanation.
    num_samples:
        Number of perturbed samples LIME generates (higher = more accurate,
        but slower — 1000 is a good balance for interactive use).

    Returns
    -------
    dict with keys:
        - ``features``         : list of {token, weight, direction}
        - ``summary``          : human-readable one-liner
        - ``top_phishing_tokens``: tokens driving toward phishing
        - ``top_safe_tokens``  : tokens driving toward safe
        - ``prediction``       : replicated label from the model
        - ``confidence``       : replicated confidence
    """
    try:
        LimeTextExplainer = _require_lime()
    except ImportError:
        log.debug("LIME not available, using TF-IDF fallback")
        return _fallback_ml_explanation(subject, body, ml_model_instance)

    from src.ml_model import clean_text  # local import to avoid circular deps

    text = clean_text(f"{subject} {body}")

    def predict_fn(texts: list[str]) -> Any:
        """Predict function wrapper for LIME."""
        import numpy as np

        if not ml_model_instance.is_loaded:
            ml_model_instance._load()

        X_vec = ml_model_instance._vectorizer.transform(texts)
        probs = ml_model_instance._classifier.predict_proba(X_vec)
        return probs

    explainer = LimeTextExplainer(class_names=["safe", "phishing"])

    try:
        explanation = explainer.explain_instance(
            text,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=(1,),  # explain "phishing" class
        )
        raw_features = explanation.as_list(label=1)  # (token, weight) pairs

        features: list[dict[str, Any]] = []
        for token, weight in raw_features:
            features.append({
                "token": token,
                "weight": round(float(weight), 4),
                "direction": "phishing" if weight > 0 else "safe",
                "abs_weight": abs(round(float(weight), 4)),
            })

        features.sort(key=lambda f: f["abs_weight"], reverse=True)

        top_phishing = [f["token"] for f in features if f["direction"] == "phishing"][:8]
        top_safe = [f["token"] for f in features if f["direction"] == "safe"][:5]

        prediction = ml_model_instance.predict(subject=subject, body=body)

        return {
            "features": features,
            "summary": (
                f"LIME identified {len(top_phishing)} phishing indicator(s). "
                f"Top signals: {', '.join(top_phishing[:3]) if top_phishing else 'none'}."
            ),
            "top_phishing_tokens": top_phishing,
            "top_safe_tokens": top_safe,
            "prediction": prediction["label"],
            "confidence": prediction["confidence"],
        }
    except Exception as exc:
        log.warning("LIME explanation failed: %s — using TF-IDF fallback", exc)
        return _fallback_ml_explanation(subject, body, ml_model_instance)


def _fallback_ml_explanation(
    subject: str,
    body: str,
    ml_model_instance: Any,
) -> dict[str, Any]:
    """
    Fallback when LIME is unavailable — uses raw TF-IDF feature weights.
    """
    try:
        features = ml_model_instance.get_feature_weights(subject=subject, body=body)
    except Exception:
        features = []

    prediction = ml_model_instance.predict(subject=subject, body=body)
    top_phishing = [f["token"] for f in features if f["direction"] == "phishing"][:8]
    top_safe = [f["token"] for f in features if f["direction"] == "safe"][:5]

    return {
        "features": [
            {
                "token": f["token"],
                "weight": f["contribution"],
                "direction": f["direction"],
                "abs_weight": abs(f["contribution"]),
            }
            for f in features
        ],
        "summary": (
            f"Top TF-IDF feature analysis: "
            f"{', '.join(top_phishing[:3]) if top_phishing else 'no strong signals'}."
        ),
        "top_phishing_tokens": top_phishing,
        "top_safe_tokens": top_safe,
        "prediction": prediction["label"],
        "confidence": prediction["confidence"],
    }


# ---------------------------------------------------------------------------
# Token highlighting for frontend
# ---------------------------------------------------------------------------

def highlight_tokens(
    body: str,
    phishing_tokens: list[str],
    safe_tokens: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Split email body into annotated spans for the frontend to colour-code.

    Tokens in *phishing_tokens* are marked with ``"danger"``;
    tokens in *safe_tokens* are marked with ``"safe"``;
    everything else is ``"neutral"``.

    Parameters
    ----------
    body:
        Raw email body text.
    phishing_tokens:
        List of tokens/phrases that indicate phishing.
    safe_tokens:
        Optional list of tokens/phrases that indicate legitimate email.

    Returns
    -------
    List of dicts: [{text: str, type: "danger"|"safe"|"neutral"}]
    """
    if safe_tokens is None:
        safe_tokens = []

    if not body.strip():
        return [{"text": "(empty body)", "type": "neutral"}]

    # Build a combined pattern from all tokens (longest first to avoid
    # partial matches swallowing longer tokens)
    all_tokens = sorted(
        [(t, "danger") for t in phishing_tokens] +
        [(t, "safe") for t in safe_tokens],
        key=lambda x: len(x[0]),
        reverse=True,
    )

    if not all_tokens:
        return [{"text": body, "type": "neutral"}]

    pattern = "|".join(re.escape(t) for t, _ in all_tokens)
    token_type_map = {t.lower(): typ for t, typ in all_tokens}

    spans: list[dict[str, Any]] = []
    cursor = 0

    for match in re.finditer(pattern, body, flags=re.IGNORECASE):
        start, end = match.start(), match.end()
        if start > cursor:
            spans.append({"text": body[cursor:start], "type": "neutral"})
        matched_text = match.group(0)
        span_type = token_type_map.get(matched_text.lower(), "neutral")
        spans.append({"text": matched_text, "type": span_type})
        cursor = end

    if cursor < len(body):
        spans.append({"text": body[cursor:], "type": "neutral"})

    return spans


# ---------------------------------------------------------------------------
# Unified explanation entry-point (used by compare.py and app.py)
# ---------------------------------------------------------------------------

def generate_full_explanation(
    subject: str,
    body: str,
    sender: str,
    rule_result: dict[str, Any],
    ml_model_instance: Any | None = None,
) -> dict[str, Any]:
    """
    Generate a complete explanation payload for all three classifiers.

    Parameters
    ----------
    subject, body, sender:
        Email fields.
    rule_result:
        Output from :func:`src.rule_based.analyze`.
    ml_model_instance:
        Optional :class:`src.ml_model.MLModel` instance for LIME explanations.
        If *None*, ML explanation is skipped.
    use_lime:
        Whether to attempt LIME (set False to use fast TF-IDF fallback).

    Returns
    -------
    dict with keys ``rule_based``, ``ml_model``, ``token_highlights``.
    """
    rule_explanation = explain_rule_based(rule_result)

    ml_explanation: dict[str, Any] = {}
    if ml_model_instance is not None:
        try:
            ml_explanation = explain_ml_model(
                subject, body, ml_model_instance, num_features=10
            )
        except Exception as exc:
            log.warning("ML explanation (LIME) failed: %s — using TF-IDF fallback", exc)
            try:
                ml_explanation = _fallback_ml_explanation(
                    subject, body, ml_model_instance
                )
            except Exception as fb_exc:
                log.error("ML explanation fallback also failed: %s", fb_exc)
                ml_explanation = {"summary": "ML explanation unavailable.", "features": []}

    # Gather tokens for highlighting
    phishing_tokens: list[str] = []
    if rule_explanation.get("rules_fired"):
        # Extract trigger words from rule detail strings
        for rule in rule_explanation["rules_fired"]:
            detail = rule.get("detail", "")
            found = re.findall(r"'([^']+)'", detail)
            phishing_tokens.extend(found)

    if ml_explanation.get("top_phishing_tokens"):
        phishing_tokens.extend(ml_explanation["top_phishing_tokens"])

    safe_tokens: list[str] = ml_explanation.get("top_safe_tokens", [])

    highlights = highlight_tokens(body, phishing_tokens[:15], safe_tokens[:5])

    return {
        "rule_based": rule_explanation,
        "ml_model": ml_explanation,
        "token_highlights": highlights,
    }
