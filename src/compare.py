"""
Multi-classifier comparison engine for PhishGuard AI.

Runs a single email through all three detectors in parallel threads and
returns a unified comparison report.  This is the primary entry-point
used by :mod:`src.app` to power the API endpoint.

Classifier order:
    1. Rule-Based  — :mod:`src.rule_based`
    2. ML Model    — :mod:`src.ml_model`   (TF-IDF + Logistic Regression)
    3. Transformer — :mod:`src.transformer_model` (DistilBERT)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy model singletons — loaded on first request
# ---------------------------------------------------------------------------
_ml_model: Any = None
_transformer_model: Any = None


def _get_ml_model() -> Any:
    global _ml_model
    if _ml_model is None:
        from src.ml_model import get_model
        _ml_model = get_model()
    return _ml_model


def _get_transformer_model() -> Any:
    global _transformer_model
    if _transformer_model is None:
        from src.transformer_model import get_model
        _transformer_model = get_model()
    return _transformer_model


# ---------------------------------------------------------------------------
# Individual runner functions (called in threads)
# ---------------------------------------------------------------------------

def _run_rule_based(
    subject: str,
    body: str,
    sender: str,
    raw_headers: dict[str, str],
) -> dict[str, Any]:
    """Run the rule-based detector and return its result dict."""
    from src.rule_based import analyze
    return analyze(subject=subject, body=body, sender=sender, raw_headers=raw_headers)


def _run_ml_model(subject: str, body: str) -> dict[str, Any]:
    """Run the TF-IDF + LR model and return its result dict."""
    try:
        model = _get_ml_model()
        return model.predict(subject=subject, body=body)
    except FileNotFoundError as exc:
        log.warning("ML model not trained: %s", exc)
        return {
            "label": "unavailable",
            "confidence": 0.0,
            "model": "tfidf_logistic_regression",
            "error": "Model not trained. Run: python -m src.ml_model --train",
        }
    except Exception as exc:
        log.error("ML model error: %s", exc, exc_info=True)
        return {
            "label": "error",
            "confidence": 0.0,
            "model": "tfidf_logistic_regression",
            "error": str(exc),
        }


def _run_transformer(subject: str, body: str) -> dict[str, Any]:
    """Run the DistilBERT model and return its result dict."""
    try:
        model = _get_transformer_model()
        return model.predict(subject=subject, body=body)
    except FileNotFoundError as exc:
        log.warning("Transformer model not found: %s", exc)
        return {
            "label": "unavailable",
            "confidence": 0.0,
            "model": "distilbert-phishing",
            "error": "Model not trained. Run: python -m src.transformer_model --train",
        }
    except Exception as exc:
        log.error("Transformer model error: %s", exc, exc_info=True)
        return {
            "label": "error",
            "confidence": 0.0,
            "model": "distilbert-phishing",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Agreement / ensemble logic
# ---------------------------------------------------------------------------

def _compute_agreement(
    rb: dict[str, Any],
    ml: dict[str, Any],
    tr: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute ensemble verdict and classifier agreement statistics.

    Classifiers that returned "unavailable" or "error" are excluded from
    the vote.

    Returns
    -------
    dict with keys:
        - ``ensemble_label``    : "phishing" | "safe"
        - ``ensemble_confidence``: float 0-1 (weighted average of valid classifiers)
        - ``agreement_level``   : "full" | "majority" | "split"
        - ``votes``             : dict of classifier → label
        - ``threat_score``      : int 0-100 (UI gauge value)
    """
    classifiers = {
        "rule_based": rb,
        "ml_model": ml,
        "transformer": tr,
    }

    valid = {
        name: res
        for name, res in classifiers.items()
        if res.get("label") in ("phishing", "safe")
    }

    if not valid:
        return {
            "ensemble_label": "unknown",
            "ensemble_confidence": 0.0,
            "agreement_level": "none",
            "votes": {k: v.get("label", "error") for k, v in classifiers.items()},
            "threat_score": 0,
        }

    votes = {name: res["label"] for name, res in valid.items()}

    # Weighted confidence average (transformers get slightly higher weight)
    weights = {"rule_based": 1.0, "ml_model": 1.2, "transformer": 1.5}
    phishing_score = 0.0
    total_weight = 0.0

    for name, res in valid.items():
        w = weights.get(name, 1.0)
        conf = res.get("confidence", 0.5)
        if res["label"] == "phishing":
            phishing_score += conf * w
        else:
            phishing_score += (1.0 - conf) * w * 0  # safe contribution is 0 to phishing score
            # We want phishing_score to reflect the phishing probability
            phishing_score += conf * w if res["label"] == "phishing" else 0
        total_weight += w

    # Recompute cleanly
    phishing_score = 0.0
    total_weight = 0.0
    for name, res in valid.items():
        w = weights.get(name, 1.0)
        conf = res.get("confidence", 0.5)
        # confidence always = probability of phishing
        phishing_score += conf * w
        total_weight += w

    ensemble_conf = phishing_score / total_weight if total_weight else 0.0
    ensemble_label = "phishing" if ensemble_conf >= 0.50 else "safe"

    # Agreement
    phishing_count = sum(1 for v in votes.values() if v == "phishing")
    n = len(votes)
    if phishing_count == n or phishing_count == 0:
        agreement_level = "full"
    elif phishing_count >= n / 2:
        agreement_level = "majority"
    else:
        agreement_level = "split"

    threat_score = int(round(ensemble_conf * 100))

    return {
        "ensemble_label": ensemble_label,
        "ensemble_confidence": round(ensemble_conf, 4),
        "agreement_level": agreement_level,
        "votes": {k: classifiers[k].get("label", "error") for k in classifiers},
        "threat_score": threat_score,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare(
    subject: str = "",
    body: str = "",
    sender: str = "",
    raw_headers: dict[str, str] | None = None,
    include_explanation: bool = True,
) -> dict[str, Any]:
    """
    Run all three phishing detectors on a single email and return a
    unified comparison report.

    Parameters
    ----------
    subject:
        Email subject line.
    body:
        Email body (plain text preferred).
    sender:
        From header value.
    raw_headers:
        Optional full header dict for extended rule checks.
    include_explanation:
        Whether to generate LIME / rule explanations (adds latency).

    Returns
    -------
    dict with keys:
        - ``rule_based``        : result from rule_based.analyze
        - ``ml_model``          : result from ml_model.predict
        - ``transformer``       : result from transformer_model.predict
        - ``ensemble``          : agreement + weighted verdict
        - ``explanation``       : (optional) explainability output
        - ``latency_ms``        : dict of per-classifier timings
        - ``input_preview``     : truncated subject + body snippet
    """
    if raw_headers is None:
        raw_headers = {}

    t_total_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Run all three classifiers concurrently
    # ------------------------------------------------------------------
    timings: dict[str, float] = {}
    results: dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map: dict[Future, str] = {
            executor.submit(_run_rule_based, subject, body, sender, raw_headers): "rule_based",
            executor.submit(_run_ml_model, subject, body): "ml_model",
            executor.submit(_run_transformer, subject, body): "transformer",
        }

        t_start = {f: time.perf_counter() for f in future_map}

        for future in as_completed(future_map):
            name = future_map[future]
            elapsed = round((time.perf_counter() - t_start[future]) * 1000, 1)
            timings[name] = elapsed
            try:
                results[name] = future.result()
            except Exception as exc:
                log.error("Classifier '%s' raised: %s", name, exc, exc_info=True)
                results[name] = {
                    "label": "error",
                    "confidence": 0.0,
                    "error": str(exc),
                }

    ensemble = _compute_agreement(
        results["rule_based"],
        results["ml_model"],
        results["transformer"],
    )

    # ------------------------------------------------------------------
    # Optional explainability
    # ------------------------------------------------------------------
    explanation: dict[str, Any] = {}
    if include_explanation:
        try:
            from src.explainability import generate_full_explanation

            ml_instance: Any = None
            try:
                ml_instance = _get_ml_model()
            except Exception as ml_exc:
                log.debug("Could not load ML model for explanation: %s", ml_exc)

            explanation = generate_full_explanation(
                subject=subject,
                body=body,
                sender=sender,
                rule_result=results["rule_based"],
                ml_model_instance=ml_instance,
            )
        except Exception as exc:
            log.warning("Explanation generation failed: %s", exc)
            explanation = {"error": str(exc)}

    total_ms = round((time.perf_counter() - t_total_start) * 1000, 1)
    timings["total"] = total_ms

    return {
        "rule_based": results["rule_based"],
        "ml_model": results["ml_model"],
        "transformer": results["transformer"],
        "ensemble": ensemble,
        "explanation": explanation,
        "latency_ms": timings,
        "input_preview": {
            "subject": subject[:120],
            "body_snippet": body[:300],
            "sender": sender[:100],
        },
    }


def compare_batch(
    emails: list[dict[str, str]],
    include_explanation: bool = False,
) -> list[dict[str, Any]]:
    """
    Classify a list of emails.

    Parameters
    ----------
    emails:
        List of dicts with keys ``subject``, ``body``, ``sender``.
    include_explanation:
        Pass True to include explanations (significantly slower for large batches).

    Returns
    -------
    List of comparison result dicts (same structure as :func:`compare`).
    """
    results: list[dict[str, Any]] = []
    for email in emails:
        result = compare(
            subject=email.get("subject", ""),
            body=email.get("body", ""),
            sender=email.get("sender", ""),
            include_explanation=include_explanation,
        )
        result["_input"] = email
        results.append(result)
    return results
