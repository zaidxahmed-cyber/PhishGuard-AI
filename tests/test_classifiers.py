"""
Unit and integration tests for PhishGuard AI classifiers.

Run with:
    pytest tests/ -v

Test coverage:
    - rule_based.py  — deterministic checks for every rule category
    - ml_model.py    — text cleaning, pipeline construction, inference (mock)
    - compare.py     — ensemble logic, agreement computation
    - explainability — rule explanation formatting, token highlighting
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from typing import Any


# ============================================================
# RULE-BASED TESTS
# ============================================================

class TestRuleBasedAnalyzer:
    """Tests for src.rule_based.analyze()."""

    def setup_method(self):
        from src.rule_based import analyze
        self.analyze = analyze

    # ----------------------------------------------------------
    def test_clean_email_returns_safe(self):
        result = self.analyze(
            subject="Team lunch on Friday",
            body="Hi everyone, just a reminder about the team lunch. See you at 12:30.",
            sender="alice@techcorp.com",
        )
        assert result["label"] == "safe"
        assert result["confidence"] < 0.40
        assert isinstance(result["triggered_rules"], list)

    def test_urgency_keyword_flagged(self):
        result = self.analyze(
            subject="URGENT: Account suspended",
            body="Your account has been suspended. Act now to restore access.",
            sender="noreply@gmail.com",
        )
        triggered_ids = [r["rule"] for r in result["triggered_rules"]]
        assert "urgency_keyword" in triggered_ids

    def test_financial_bait_flagged(self):
        result = self.analyze(
            subject="You have won a lottery prize",
            body="Congratulations! You have won $1,000,000. Claim your prize now.",
            sender="info@gmail.com",
        )
        triggered_ids = [r["rule"] for r in result["triggered_rules"]]
        assert "financial_bait" in triggered_ids
        assert result["label"] == "phishing"

    def test_ip_in_url_flagged(self):
        result = self.analyze(
            subject="Click here",
            body="Please verify at http://192.168.1.100/login",
            sender="test@example.com",
        )
        triggered_ids = [r["rule"] for r in result["triggered_rules"]]
        assert "ip_in_url" in triggered_ids

    def test_url_shortener_flagged(self):
        result = self.analyze(
            subject="Important link",
            body="Click here: http://bit.ly/xyz123abc",
            sender="test@example.com",
        )
        triggered_ids = [r["rule"] for r in result["triggered_rules"]]
        assert "url_shortener" in triggered_ids

    def test_sender_impersonation_flagged(self):
        result = self.analyze(
            subject="PayPal account update",
            body="Please verify your PayPal account.",
            sender="paypal-support@gmail.com",
        )
        triggered_ids = [r["rule"] for r in result["triggered_rules"]]
        assert "sender_impersonation" in triggered_ids

    def test_threat_language_flagged(self):
        result = self.analyze(
            subject="Legal action will be taken",
            body="If you do not respond, legal action and criminal prosecution will follow.",
            sender="legal@example.com",
        )
        triggered_ids = [r["rule"] for r in result["triggered_rules"]]
        assert "threat_language" in triggered_ids

    def test_cta_phrases_flagged(self):
        result = self.analyze(
            subject="Action required",
            body="Click here to verify your account. Confirm your details now.",
            sender="noreply@site.com",
        )
        triggered_ids = [r["rule"] for r in result["triggered_rules"]]
        assert "phishing_cta" in triggered_ids

    def test_mismatched_reply_to_flagged(self):
        result = self.analyze(
            subject="Important",
            body="Please reply to verify.",
            sender="ceo@legit-corp.com",
            raw_headers={"reply-to": "attacker@evil.com"},
        )
        triggered_ids = [r["rule"] for r in result["triggered_rules"]]
        assert "mismatched_reply_to" in triggered_ids

    def test_result_schema(self):
        result = self.analyze(subject="Test", body="Hello world", sender="x@y.com")
        assert "label" in result
        assert "confidence" in result
        assert "score" in result
        assert "triggered_rules" in result
        assert result["label"] in ("phishing", "safe")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_empty_email_handled(self):
        result = self.analyze(subject="", body="", sender="")
        assert result["label"] in ("phishing", "safe")
        assert result["confidence"] >= 0.0

    def test_confidence_increases_with_rules(self):
        few_rules = self.analyze(
            subject="Urgent",
            body="Act now!",
            sender="test@gmail.com",
        )
        many_rules = self.analyze(
            subject="URGENT: Account SUSPENDED!!!",
            body="Click here to verify your account immediately. Legal action! Claim your prize. "
                 "http://bit.ly/fake-link Enter your SSN and credit card.",
            sender="paypal-verify@gmail.com",
        )
        assert many_rules["confidence"] >= few_rules["confidence"]

    def test_lookalike_domain_flagged(self):
        result = self.analyze(
            subject="Paypal notice",
            body="Login at http://paypa1.com/secure",
            sender="support@paypa1.com",
        )
        triggered_ids = [r["rule"] for r in result["triggered_rules"]]
        assert "lookalike_domain" in triggered_ids

    def test_high_severity_rules_have_higher_weight(self):
        result = self.analyze(
            subject="Verify",
            body="http://user:pass@malicious.com/login",
            sender="test@x.com",
        )
        for rule in result["triggered_rules"]:
            if rule["rule"] == "obfuscated_url":
                assert rule["weight"] >= 0.30


# ============================================================
# ML MODEL TESTS (unit — mocked sklearn)
# ============================================================

class TestMLModelTextCleaning:
    """Tests for text preprocessing used by ml_model."""

    def test_clean_text_lowercase(self):
        from src.ml_model import clean_text
        assert clean_text("HELLO WORLD") == "hello world"

    def test_clean_text_strips_html(self):
        from src.ml_model import clean_text
        result = clean_text("<b>Important</b> <br/> message")
        assert "<b>" not in result
        assert "important" in result

    def test_clean_text_replaces_url(self):
        from src.ml_model import clean_text
        result = clean_text("Visit http://evil.com/login for details")
        assert "url" in result
        assert "evil.com" not in result

    def test_clean_text_replaces_email(self):
        from src.ml_model import clean_text
        result = clean_text("Contact user@example.com for help")
        assert "email" in result

    def test_clean_text_empty_string(self):
        from src.ml_model import clean_text
        assert clean_text("") == ""

    def test_clean_text_whitespace_collapsed(self):
        from src.ml_model import clean_text
        result = clean_text("hello    \n\t   world")
        assert result == "hello world"


class TestMLModelInference:
    """Tests for MLModel.predict() with mocked artefacts."""

    def _make_mock_model(self) -> Any:
        from src.ml_model import MLModel
        model = MLModel.__new__(MLModel)
        model._vectorizer_path = MagicMock()
        model._classifier_path = MagicMock()

        import numpy as np
        mock_vec = MagicMock()
        mock_vec.transform.return_value = MagicMock(
            indices=np.array([0, 1]),
            tocsr=lambda: MagicMock(indices=np.array([0]), __getitem__=lambda s, x: np.array([[0.5, 0.3]])),
        )
        mock_vec.get_feature_names_out.return_value = np.array(["click", "verify"])

        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = np.array([[0.1, 0.9]])
        mock_clf.coef_ = np.array([[0.5, -0.3]])

        model._vectorizer = mock_vec
        model._classifier = mock_clf
        return model

    def test_predict_returns_phishing(self):
        model = self._make_mock_model()
        result = model.predict(subject="Urgent", body="Click here now")
        assert result["label"] == "phishing"
        assert result["confidence"] >= 0.5
        assert result["model"] == "tfidf_logistic_regression"

    def test_predict_schema(self):
        model = self._make_mock_model()
        result = model.predict(subject="Test", body="Hello")
        assert "label" in result
        assert "confidence" in result
        assert "model" in result

    def test_predict_empty_text(self):
        model = self._make_mock_model()
        model._classifier.predict_proba.return_value = __import__('numpy').array([[0.9, 0.1]])
        result = model.predict(subject="", body="")
        assert result["label"] == "safe"


# ============================================================
# COMPARE ENGINE TESTS
# ============================================================

class TestEnsembleAgreement:
    """Tests for _compute_agreement() in compare.py."""

    def setup_method(self):
        from src.compare import _compute_agreement
        self.compute = _compute_agreement

    def _result(self, label: str, conf: float) -> dict:
        return {"label": label, "confidence": conf}

    def test_full_agreement_phishing(self):
        out = self.compute(
            self._result("phishing", 0.95),
            self._result("phishing", 0.90),
            self._result("phishing", 0.92),
        )
        assert out["ensemble_label"] == "phishing"
        assert out["agreement_level"] == "full"
        assert out["threat_score"] == 100 or out["threat_score"] > 85

    def test_full_agreement_safe(self):
        out = self.compute(
            self._result("safe", 0.05),
            self._result("safe", 0.08),
            self._result("safe", 0.06),
        )
        assert out["ensemble_label"] == "safe"
        assert out["agreement_level"] == "full"
        assert out["threat_score"] < 15

    def test_majority_vote_phishing(self):
        out = self.compute(
            self._result("phishing", 0.85),
            self._result("phishing", 0.80),
            self._result("safe",     0.15),
        )
        assert out["ensemble_label"] == "phishing"
        assert out["agreement_level"] in ("majority", "full")

    def test_unavailable_classifier_excluded(self):
        out = self.compute(
            self._result("phishing", 0.90),
            {"label": "unavailable", "confidence": 0.0},
            {"label": "error", "confidence": 0.0},
        )
        # Only one valid classifier — should still work
        assert out["ensemble_label"] in ("phishing", "safe")

    def test_threat_score_range(self):
        for conf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            out = self.compute(
                self._result("phishing", conf),
                self._result("phishing", conf),
                self._result("phishing", conf),
            )
            assert 0 <= out["threat_score"] <= 100

    def test_votes_dict_present(self):
        out = self.compute(
            self._result("phishing", 0.9),
            self._result("safe", 0.1),
            self._result("phishing", 0.85),
        )
        assert "votes" in out
        assert len(out["votes"]) == 3


# ============================================================
# EXPLAINABILITY TESTS
# ============================================================

class TestExplainRuleBased:

    def setup_method(self):
        from src.explainability import explain_rule_based
        self.explain = explain_rule_based

    def test_phishing_result_summary(self):
        rule_result = {
            "label": "phishing",
            "confidence": 0.85,
            "triggered_rules": [
                {"rule": "urgency_keyword", "detail": "Urgency keyword detected: 'urgent'", "weight": 0.15},
                {"rule": "financial_bait", "detail": "Financial bait keywords: ['lottery']", "weight": 0.20},
            ],
        }
        exp = self.explain(rule_result)
        assert exp["verdict"] == "phishing"
        assert exp["rule_count"] == 2
        assert "PHISHING" in exp["summary"].upper()
        assert exp["top_rule"]["rule"] in ("financial_bait", "urgency_keyword")

    def test_safe_result_summary(self):
        rule_result = {"label": "safe", "confidence": 0.05, "triggered_rules": []}
        exp = self.explain(rule_result)
        assert exp["verdict"] == "safe"
        assert exp["rule_count"] == 0
        assert exp["top_rule"] is None

    def test_rules_sorted_by_weight(self):
        rule_result = {
            "label": "phishing",
            "confidence": 0.7,
            "triggered_rules": [
                {"rule": "low_rule", "detail": "x", "weight": 0.05},
                {"rule": "high_rule", "detail": "y", "weight": 0.40},
            ],
        }
        exp = self.explain(rule_result)
        assert exp["rules_fired"][0]["rule"] == "high_rule"

    def test_severity_assignment(self):
        rule_result = {
            "label": "phishing",
            "confidence": 0.9,
            "triggered_rules": [
                {"rule": "critical_rule", "detail": "x", "weight": 0.45},
                {"rule": "high_rule",     "detail": "y", "weight": 0.30},
                {"rule": "medium_rule",   "detail": "z", "weight": 0.15},
                {"rule": "low_rule",      "detail": "w", "weight": 0.05},
            ],
        }
        exp = self.explain(rule_result)
        severities = {r["rule"]: r["severity"] for r in exp["rules_fired"]}
        assert severities["critical_rule"] == "critical"
        assert severities["high_rule"]     == "high"
        assert severities["medium_rule"]   == "medium"
        assert severities["low_rule"]      == "low"


class TestTokenHighlighting:

    def setup_method(self):
        from src.explainability import highlight_tokens
        self.highlight = highlight_tokens

    def test_phishing_tokens_marked_danger(self):
        spans = self.highlight(
            "Please verify your account immediately.",
            phishing_tokens=["verify", "immediately"],
        )
        types = {s["text"].lower(): s["type"] for s in spans}
        assert types.get("verify") == "danger"
        assert types.get("immediately") == "danger"

    def test_neutral_text_preserved(self):
        spans = self.highlight("Hello world", phishing_tokens=["click"])
        combined = "".join(s["text"] for s in spans)
        assert combined == "Hello world"

    def test_empty_body_returns_placeholder(self):
        spans = self.highlight("", phishing_tokens=["urgent"])
        assert len(spans) == 1
        assert spans[0]["type"] == "neutral"

    def test_safe_tokens_marked_safe(self):
        spans = self.highlight(
            "This is a legitimate newsletter from our team.",
            phishing_tokens=[],
            safe_tokens=["newsletter", "team"],
        )
        types = {s["text"].lower(): s["type"] for s in spans}
        assert types.get("newsletter") == "safe"

    def test_no_tokens_returns_single_neutral_span(self):
        spans = self.highlight("Hello world.", phishing_tokens=[], safe_tokens=[])
        assert len(spans) == 1
        assert spans[0]["type"] == "neutral"
        assert spans[0]["text"] == "Hello world."


# ============================================================
# INTEGRATION SMOKE TEST (no models required)
# ============================================================

class TestCompareSmoke:
    """Integration smoke test for compare() — runs rule-based only."""

    def test_compare_rule_based_only(self):
        """compare() should always return a result even if ML models are absent."""
        from src.compare import compare

        result = compare(
            subject="URGENT: Your account has been suspended!!!",
            body="Click here to verify. Act now or lose access. Claim your prize.",
            sender="paypal-support@gmail.com",
            include_explanation=True,
        )

        # Core shape
        assert "rule_based" in result
        assert "ml_model" in result
        assert "transformer" in result
        assert "ensemble" in result
        assert "latency_ms" in result

        # Rule-based should always work
        assert result["rule_based"]["label"] in ("phishing", "safe")
        assert 0 <= result["ensemble"]["threat_score"] <= 100

        # ML / transformer may be unavailable — but should not raise
        assert result["ml_model"].get("label") in ("phishing", "safe", "unavailable", "error")
        assert result["transformer"].get("label") in ("phishing", "safe", "unavailable", "error")

    def test_compare_clean_email(self):
        from src.compare import compare

        result = compare(
            subject="Team meeting at 3pm",
            body="Hi all, just a reminder about the team meeting today at 3pm.",
            sender="boss@company.com",
        )
        # Rule-based should classify as safe
        assert result["rule_based"]["label"] == "safe"

    def test_compare_batch(self):
        from src.compare import compare_batch

        emails = [
            {"subject": "Urgent!", "body": "Click here now!", "sender": "x@gmail.com"},
            {"subject": "Lunch?",  "body": "Want to grab lunch?", "sender": "friend@work.com"},
        ]
        results = compare_batch(emails, include_explanation=False)
        assert len(results) == 2
        for r in results:
            assert "rule_based" in r
            assert "ensemble" in r
