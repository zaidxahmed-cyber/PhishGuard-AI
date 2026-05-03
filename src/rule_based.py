"""
Rule-based phishing email detector.

Evaluates emails against a curated set of heuristic rules:
  - Urgency / threat language keywords
  - Financial bait keywords
  - Suspicious URL patterns (IP-in-link, URL shorteners, lookalike domains)
  - Sender domain anomalies (free providers impersonating corporate senders)
  - Excessive punctuation / CAPS abuse
  - Common phishing call-to-action phrases

Each triggered rule contributes a weighted score. Final confidence is the
normalised score clamped to [0, 1]. Labels above PHISHING_THRESHOLD are
classified as "phishing".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------
PHISHING_THRESHOLD: float = 0.40  # score ≥ this → "phishing"


# ---------------------------------------------------------------------------
# Rule weight definitions
# ---------------------------------------------------------------------------
# Each entry: (rule_id, human-readable description, weight 0-1, pattern/list)

URGENCY_KEYWORDS: list[str] = [
    "urgent", "immediately", "act now", "right away", "final notice",
    "last chance", "limited time", "expire", "expires", "expiring",
    "deadline", "action required", "response required", "don't delay",
    "do not ignore", "account suspended", "account locked", "account disabled",
    "account on hold", "account terminated", "suspended", "disabled",
    "verify now", "confirm now", "update now", "login now", "click now",
    "warning", "alert", "critical", "important notice",
]

FINANCIAL_KEYWORDS: list[str] = [
    "prize", "winner", "won", "lottery", "jackpot", "congratulations",
    "million dollars", "billion dollars", "inheritance", "fund transfer",
    "wire transfer", "western union", "moneygram", "bitcoin", "cryptocurrency",
    "refund", "reimbursement", "unclaimed", "uncashed", "compensation",
    "reward", "bonus", "free gift", "gift card", "claim your",
    "bank account", "credit card", "ssn", "social security",
    "tax refund", "irs", "hmrc", "payment required", "invoice",
    "outstanding balance", "overdue payment",
]

THREAT_KEYWORDS: list[str] = [
    "legal action", "sue", "lawsuit", "criminal", "prosecute",
    "arrest", "police", "fbi", "irs investigation", "fraud",
    "identity theft", "unauthorized access", "hacked", "compromised",
    "data breach", "security breach", "virus", "malware",
    "report you", "reported", "penalty", "fine", "fee",
]

CTA_KEYWORDS: list[str] = [
    "click here", "click the link", "click below", "follow this link",
    "open the attachment", "download the attachment", "open attachment",
    "confirm your", "verify your", "update your", "provide your",
    "enter your", "submit your", "sign in here", "login here",
    "reset your password", "change your password",
]

# Free / consumer email providers that are suspicious when impersonating
# corporate-looking senders (e.g. "paypal-support@gmail.com")
FREE_EMAIL_PROVIDERS: set[str] = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "aol.com", "mail.com", "protonmail.com", "icloud.com",
    "ymail.com", "live.com", "msn.com", "rediffmail.com",
    "zohomail.com", "inbox.com",
}

# Known brand names that should never originate from free providers
IMPERSONATED_BRANDS: list[str] = [
    "paypal", "amazon", "apple", "microsoft", "google", "netflix",
    "facebook", "instagram", "twitter", "linkedin", "dropbox",
    "bank", "chase", "wellsfargo", "citibank", "barclays",
    "hsbc", "americanexpress", "amex", "visa", "mastercard",
    "irs", "hmrc", "usps", "fedex", "ups", "dhl",
]

# URL shortener hostnames
URL_SHORTENERS: set[str] = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "short.link", "buff.ly", "dlvr.it", "ift.tt", "is.gd",
    "v.gd", "rb.gy", "cutt.ly", "shorturl.at", "tiny.cc",
}

# Regex patterns
_RE_IP_IN_URL = re.compile(
    r"https?://(?:\d{1,3}\.){3}\d{1,3}",
    re.IGNORECASE,
)
_RE_URL = re.compile(
    r"https?://[^\s\"'<>]+",
    re.IGNORECASE,
)
_RE_LOOKALIKE_CHARS = re.compile(
    r"(?:paypa1|paypаl|amaz0n|g00gle|micros0ft|appie|netf1ix|faceb00k)",
    re.IGNORECASE,
)
_RE_EXCESSIVE_CAPS = re.compile(r"\b[A-Z]{5,}\b")
_RE_EXCESSIVE_PUNCT = re.compile(r"[!?]{2,}")
_RE_OBFUSCATED_URL = re.compile(r"https?://[^/\s]+@[^/\s]+", re.IGNORECASE)
_RE_DATA_URI = re.compile(r"data:[^;]+;base64,", re.IGNORECASE)
_RE_HTML_HIDDEN = re.compile(
    r'style\s*=\s*["\'][^"\']*(?:display\s*:\s*none|visibility\s*:\s*hidden)',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class RuleResult:
    """Return value from :func:`analyze`."""

    label: str                          # "phishing" | "safe"
    confidence: float                   # 0.0 – 1.0
    score: float                        # raw weighted score (0.0 – 1.0+)
    triggered_rules: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (JSON-safe)."""
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "score": round(self.score, 4),
            "triggered_rules": self.triggered_rules,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Lower-case, collapse whitespace."""
    return " ".join(text.lower().split())


def _extract_urls(text: str) -> list[str]:
    """Return all HTTP/HTTPS URLs found in *text*."""
    return _RE_URL.findall(text)


def _sender_domain(sender: str) -> str:
    """Extract the domain part from an email address or display-name string."""
    match = re.search(r"[\w.+-]+@([\w.-]+\.\w+)", sender)
    return match.group(1).lower() if match else ""


# ---------------------------------------------------------------------------
# Individual rule checkers  (each returns list[dict] of triggered sub-rules)
# ---------------------------------------------------------------------------

def _check_urgency(text_lower: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for kw in URGENCY_KEYWORDS:
        if kw in text_lower:
            hits.append({
                "rule": "urgency_keyword",
                "detail": f"Urgency keyword detected: '{kw}'",
                "weight": 0.15,
            })
            break  # one hit per rule category is enough for scoring
    return hits


def _check_financial(text_lower: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    matched: list[str] = [kw for kw in FINANCIAL_KEYWORDS if kw in text_lower]
    if matched:
        hits.append({
            "rule": "financial_bait",
            "detail": f"Financial bait keywords: {matched[:5]}",
            "weight": 0.20,
        })
    return hits


def _check_threats(text_lower: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    matched = [kw for kw in THREAT_KEYWORDS if kw in text_lower]
    if matched:
        hits.append({
            "rule": "threat_language",
            "detail": f"Threat / legal-action language: {matched[:5]}",
            "weight": 0.20,
        })
    return hits


def _check_cta(text_lower: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    matched = [kw for kw in CTA_KEYWORDS if kw in text_lower]
    if matched:
        hits.append({
            "rule": "phishing_cta",
            "detail": f"Phishing call-to-action phrases: {matched[:3]}",
            "weight": 0.15,
        })
    return hits


def _check_urls(text: str) -> list[dict[str, Any]]:
    """Analyse all URLs found in the email body."""
    hits: list[dict[str, Any]] = []
    urls = _extract_urls(text)

    if not urls:
        return hits

    # IP address in URL
    for url in urls:
        if _RE_IP_IN_URL.match(url):
            hits.append({
                "rule": "ip_in_url",
                "detail": f"IP address used instead of domain name: {url[:80]}",
                "weight": 0.30,
            })
            break

    # URL shortener
    for url in urls:
        try:
            host = urlparse(url).netloc.lower().lstrip("www.")
            if host in URL_SHORTENERS:
                hits.append({
                    "rule": "url_shortener",
                    "detail": f"URL shortener detected: {host}",
                    "weight": 0.20,
                })
                break
        except Exception:
            pass

    # Obfuscated URL (user:pass@host)
    if _RE_OBFUSCATED_URL.search(text):
        hits.append({
            "rule": "obfuscated_url",
            "detail": "Credentials embedded in URL (user@host trick)",
            "weight": 0.35,
        })

    # Lookalike / homoglyph domains
    if _RE_LOOKALIKE_CHARS.search(text):
        hits.append({
            "rule": "lookalike_domain",
            "detail": "Homoglyph / lookalike brand name in URL or body",
            "weight": 0.35,
        })

    # Mismatched visible vs. href text
    mismatch_re = re.compile(
        r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]{4,})</a>',
        re.IGNORECASE | re.DOTALL,
    )
    for href, visible in mismatch_re.findall(text):
        if visible.strip().lower().startswith("http") and href.lower() != visible.strip().lower():
            hits.append({
                "rule": "mismatched_anchor",
                "detail": "Anchor text URL differs from href",
                "weight": 0.25,
            })
            break

    return hits


def _check_sender(sender: str, subject_body_lower: str) -> list[dict[str, Any]]:
    """Flag free-provider senders impersonating known brands."""
    hits: list[dict[str, Any]] = []
    if not sender:
        return hits

    domain = _sender_domain(sender)
    if not domain:
        return hits

    if domain in FREE_EMAIL_PROVIDERS:
        for brand in IMPERSONATED_BRANDS:
            if brand in subject_body_lower or brand in sender.lower():
                hits.append({
                    "rule": "sender_impersonation",
                    "detail": (
                        f"Free provider '{domain}' impersonating brand '{brand}'"
                    ),
                    "weight": 0.40,
                })
                break
        else:
            # Free provider but no brand match — mild flag
            hits.append({
                "rule": "free_provider_sender",
                "detail": f"Email sent from consumer provider: {domain}",
                "weight": 0.05,
            })

    # Subdomain spoofing: paypal.com.evil.net
    if re.search(
        r"(?:paypal|amazon|microsoft|apple|google)\.com\.",
        domain,
        re.IGNORECASE,
    ):
        hits.append({
            "rule": "subdomain_spoofing",
            "detail": f"Domain spoofing via subdomain: {domain}",
            "weight": 0.45,
        })

    return hits


def _check_formatting(text: str, text_lower: str) -> list[dict[str, Any]]:
    """Flag excessive CAPS, repeated punctuation, hidden HTML elements."""
    hits: list[dict[str, Any]] = []

    caps_hits = _RE_EXCESSIVE_CAPS.findall(text)
    if len(caps_hits) >= 3:
        hits.append({
            "rule": "excessive_caps",
            "detail": f"Excessive ALL-CAPS words detected: {caps_hits[:5]}",
            "weight": 0.10,
        })

    punct_hits = _RE_EXCESSIVE_PUNCT.findall(text)
    if len(punct_hits) >= 2:
        hits.append({
            "rule": "excessive_punctuation",
            "detail": "Repeated !!! or ??? punctuation",
            "weight": 0.08,
        })

    if _RE_HTML_HIDDEN.search(text):
        hits.append({
            "rule": "hidden_html_content",
            "detail": "Hidden HTML elements (display:none / visibility:hidden)",
            "weight": 0.25,
        })

    if _RE_DATA_URI.search(text):
        hits.append({
            "rule": "data_uri_attachment",
            "detail": "Base64 data-URI embedded — possible hidden payload",
            "weight": 0.30,
        })

    return hits


def _check_mismatched_reply_to(sender: str, raw_headers: dict[str, str]) -> list[dict[str, Any]]:
    """Flag when Reply-To domain differs from From domain."""
    hits: list[dict[str, Any]] = []
    reply_to = raw_headers.get("reply-to", "")
    if not reply_to:
        return hits

    from_domain = _sender_domain(sender)
    reply_domain = _sender_domain(reply_to)

    if from_domain and reply_domain and from_domain != reply_domain:
        hits.append({
            "rule": "mismatched_reply_to",
            "detail": (
                f"Reply-To domain '{reply_domain}' differs from "
                f"From domain '{from_domain}'"
            ),
            "weight": 0.30,
        })
    return hits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(
    subject: str = "",
    body: str = "",
    sender: str = "",
    raw_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Analyse a single email and return a rule-based classification result.

    Parameters
    ----------
    subject:
        Email subject line.
    body:
        Plain-text or HTML body of the email.
    sender:
        The From header value, e.g. ``"PayPal Support <support@gmail.com>"``.
    raw_headers:
        Optional dict of lowercase header names → values for extended checks
        (currently used for Reply-To comparison).

    Returns
    -------
    dict with keys:
        - ``label``:           ``"phishing"`` or ``"safe"``
        - ``confidence``:      float in [0, 1]
        - ``score``:           raw cumulative weight (may exceed 1 before clamping)
        - ``triggered_rules``: list of dicts describing each fired rule
    """
    if raw_headers is None:
        raw_headers = {}

    combined = f"{subject} {body}"
    combined_lower = _norm(combined)

    triggered: list[dict[str, Any]] = []
    triggered += _check_urgency(combined_lower)
    triggered += _check_financial(combined_lower)
    triggered += _check_threats(combined_lower)
    triggered += _check_cta(combined_lower)
    triggered += _check_urls(combined)
    triggered += _check_sender(sender, combined_lower)
    triggered += _check_formatting(combined, combined_lower)
    triggered += _check_mismatched_reply_to(sender, raw_headers)

    # Deduplicate by rule id (keep highest-weight entry)
    seen: dict[str, dict[str, Any]] = {}
    for rule in triggered:
        rid = rule["rule"]
        if rid not in seen or rule["weight"] > seen[rid]["weight"]:
            seen[rid] = rule
    unique_triggered = list(seen.values())

    raw_score: float = sum(r["weight"] for r in unique_triggered)
    # Sigmoid-inspired normalisation so score stays in (0, 1)
    confidence: float = min(raw_score / (raw_score + 0.5), 0.99) if raw_score > 0 else 0.01

    label = "phishing" if confidence >= PHISHING_THRESHOLD else "safe"

    result = RuleResult(
        label=label,
        confidence=round(confidence, 4),
        score=round(raw_score, 4),
        triggered_rules=unique_triggered,
    )
    return result.to_dict()
