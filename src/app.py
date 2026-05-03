"""
PhishGuard AI — FastAPI backend.

Serves the single-page frontend (templates/index.html) and exposes a
REST API for email classification and Gmail integration.

Run
---
    uvicorn src.app:app --reload --port 8000

Environment Variables (see .env.example)
-----------------------------------------
    HOST            uvicorn host     (default: 0.0.0.0)
    PORT            uvicorn port     (default: 8000)
    LOG_LEVEL       logging level    (default: info)
    USE_LIME        enable LIME      (default: false — too slow for interactive use)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

load_dotenv()

log = logging.getLogger("phishguard")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "info").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
TEMPLATE_DIR = _ROOT / "templates"
SAMPLE_EMAILS_DIR = _ROOT / "sample_emails"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PhishGuard AI",
    description="AI-powered phishing email detection with rule-based, ML, and Transformer classifiers.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class EmailInput(BaseModel):
    """Input payload for the /api/analyze endpoint."""

    subject: str = Field(default="", max_length=998, description="Email subject line")
    body: str = Field(default="", max_length=100_000, description="Email body (plain text or HTML)")
    sender: str = Field(default="", max_length=320, description="From header value")
    raw_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Optional raw email headers (lowercase keys)",
    )
    include_explanation: bool = Field(
        default=True,
        description="Include LIME / rule explainability in the response",
    )

    @field_validator("body")
    @classmethod
    def body_not_empty(cls, v: str) -> str:
        if not v.strip() and not v:
            raise ValueError("Email body cannot be completely empty")
        return v


class GmailFetchInput(BaseModel):
    """Input for the /api/gmail/fetch endpoint."""

    max_results: int = Field(default=10, ge=1, le=100)
    query: str = Field(default="in:inbox", max_length=500)
    analyze: bool = Field(default=True, description="Run classification on fetched emails")


class SampleEmailInput(BaseModel):
    """Request a pre-loaded sample email by filename."""

    filename: str = Field(..., description="Filename from /api/sample-emails list")


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    """Serve the PhishGuard AI single-page application."""
    index = TEMPLATE_DIR / "index.html"
    if not index.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Frontend not built. Expected templates/index.html",
        )
    return FileResponse(str(index), media_type="text/html")


# ---------------------------------------------------------------------------
# Core analysis endpoint
# ---------------------------------------------------------------------------

@app.post("/api/analyze", summary="Analyse a single email")
async def analyze_email(payload: EmailInput) -> JSONResponse:
    """
    Run the email through all three detectors (Rule-Based, ML, DistilBERT)
    and return a comparison report.

    Returns
    -------
    JSON with fields:
        rule_based, ml_model, transformer, ensemble,
        explanation, latency_ms, input_preview
    """
    if not payload.subject.strip() and not payload.body.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one of 'subject' or 'body' must be non-empty.",
        )

    try:
        from src.compare import compare

        result = compare(
            subject=payload.subject,
            body=payload.body,
            sender=payload.sender,
            raw_headers=payload.raw_headers,
            include_explanation=payload.include_explanation,
        )
        return JSONResponse(content=result)

    except Exception as exc:
        log.error("Analysis failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification error: {exc}",
        )


# ---------------------------------------------------------------------------
# Sample emails endpoints
# ---------------------------------------------------------------------------

@app.get("/api/sample-emails", summary="List available sample emails")
async def list_sample_emails() -> JSONResponse:
    """
    Return metadata for all sample emails in the ``sample_emails/`` directory.

    Each entry includes filename, category (phishing / safe), and subject
    extracted from the first line of the file.
    """
    if not SAMPLE_EMAILS_DIR.exists():
        return JSONResponse(content={"samples": []})

    samples: list[dict[str, str]] = []
    for txt_file in sorted(SAMPLE_EMAILS_DIR.glob("*.txt")):
        category = "phishing" if txt_file.stem.startswith("phishing") else "safe"
        try:
            first_lines = txt_file.read_text(encoding="utf-8").splitlines()
            subject = ""
            for line in first_lines[:5]:
                if line.lower().startswith("subject:"):
                    subject = line[8:].strip()
                    break
        except Exception:
            subject = txt_file.stem

        samples.append({
            "filename": txt_file.name,
            "category": category,
            "subject": subject,
        })

    return JSONResponse(content={"samples": samples})


@app.get("/api/sample-emails/{filename}", summary="Get content of a sample email")
async def get_sample_email(filename: str) -> JSONResponse:
    """
    Return the parsed content of a sample email file.

    The .txt file format is:
        Subject: <subject>
        From: <sender>
        Body:
        <body text>
    """
    # Sanitise: only allow simple filenames, no path traversal
    safe_name = Path(filename).name
    file_path = SAMPLE_EMAILS_DIR / safe_name

    if not file_path.exists() or file_path.suffix != ".txt":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample email '{safe_name}' not found",
        )

    try:
        content = file_path.read_text(encoding="utf-8")
        parsed = _parse_sample_file(content)
        return JSONResponse(content=parsed)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read sample file: {exc}",
        )


def _parse_sample_file(content: str) -> dict[str, str]:
    """Parse the key: value header format used in sample email .txt files."""
    lines = content.splitlines()
    subject, sender, body_lines = "", "", []
    in_body = False

    for line in lines:
        if in_body:
            body_lines.append(line)
        elif line.lower().startswith("subject:"):
            subject = line[8:].strip()
        elif line.lower().startswith("from:"):
            sender = line[5:].strip()
        elif line.lower().startswith("body:") or line.strip() == "":
            if line.lower().startswith("body:"):
                rest = line[5:].strip()
                if rest:
                    body_lines.append(rest)
                in_body = True

    return {
        "subject": subject,
        "sender": sender,
        "body": "\n".join(body_lines).strip(),
    }


# ---------------------------------------------------------------------------
# Gmail integration endpoints
# ---------------------------------------------------------------------------

@app.post("/api/gmail/fetch", summary="Fetch and optionally classify Gmail inbox")
async def fetch_gmail(payload: GmailFetchInput) -> JSONResponse:
    """
    Authenticate with Gmail (OAuth2) and fetch recent emails.

    Requires ``credentials.json`` in the project root.
    On first call, opens a browser window for OAuth consent.
    """
    try:
        from src.gmail_fetch import fetch_emails, email_to_classifier_input

        emails = fetch_emails(
            max_results=payload.max_results,
            query=payload.query,
        )

        if not payload.analyze:
            return JSONResponse(content={"emails": emails, "count": len(emails)})

        from src.compare import compare

        analyzed: list[dict[str, Any]] = []
        for email in emails:
            classifier_input = email_to_classifier_input(email)
            analysis = compare(**classifier_input, include_explanation=False)
            analyzed.append({
                "email": {
                    "id": email.get("id"),
                    "subject": email.get("subject"),
                    "sender": email.get("sender"),
                    "date": email.get("date"),
                    "snippet": email.get("snippet"),
                    "links": email.get("links", [])[:10],
                },
                "analysis": analysis,
            })

        return JSONResponse(content={"emails": analyzed, "count": len(analyzed)})

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Gmail API libraries not installed: {exc}",
        )
    except Exception as exc:
        log.error("Gmail fetch failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gmail fetch error: {exc}",
        )


# ---------------------------------------------------------------------------
# Health / model status
# ---------------------------------------------------------------------------

@app.get("/api/health", summary="Health check")
async def health_check() -> JSONResponse:
    """Return service health and model availability status."""
    from src.ml_model import CLASSIFIER_PATH, VECTORIZER_PATH
    from src.transformer_model import TRANSFORMER_MODEL_DIR

    return JSONResponse(content={
        "status": "ok",
        "models": {
            "rule_based": "always_available",
            "ml_model": "ready" if CLASSIFIER_PATH.exists() else "not_trained",
            "transformer": "ready" if TRANSFORMER_MODEL_DIR.exists() else "not_trained",
        },
        "sample_emails": len(list(SAMPLE_EMAILS_DIR.glob("*.txt"))) if SAMPLE_EMAILS_DIR.exists() else 0,
    })


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(404)
async def not_found_handler(request: Request, _exc: Any) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"detail": f"Endpoint {request.url.path!r} not found"},
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, _exc: Any) -> JSONResponse:
    log.exception("Unhandled server error at %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error — check application logs"},
    )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "true").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
