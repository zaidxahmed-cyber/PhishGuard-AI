"""
Gmail API email fetcher.

Authenticates via OAuth2 (credentials.json) and retrieves the last *N*
emails from the inbox, returning them as structured dicts ready for
classification by any of the three detector modules.

Setup (one-time)
----------------
1. Enable the Gmail API in Google Cloud Console.
2. Create OAuth2 credentials (Desktop app type).
3. Download ``credentials.json`` and place it in the project root.
4. On first run the browser will open for consent; a ``token.json``
   refresh token is stored locally so subsequent runs are headless.

Never commit ``credentials.json`` or ``token.json`` to version control.

Environment variables (override defaults)
------------------------------------------
    GMAIL_CREDENTIALS_PATH   path to credentials.json   (default: ./credentials.json)
    GMAIL_TOKEN_PATH         path to token.json          (default: ./token.json)
    GMAIL_SCOPES             space-separated OAuth scopes (default: readonly)
"""

from __future__ import annotations

import base64
import email as email_lib
import html
import logging
import os
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (can be overridden via environment variables)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent

CREDENTIALS_PATH: Path = Path(
    os.getenv("GMAIL_CREDENTIALS_PATH", str(_ROOT / "credentials.json"))
)
TOKEN_PATH: Path = Path(
    os.getenv("GMAIL_TOKEN_PATH", str(_ROOT / "token.json"))
)
SCOPES: list[str] = os.getenv(
    "GMAIL_SCOPES", "https://www.googleapis.com/auth/gmail.readonly"
).split()


# ---------------------------------------------------------------------------
# Lazy import of Google API libraries
# ---------------------------------------------------------------------------

def _require_google_libs() -> tuple[Any, Any, Any]:
    """
    Import google-auth and googleapiclient.

    Raises ImportError with a helpful install message if unavailable.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        return Request, Credentials, InstalledAppFlow
    except ImportError as exc:
        raise ImportError(
            "Google API libraries are required for Gmail integration.\n"
            "Install with:  pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client"
        ) from exc


def _require_build() -> Any:
    try:
        from googleapiclient.discovery import build
        return build
    except ImportError as exc:
        raise ImportError(
            "google-api-python-client is required.\n"
            "Install with:  pip install google-api-python-client"
        ) from exc


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def get_credentials(
    credentials_path: Path = CREDENTIALS_PATH,
    token_path: Path = TOKEN_PATH,
    scopes: list[str] = SCOPES,
) -> Any:
    """
    Obtain valid Google OAuth2 credentials.

    Loads a cached token if available; otherwise runs the browser-based
    consent flow.  A refreshed token is persisted for future runs.

    Parameters
    ----------
    credentials_path:
        Path to the OAuth2 ``credentials.json`` downloaded from Google Cloud.
    token_path:
        Path where the access/refresh token is cached.
    scopes:
        List of OAuth2 scope strings.

    Returns
    -------
    google.oauth2.credentials.Credentials
    """
    Request, Credentials, InstalledAppFlow = _require_google_libs()

    creds: Any = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            log.info("Refreshing expired Gmail token …")
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                raise FileNotFoundError(
                    f"credentials.json not found at {credentials_path}.\n"
                    "Download it from Google Cloud Console → APIs & Services → Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path), scopes
            )
            creds = flow.run_local_server(port=0)

        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")
        log.info("Token saved to %s", token_path)

    return creds


# ---------------------------------------------------------------------------
# Gmail API helpers
# ---------------------------------------------------------------------------

def _build_service(
    credentials_path: Path = CREDENTIALS_PATH,
    token_path: Path = TOKEN_PATH,
) -> Any:
    """Build and return an authenticated Gmail API service object."""
    build = _require_build()
    creds = get_credentials(credentials_path, token_path)
    return build("gmail", "v1", credentials=creds)


def _decode_body_part(part: dict[str, Any]) -> str:
    """Decode a single MIME body part to UTF-8 string."""
    data = part.get("body", {}).get("data", "")
    if not data:
        return ""
    decoded_bytes = base64.urlsafe_b64decode(data + "==")
    try:
        return decoded_bytes.decode("utf-8", errors="replace")
    except Exception:
        return decoded_bytes.decode("latin-1", errors="replace")


def _extract_parts(payload: dict[str, Any]) -> tuple[str, str]:
    """
    Recursively walk MIME payload parts to extract plain text and HTML.

    Returns
    -------
    (plain_text, html_text)
    """
    plain, html_content = "", ""
    mime_type = payload.get("mimeType", "")

    if mime_type == "text/plain":
        plain = _decode_body_part(payload)
    elif mime_type == "text/html":
        html_content = _decode_body_part(payload)
    elif "parts" in payload:
        for part in payload["parts"]:
            p, h = _extract_parts(part)
            plain += p
            html_content += h

    return plain, html_content


def _html_to_text(html_str: str) -> str:
    """Strip HTML tags and unescape entities from an HTML string."""
    text = re.sub(r"<style[^>]*>.*?</style>", " ", html_str, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_links(html_str: str, plain_str: str) -> list[str]:
    """Extract all hyperlinks from HTML and plain-text."""
    links: list[str] = []
    # HTML href attributes
    links += re.findall(r'href=["\']([^"\']+)["\']', html_str, re.IGNORECASE)
    # Plain-text URLs
    links += re.findall(r"https?://[^\s\"'<>]+", plain_str)
    return list(dict.fromkeys(links))  # deduplicate while preserving order


def _parse_message(msg_data: dict[str, Any]) -> dict[str, Any]:
    """
    Transform a raw Gmail API message dict into a structured email dict.

    Returns
    -------
    dict with keys:
        id, subject, sender, date, snippet, body_plain, body_html,
        body_text (clean readable text), links (list[str]), raw_headers (dict)
    """
    payload = msg_data.get("payload", {})
    headers: list[dict[str, str]] = payload.get("headers", [])

    header_map: dict[str, str] = {
        h["name"].lower(): h["value"] for h in headers
    }

    subject = header_map.get("subject", "(no subject)")
    sender = header_map.get("from", "")
    date = header_map.get("date", "")

    plain, html_content = _extract_parts(payload)
    body_text = plain.strip() if plain.strip() else _html_to_text(html_content)
    links = _extract_links(html_content, plain)

    return {
        "id": msg_data.get("id", ""),
        "thread_id": msg_data.get("threadId", ""),
        "subject": subject,
        "sender": sender,
        "date": date,
        "snippet": msg_data.get("snippet", ""),
        "body_plain": plain,
        "body_html": html_content,
        "body_text": body_text,
        "links": links,
        "raw_headers": header_map,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_emails(
    max_results: int = 20,
    query: str = "in:inbox",
    credentials_path: Path = CREDENTIALS_PATH,
    token_path: Path = TOKEN_PATH,
    include_spam_trash: bool = False,
) -> list[dict[str, Any]]:
    """
    Fetch and parse the most recent emails from Gmail.

    Parameters
    ----------
    max_results:
        Maximum number of emails to retrieve (capped at 500 by the API).
    query:
        Gmail search query string (default: inbox only).
        Examples: ``"is:unread"``, ``"from:noreply@amazon.com"``.
    credentials_path:
        Path to ``credentials.json``.
    token_path:
        Path to cached ``token.json``.
    include_spam_trash:
        Whether to include messages from Spam and Trash.

    Returns
    -------
    List of parsed email dicts (see :func:`_parse_message`).
    """
    service = _build_service(credentials_path, token_path)

    log.info("Fetching up to %d messages (query='%s') …", max_results, query)
    list_response = service.users().messages().list(
        userId="me",
        q=query,
        maxResults=min(max_results, 500),
        includeSpamTrash=include_spam_trash,
    ).execute()

    messages_meta = list_response.get("messages", [])
    if not messages_meta:
        log.info("No messages found matching query '%s'", query)
        return []

    emails: list[dict[str, Any]] = []
    for meta in messages_meta:
        try:
            msg_data = service.users().messages().get(
                userId="me",
                id=meta["id"],
                format="full",
            ).execute()
            emails.append(_parse_message(msg_data))
        except Exception as exc:
            log.warning("Failed to fetch message %s: %s", meta["id"], exc)

    log.info("Successfully fetched %d emails", len(emails))
    return emails


def fetch_single_email(
    message_id: str,
    credentials_path: Path = CREDENTIALS_PATH,
    token_path: Path = TOKEN_PATH,
) -> dict[str, Any]:
    """
    Fetch and parse a single Gmail message by its ID.

    Parameters
    ----------
    message_id:
        The Gmail message ID string.
    credentials_path:
        Path to ``credentials.json``.
    token_path:
        Path to cached ``token.json``.

    Returns
    -------
    Parsed email dict.
    """
    service = _build_service(credentials_path, token_path)
    msg_data = service.users().messages().get(
        userId="me",
        id=message_id,
        format="full",
    ).execute()
    return _parse_message(msg_data)


def email_to_classifier_input(email: dict[str, Any]) -> dict[str, str]:
    """
    Convert a fetched email dict to the flat input expected by all classifiers.

    Parameters
    ----------
    email:
        Dict returned by :func:`fetch_emails` or :func:`fetch_single_email`.

    Returns
    -------
    dict with keys: ``subject``, ``body``, ``sender``.
    """
    return {
        "subject": email.get("subject", ""),
        "body": email.get("body_text", ""),
        "sender": email.get("sender", ""),
    }
