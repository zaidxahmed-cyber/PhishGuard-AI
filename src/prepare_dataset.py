"""
Dataset preparation script for PhishGuard AI.

Merges all 7 source CSVs from dataset/ into a single clean
data/processed/emails.csv ready for model training.

Source files expected in dataset/:
  CEAS_08.csv         — sender, receiver, date, subject, body, label, urls
  Enron.csv           — subject, body, label
  Ling.csv            — subject, body, label
  Nazario.csv         — sender, receiver, date, subject, body, urls, label
  Nigerian_Fraud.csv  — sender, receiver, date, subject, body, urls, label
  phishing_email.csv  — text_combined, label
  SpamAssasin.csv     — sender, receiver, date, subject, body, label, urls

Labels: 1 = phishing / spam,  0 = safe / ham

Run:
    python -m src.prepare_dataset
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_ROOT       = Path(__file__).parent.parent
DATASET_DIR = _ROOT / "dataset"
OUTPUT_CSV  = _ROOT / "data" / "processed" / "emails.csv"


# ---------------------------------------------------------------------------
# Per-file normalisation rules
# ---------------------------------------------------------------------------

def _load_standard(path: Path) -> pd.DataFrame:
    """Load a CSV that has 'subject' + 'body' + 'label' columns."""
    df = pd.read_csv(path, low_memory=False)
    df["subject"] = df.get("subject", pd.Series([""] * len(df))).fillna("")
    df["body"]    = df.get("body",    pd.Series([""] * len(df))).fillna("")
    df["text"]    = (df["subject"] + " " + df["body"]).str.strip()
    df["label"]   = df["label"].astype(int)
    return df[["text", "label"]]


def _load_text_combined(path: Path) -> pd.DataFrame:
    """Load phishing_email.csv which uses a pre-combined 'text_combined' column."""
    df = pd.read_csv(path, low_memory=False)
    df.rename(columns={"text_combined": "text"}, inplace=True)
    df["text"]  = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]]


# ---------------------------------------------------------------------------
# Source map
# ---------------------------------------------------------------------------

SOURCE_MAP: dict[str, callable] = {
    "CEAS_08.csv":        _load_standard,
    "Enron.csv":          _load_standard,
    "Ling.csv":           _load_standard,
    "Nazario.csv":        _load_standard,
    "Nigerian_Fraud.csv": _load_standard,
    "phishing_email.csv": _load_text_combined,
    "SpamAssasin.csv":    _load_standard,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def merge_and_save(
    dataset_dir: Path = DATASET_DIR,
    output_csv:  Path = OUTPUT_CSV,
    min_text_len: int = 10,
) -> pd.DataFrame:
    """
    Merge all source CSVs, clean, deduplicate, and save.

    Parameters
    ----------
    dataset_dir:
        Directory containing the raw source CSVs.
    output_csv:
        Destination for the merged processed CSV.
    min_text_len:
        Drop rows where text is shorter than this many characters.

    Returns
    -------
    pd.DataFrame with columns ``text`` and ``label``.
    """
    frames: list[pd.DataFrame] = []

    for filename, loader in SOURCE_MAP.items():
        path = dataset_dir / filename
        if not path.exists():
            log.warning("File not found — skipping: %s", path)
            continue
        log.info("Loading %-30s ...", filename)
        df = loader(path)
        log.info("  → %d rows  |  phishing=%d  safe=%d",
                 len(df), (df["label"] == 1).sum(), (df["label"] == 0).sum())
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No dataset CSVs found in {dataset_dir}")

    combined = pd.concat(frames, ignore_index=True)
    log.info("Combined: %d rows before cleaning", len(combined))

    # Clean
    from src.ml_model import clean_text
    combined["text"] = combined["text"].apply(clean_text)

    # Drop empties and short texts
    combined = combined[combined["text"].str.len() >= min_text_len]

    # Deduplicate on exact text
    before_dedup = len(combined)
    combined.drop_duplicates(subset=["text"], inplace=True)
    log.info("Removed %d duplicate rows", before_dedup - len(combined))

    # Shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    phish = (combined["label"] == 1).sum()
    safe  = (combined["label"] == 0).sum()
    log.info("Saved %d rows → %s", len(combined), output_csv)
    log.info("Final distribution: phishing=%d (%.1f%%)  safe=%d (%.1f%%)",
             phish, 100 * phish / len(combined),
             safe,  100 * safe  / len(combined))

    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    merge_and_save()
