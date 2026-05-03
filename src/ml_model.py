"""
TF-IDF + Logistic Regression phishing classifier.

Dataset
-------
Expects a CSV at ``data/processed/emails.csv`` with columns:
    - ``text``  : combined subject + body
    - ``label`` : 1 = phishing / spam,  0 = safe / ham

The script :func:`prepare_dataset` merges the two recommended public datasets:

1. **Enron Spam Dataset**
   URL: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
   (or the pre-labelled version: https://github.com/MWiechmann/enron_spam_data)

2. **CEAS 2008 Phishing Dataset**
   URL: http://www.ceas.cc/2008/

After downloading, place CSVs in ``data/raw/`` and run:
    python -m src.ml_model --train

The trained artefacts are saved to ``models/``:
    - ``models/tfidf_vectorizer.joblib``
    - ``models/lr_classifier.joblib``
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Optional visualisation — soft-import so the module loads without matplotlib
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = _ROOT / "data" / "processed"
DATA_RAW = _ROOT / "data" / "raw"
FIGURES_DIR = _ROOT / "data" / "figures"
MODELS_DIR = _ROOT / "models"

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
CLASSIFIER_PATH = MODELS_DIR / "lr_classifier.joblib"
PROCESSED_CSV = DATA_PROCESSED / "emails.csv"

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalise email text for TF-IDF ingestion.

    - Lower-case
    - Collapse whitespace
    - Strip HTML tags
    - Remove email headers artifacts (Re:, Fwd:)
    """
    import re
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML
    text = re.sub(r"https?://\S+", " URL ", text)  # replace URLs with token
    text = re.sub(r"\S+@\S+", " EMAIL ", text)     # replace addresses
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_dataset(
    enron_csv: Path | None = None,
    ceas_csv: Path | None = None,
    output_csv: Path = PROCESSED_CSV,
) -> pd.DataFrame:
    """
    Merge, clean, and save the combined dataset.

    Parameters
    ----------
    enron_csv:
        Path to the Enron labelled CSV (columns: ``subject``, ``body``,
        ``label`` where label is 1=spam/phishing, 0=ham).
    ceas_csv:
        Path to CEAS 2008 CSV (columns: ``subject``, ``body``,
        ``label`` where 1=phishing).
    output_csv:
        Destination path for the merged processed CSV.

    Returns
    -------
    pd.DataFrame with columns ``text`` and ``label``.
    """
    frames: list[pd.DataFrame] = []

    for csv_path in filter(None, [enron_csv, ceas_csv]):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower().str.strip()

        # Flexible column normalisation
        if "text" not in df.columns:
            subject_col = next((c for c in df.columns if "subject" in c), None)
            body_col = next(
                (c for c in df.columns if c in ("body", "message", "content")),
                None,
            )
            parts = []
            if subject_col:
                parts.append(df[subject_col].fillna(""))
            if body_col:
                parts.append(df[body_col].fillna(""))
            df["text"] = parts[0] if len(parts) == 1 else parts[0] + " " + parts[1]

        label_col = next(
            (c for c in df.columns if c in ("label", "spam", "phishing", "class")),
            None,
        )
        if label_col is None:
            raise ValueError(f"Cannot find label column in {csv_path}")
        df["label"] = df[label_col].astype(int)
        frames.append(df[["text", "label"]].copy())

    if not frames:
        raise FileNotFoundError(
            "No dataset CSVs provided. Pass enron_csv and/or ceas_csv."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["text"] = combined["text"].apply(clean_text)
    combined.dropna(subset=["text", "label"], inplace=True)
    combined.drop_duplicates(subset=["text"], inplace=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    log.info("Saved %d samples to %s", len(combined), output_csv)
    return combined


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """
    Construct the sklearn Pipeline (TF-IDF → Logistic Regression).

    TF-IDF parameters are tuned for email text:
    - sublinear_tf dampens term frequencies
    - ngram_range (1,2) captures common phishing bigrams
    - min_df=3 removes noise terms
    """
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=50_000,
        sublinear_tf=True,
        min_df=3,
        stop_words="english",
    )
    classifier = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",  # handles class imbalance
        solver="lbfgs",
        random_state=42,
    )
    return Pipeline([("tfidf", vectorizer), ("lr", classifier)])


def train(
    df: pd.DataFrame | None = None,
    data_csv: Path = PROCESSED_CSV,
    test_size: float = 0.20,
    random_state: int = 42,
    save: bool = True,
) -> dict[str, Any]:
    """
    Train the TF-IDF + LR model and return evaluation metrics.

    Parameters
    ----------
    df:
        Pre-loaded DataFrame.  If *None*, loads from *data_csv*.
    data_csv:
        CSV path used when *df* is not supplied.
    test_size:
        Fraction of data reserved for evaluation.
    random_state:
        Seed for reproducibility.
    save:
        Whether to persist the trained pipeline artefacts.

    Returns
    -------
    dict containing ``pipeline``, ``metrics``, and split data arrays.
    """
    if df is None:
        if not data_csv.exists():
            raise FileNotFoundError(
                f"Processed dataset not found at {data_csv}.\n"
                "Run prepare_dataset() first or pass a DataFrame directly."
            )
        df = pd.read_csv(data_csv)

    X = df["text"].astype(str).tolist()
    y = df["label"].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    log.info(
        "Train: %d  |  Test: %d  |  Phishing ratio (train): %.2f%%",
        len(X_train), len(X_test),
        100 * sum(y_train) / len(y_train),
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics: dict[str, float] = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }
    log.info("Metrics: %s", metrics)
    log.info("\n%s", classification_report(y_test, y_pred, target_names=["safe", "phishing"]))

    if save:
        _save_artefacts(pipeline)

    _plot_confusion_matrix(confusion_matrix(y_test, y_pred))

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def _save_artefacts(pipeline: Pipeline) -> None:
    """Persist vectorizer and classifier components separately."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline.named_steps["tfidf"], VECTORIZER_PATH)
    joblib.dump(pipeline.named_steps["lr"], CLASSIFIER_PATH)
    log.info("Saved artefacts to %s", MODELS_DIR)


def _plot_confusion_matrix(cm: np.ndarray) -> None:
    """Save a styled confusion matrix heat-map to data/figures/."""
    if not _PLOT_AVAILABLE:
        log.warning("matplotlib not available — skipping confusion matrix plot")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Safe", "Predicted Phishing"],
        yticklabels=["Actual Safe", "Actual Phishing"],
        ax=ax,
    )
    ax.set_title("TF-IDF + LR — Confusion Matrix")
    fig.tight_layout()
    out = FIGURES_DIR / "lr_confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Confusion matrix saved to %s", out)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class MLModel:
    """
    Thin wrapper around the saved TF-IDF + LR artefacts.

    Loads artefacts lazily on first call to :meth:`predict`.
    """

    def __init__(
        self,
        vectorizer_path: Path = VECTORIZER_PATH,
        classifier_path: Path = CLASSIFIER_PATH,
    ) -> None:
        self._vectorizer_path = vectorizer_path
        self._classifier_path = classifier_path
        self._vectorizer: TfidfVectorizer | None = None
        self._classifier: LogisticRegression | None = None

    def _load(self) -> None:
        """Load artefacts from disk (called lazily)."""
        if not self._vectorizer_path.exists():
            raise FileNotFoundError(
                f"Vectorizer not found at {self._vectorizer_path}. "
                "Run training first: python -m src.ml_model --train"
            )
        if not self._classifier_path.exists():
            raise FileNotFoundError(
                f"Classifier not found at {self._classifier_path}. "
                "Run training first: python -m src.ml_model --train"
            )
        self._vectorizer = joblib.load(self._vectorizer_path)
        self._classifier = joblib.load(self._classifier_path)
        log.debug("ML artefacts loaded from %s", self._vectorizer_path.parent)

    @property
    def is_loaded(self) -> bool:
        """True once artefacts are successfully loaded."""
        return self._vectorizer is not None and self._classifier is not None

    def predict(
        self,
        subject: str = "",
        body: str = "",
    ) -> dict[str, Any]:
        """
        Classify a single email.

        Parameters
        ----------
        subject:
            Email subject line.
        body:
            Email body (plain text or HTML).

        Returns
        -------
        dict with keys ``label``, ``confidence``, ``model``.
        """
        if not self.is_loaded:
            self._load()

        text = clean_text(f"{subject} {body}")
        X_vec = self._vectorizer.transform([text])  # type: ignore[union-attr]
        prob = self._classifier.predict_proba(X_vec)[0]  # type: ignore[union-attr]
        phishing_prob = float(prob[1])
        label = "phishing" if phishing_prob >= 0.50 else "safe"

        return {
            "label": label,
            "confidence": round(phishing_prob, 4),
            "model": "tfidf_logistic_regression",
        }

    def get_feature_weights(
        self,
        subject: str = "",
        body: str = "",
        top_n: int = 15,
    ) -> list[dict[str, Any]]:
        """
        Return the top *top_n* TF-IDF features and their contribution to the
        phishing decision for a specific email.

        Used by :mod:`src.explainability`.
        """
        if not self.is_loaded:
            self._load()

        text = clean_text(f"{subject} {body}")
        X_vec = self._vectorizer.transform([text])  # type: ignore[union-attr]
        feature_names = np.array(self._vectorizer.get_feature_names_out())  # type: ignore
        coefficients = self._classifier.coef_[0]  # type: ignore[union-attr]

        # Non-zero TF-IDF features present in this document
        cx = X_vec.tocsr()
        non_zero_indices = cx.indices
        tfidf_values = np.asarray(cx[0, non_zero_indices]).flatten()

        contributions = tfidf_values * coefficients[non_zero_indices]
        order = np.argsort(np.abs(contributions))[::-1][:top_n]

        features = []
        for idx in order:
            orig_idx = non_zero_indices[idx]
            features.append({
                "token": feature_names[orig_idx],
                "tfidf": round(float(tfidf_values[idx]), 4),
                "coefficient": round(float(coefficients[orig_idx]), 4),
                "contribution": round(float(contributions[idx]), 4),
                "direction": "phishing" if contributions[idx] > 0 else "safe",
            })
        return features


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_model: MLModel | None = None


def get_model() -> MLModel:
    """Return the module-level singleton :class:`MLModel` instance."""
    global _model
    if _model is None:
        _model = MLModel()
    return _model


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate the TF-IDF + LR model")
    parser.add_argument("--train", action="store_true", help="Run training pipeline")
    parser.add_argument("--enron", type=Path, default=None, help="Path to Enron CSV")
    parser.add_argument("--ceas", type=Path, default=None, help="Path to CEAS CSV")
    args = parser.parse_args()

    if args.train:
        if args.enron or args.ceas:
            df = prepare_dataset(enron_csv=args.enron, ceas_csv=args.ceas)
        else:
            df = None
        results = train(df=df)
        print("\n=== Training Complete ===")
        for k, v in results["metrics"].items():
            print(f"  {k:12s}: {v:.4f}")
