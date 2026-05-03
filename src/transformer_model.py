"""
DistilBERT-based phishing email classifier.

Fine-tunes ``distilbert-base-uncased`` on the processed email dataset and
exposes a simple :class:`TransformerModel` inference class.

Training
--------
    python -m src.transformer_model --train [--epochs 3] [--batch-size 16]

The fine-tuned model is saved to ``models/distilbert-phishing/``.

Requirements
------------
    transformers >= 4.30
    torch >= 2.0         (CPU is fine for fine-tuning on small datasets)
    datasets >= 2.0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = _ROOT / "data" / "processed"
MODELS_DIR = _ROOT / "models"
TRANSFORMER_MODEL_DIR = MODELS_DIR / "distilbert-phishing"

PROCESSED_CSV = DATA_PROCESSED / "emails.csv"
BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256   # tokens — balances context vs. GPU/CPU memory
LABEL2ID = {"safe": 0, "phishing": 1}
ID2LABEL = {0: "safe", 1: "phishing"}


# ---------------------------------------------------------------------------
# Lazy-import heavy dependencies so the module stays importable even when
# transformers / torch are not installed (e.g. during rule_based-only runs).
# ---------------------------------------------------------------------------

def _import_transformers() -> tuple[Any, Any, Any, Any]:
    """Import HuggingFace + torch and return key classes."""
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        return AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError(
            "HuggingFace Transformers is required for the DistilBERT model.\n"
            "Install with:  pip install transformers torch datasets"
        ) from exc


def _import_datasets() -> Any:
    try:
        from datasets import Dataset
        return Dataset
    except ImportError as exc:
        raise ImportError(
            "HuggingFace datasets library required.\n"
            "Install with:  pip install datasets"
        ) from exc


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def _load_hf_dataset(
    csv_path: Path,
    tokenizer: Any,
    max_length: int,
    max_samples: int | None = None,
) -> Any:
    """
    Load the processed CSV as a HuggingFace Dataset, tokenize, and return
    train/test splits.
    """
    import pandas as pd

    Dataset = _import_datasets()

    df = pd.read_csv(csv_path)[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    if max_samples is not None:
        # Stratified subsample — take equal slices from each class
        n_per_class = max_samples // 2
        phishing = df[df["label"] == 1].sample(min(len(df[df["label"] == 1]), n_per_class), random_state=42)
        safe     = df[df["label"] == 0].sample(min(len(df[df["label"] == 0]), n_per_class), random_state=42)
        df = pd.concat([phishing, safe]).sample(frac=1, random_state=42).reset_index(drop=True)
        log.info("Subsampled to %d rows for CPU training", len(df))

    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.train_test_split(test_size=0.20, seed=42)

    def tokenize(batch: dict[str, list]) -> dict[str, list]:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(tokenize, batched=True, batch_size=256, remove_columns=["text"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return dataset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_csv: Path = PROCESSED_CSV,
    output_dir: Path = TRANSFORMER_MODEL_DIR,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 32,
    warmup_steps: int = 200,
    weight_decay: float = 0.01,
    learning_rate: float = 2e-5,
    max_length: int = MAX_LENGTH,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """
    Fine-tune DistilBERT for binary phishing classification.

    Parameters
    ----------
    data_csv:
        Processed dataset CSV (columns: ``text``, ``label``).
    output_dir:
        Directory where the fine-tuned model and tokenizer are saved.
    num_train_epochs:
        Number of full passes over the training data.
    per_device_train_batch_size:
        Training micro-batch size per GPU/CPU.
    per_device_eval_batch_size:
        Evaluation micro-batch size.
    warmup_steps:
        LR scheduler warm-up steps.
    weight_decay:
        L2 regularisation coefficient.
    learning_rate:
        Peak learning rate.
    max_length:
        Maximum token sequence length.

    Returns
    -------
    dict with ``output_dir`` and HuggingFace ``TrainOutput``.
    """
    if not data_csv.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {data_csv}. "
            "Run ml_model.prepare_dataset() first."
        )

    AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments = (
        _import_transformers()
    )

    log.info("Loading tokenizer: %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    log.info("Tokenising dataset from %s …", data_csv)
    dataset = _load_hf_dataset(data_csv, tokenizer, max_length, max_samples)

    log.info("Loading base model: %s", BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        report_to="none",
        fp16=False,
    )

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        import numpy as np
        logits, labels = eval_pred
        probs = _softmax(logits)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probs)),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )

    log.info("Starting fine-tuning …")
    train_output = trainer.train()
    log.info("Training complete. Saving model to %s", output_dir)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = trainer.evaluate()
    log.info("Final eval metrics: %s", metrics)

    return {"output_dir": str(output_dir), "train_output": train_output, "eval_metrics": metrics}


def _softmax(x: Any) -> Any:
    """Numerically stable softmax along last axis."""
    import numpy as np
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class TransformerModel:
    """
    Inference wrapper for the fine-tuned DistilBERT model.

    Artefacts are loaded lazily on the first call to :meth:`predict`.
    Falls back to the base (un-fine-tuned) model if the fine-tuned
    version is not present and ``fallback=True``.
    """

    def __init__(
        self,
        model_dir: Path = TRANSFORMER_MODEL_DIR,
        max_length: int = MAX_LENGTH,
        device: str | None = None,
        fallback: bool = True,
    ) -> None:
        self._model_dir = model_dir
        self._max_length = max_length
        self._device = device
        self._fallback = fallback
        self._tokenizer: Any = None
        self._model: Any = None
        self._pipeline: Any = None

    def _load(self) -> None:
        """Load model and tokenizer from disk (or fall back to base model)."""
        import torch

        AutoModelForSequenceClassification, AutoTokenizer, _, _ = _import_transformers()

        model_path = str(self._model_dir) if self._model_dir.exists() else BASE_MODEL

        if not self._model_dir.exists():
            if self._fallback:
                log.warning(
                    "Fine-tuned model not found at %s. "
                    "Loading base model (predictions will be random).",
                    self._model_dir,
                )
                model_path = BASE_MODEL
            else:
                raise FileNotFoundError(
                    f"Fine-tuned model not found at {self._model_dir}. "
                    "Run training first: python -m src.transformer_model --train"
                )

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = self._model.to(self._device)
        self._model.eval()
        log.debug("TransformerModel loaded on device: %s", self._device)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(
        self,
        subject: str = "",
        body: str = "",
    ) -> dict[str, Any]:
        """
        Classify a single email using the fine-tuned DistilBERT model.

        Parameters
        ----------
        subject:
            Email subject line.
        body:
            Email body text.

        Returns
        -------
        dict with keys ``label``, ``confidence``, ``model``.
        """
        import torch

        if not self.is_loaded:
            self._load()

        text = f"{subject} {body}".strip()
        if not text:
            return {"label": "safe", "confidence": 0.01, "model": "distilbert"}

        encoding = self._tokenizer(
            text,
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self._device)
        attention_mask = encoding["attention_mask"].to(self._device)

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()

        probs = _softmax(logits)[0]
        phishing_conf = float(probs[LABEL2ID["phishing"]])
        label = "phishing" if phishing_conf >= 0.50 else "safe"

        return {
            "label": label,
            "confidence": round(phishing_conf, 4),
            "model": "distilbert-phishing",
        }

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        """
        Classify a list of text strings efficiently.

        Parameters
        ----------
        texts:
            List of ``"{subject} {body}"`` strings.
        batch_size:
            Number of sequences processed in one forward pass.

        Returns
        -------
        List of prediction dicts (same format as :meth:`predict`).
        """
        import torch

        if not self.is_loaded:
            self._load()

        results: list[dict[str, Any]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoding = self._tokenizer(
                batch,
                max_length=self._max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(self._device)
            attention_mask = encoding["attention_mask"].to(self._device)

            with torch.no_grad():
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.cpu().numpy()

            probs = _softmax(logits)
            for prob_row in probs:
                phishing_conf = float(prob_row[LABEL2ID["phishing"]])
                results.append({
                    "label": "phishing" if phishing_conf >= 0.50 else "safe",
                    "confidence": round(phishing_conf, 4),
                    "model": "distilbert-phishing",
                })

        return results


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_model: TransformerModel | None = None


def get_model() -> TransformerModel:
    """Return the module-level :class:`TransformerModel` singleton."""
    global _model
    if _model is None:
        _model = TransformerModel()
    return _model


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune or run DistilBERT phishing classifier")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap training data for CPU runs (e.g. 10000)")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH,
                        help="Token sequence length (default 256; use 64 for fast CPU runs)")
    args = parser.parse_args()

    if args.train:
        results = train(
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            max_samples=args.max_samples,
            max_length=args.max_length,
        )
        print("\n=== Fine-tuning Complete ===")
        print(f"Model saved to: {results['output_dir']}")
        for k, v in results["eval_metrics"].items():
            print(f"  {k}: {v}")
