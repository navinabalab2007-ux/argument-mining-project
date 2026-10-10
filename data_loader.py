"""Loading, validating and splitting the Feedback Prize dataset (CSV files)."""
from pathlib import Path

import pandas as pd

import config
from utils import fingerprint, get_logger

log = get_logger()

REQUIRED_COLUMNS = ["discourse_id", "essay_id", "discourse_text", "discourse_type",
                    "discourse_effectiveness"]


class DataLoaderError(ValueError):
    """Raised with a beginner-friendly message when the dataset cannot be used."""


def find_train_csv(directory):
    directory = Path(directory)
    if directory.is_file() and directory.suffix.lower() == ".csv":
        return directory
    for name in ("train.csv", "sample_train.csv"):
        p = directory / name
        if p.exists():
            return p
    csvs = sorted(directory.glob("*.csv")) if directory.is_dir() else []
    if csvs:
        return csvs[0]
    raise DataLoaderError(
        f"No CSV file found in '{directory}'. Download the Feedback Prize - Predicting Effective "
        f"Arguments data from Kaggle and place train.csv in that folder, or use the built-in "
        f"sample dataset.")


def load_dataset_csv(path, max_rows=None):
    """Load and clean a Feedback Prize style CSV. Returns (DataFrame, report dict)."""
    path = Path(path)
    if not path.exists():
        raise DataLoaderError(f"File not found: {path}")
    try:
        df = pd.read_csv(path, nrows=max_rows)
    except Exception as exc:  # unreadable / malformed CSV
        raise DataLoaderError(f"Could not read '{path}': {exc}") from exc
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataLoaderError(
            f"The CSV is missing required columns: {missing}. Found columns: {list(df.columns)}")
    report = {"file": str(path), "rows_in": int(len(df))}
    df = df[REQUIRED_COLUMNS].copy()
    df = df.dropna(subset=["discourse_text", "discourse_type", "discourse_effectiveness"])
    for col in ("discourse_id", "essay_id", "discourse_text", "discourse_type",
                "discourse_effectiveness"):
        df[col] = df[col].astype(str).str.strip()
    df = df[df["discourse_text"].str.len() > 0]
    valid_type = df["discourse_type"].isin(config.DISCOURSE_TYPES)
    valid_eff = df["discourse_effectiveness"].isin(config.EFFECTIVENESS_LABELS)
    report["dropped_invalid_labels"] = int((~(valid_type & valid_eff)).sum())
    df = df[valid_type & valid_eff]
    before = len(df)
    df = df.drop_duplicates(subset=["discourse_id"])
    report["dropped_duplicates"] = int(before - len(df))
    df = df.reset_index(drop=True)
    report["rows_out"] = int(len(df))
    if df.empty:
        raise DataLoaderError("No valid rows remain after cleaning the dataset.")
    return df, report


def get_dataset(source="sample", directory=None, max_rows=None):
    """source: 'sample' (built-in demo CSV) or 'feedback-prize' (your download).

    Returns (DataFrame, report). report['source'] tells which one was used.
    """
    if source == "sample":
        df, rep = load_dataset_csv(config.SAMPLE_CSV, max_rows)
        rep["source"] = "sample"
        return df, rep
    directory = Path(directory) if directory else config.FEEDBACK_PRIZE_DIR
    csv_path = find_train_csv(directory)
    df, rep = load_dataset_csv(csv_path, max_rows)
    rep["source"] = "feedback-prize"
    return df, rep


def split_dataset(df, test_size=0.2, val_size=0.1, seed=config.SEED):
    """Essay-grouped train/val/test split (no essay appears in two splits)."""
    import numpy as np
    rng = np.random.RandomState(seed)
    essays = np.array(sorted(df["essay_id"].unique()))
    rng.shuffle(essays)
    n = len(essays)
    n_test = max(1, int(round(n * test_size))) if n >= 3 else 0
    n_val = max(1, int(round(n * val_size))) if n >= 6 else 0
    test_e = set(essays[:n_test])
    val_e = set(essays[n_test:n_test + n_val])
    test = df[df["essay_id"].isin(test_e)]
    val = df[df["essay_id"].isin(val_e)]
    train = df[~df["essay_id"].isin(test_e | val_e)]
    return (train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))


def subsample(df, n, seed=config.SEED):
    """Random subsample of n rows (roughly balanced over discourse types)."""
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    per = max(1, n // max(1, df["discourse_type"].nunique()))
    parts = [g.sample(min(len(g), per), random_state=seed) for _, g in df.groupby("discourse_type")]
    out = pd.concat(parts)
    if len(out) < n:
        rest = df.drop(out.index)
        out = pd.concat([out, rest.sample(min(len(rest), n - len(out)), random_state=seed)])
    return out.sample(frac=1.0, random_state=seed).head(n).reset_index(drop=True)


def dataset_fingerprint(df):
    return fingerprint(list(df["discourse_id"]) + list(df["discourse_text"]))
