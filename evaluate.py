"""Evaluation: accuracy, precision, recall, F1, confusion matrices, saved reports, run comparison.

Only MEASURED results are stored and shown. Predictions are compared with the dataset's
ground-truth labels (discourse_type / discourse_effectiveness).
"""
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

import config
from data_loader import DataLoaderError, dataset_fingerprint, get_dataset, split_dataset, subsample
from utils import ensure_dirs, get_logger, load_json, save_json, timestamp

log = get_logger()


def _task_metrics(y_true, y_pred, base_labels):
    labels = [l for l in base_labels if l in set(y_true) | set(y_pred)]
    for extra in sorted((set(y_true) | set(y_pred)) - set(labels)):
        labels.append(extra)
    acc = accuracy_score(y_true, y_pred)
    out = {"labels": labels, "accuracy": float(acc), "n": len(y_true)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        out[f"precision_{avg}"], out[f"recall_{avg}"], out[f"f1_{avg}"] = float(p), float(r), float(f)
    out["per_class"] = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return out


def _plot_confusion(cm, labels, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(labels) + 2), max(4, 0.8 * len(labels) + 1.5)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth"); ax.set_title(title)
    thresh = cm.max() / 2 if cm.size and cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def evaluate_predictor(predictor, df, run_label, dataset_source, split_name="test",
                       progress_cb=None, output_root=None):
    """Evaluate any object with classify_texts(list[str]) -> list[{'type','effectiveness'}]."""
    if df is None or df.empty:
        raise ValueError("The evaluation set is empty.")
    ensure_dirs()
    preds = predictor.classify_texts(list(df["discourse_text"]), progress_cb=progress_cb)
    p_type = [p["type"] if p["type"] in config.SEGMENT_TYPES else "Other" for p in preds]
    p_eff = [p["effectiveness"] if p["effectiveness"] in config.EFFECTIVENESS_LABELS
             else config.NOT_APPLICABLE for p in preds]
    y_type, y_eff = list(df["discourse_type"]), list(df["discourse_effectiveness"])
    type_m = _task_metrics(y_type, p_type, config.SEGMENT_TYPES)
    eff_m = _task_metrics(y_eff, p_eff, config.EFFECTIVENESS_LABELS + [config.NOT_APPLICABLE])
    run_id = f"{run_label}_{dataset_source}_{timestamp()}"
    out_dir = (output_root or config.EVAL_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({"discourse_id": df["discourse_id"].values, "text": df["discourse_text"].values,
                            "true_type": y_type, "pred_type": p_type,
                            "true_effectiveness": y_eff, "pred_effectiveness": p_eff})
    pred_df["type_correct"] = pred_df["true_type"] == pred_df["pred_type"]
    pred_df["effectiveness_correct"] = pred_df["true_effectiveness"] == pred_df["pred_effectiveness"]
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    _plot_confusion(type_m["confusion_matrix"], type_m["labels"], f"Argument type - {run_label}", out_dir / "confusion_type.png")
    _plot_confusion(eff_m["confusion_matrix"], eff_m["labels"], f"Effectiveness - {run_label}", out_dir / "confusion_effectiveness.png")
    result = {"run_id": run_id, "run_label": run_label,
              "engine_kind": getattr(predictor, "kind", "unknown"),
              "model": getattr(predictor, "name", str(predictor)),
              "dataset_source": dataset_source, "split": split_name, "n_samples": int(len(df)),
              "test_fingerprint": dataset_fingerprint(df), "created": timestamp(),
              "type": type_m, "effectiveness": eff_m, "output_dir": str(out_dir),
              "note": ("Sample dataset is tiny and hand-written; scores only prove the pipeline works."
                       if dataset_source == "sample" else "Measured on the Feedback Prize CSV provided.")}
    save_json(result, out_dir / "metrics.json")
    lines = [f"Evaluation report: {run_id}", f"Model: {result['model']}",
             f"Dataset: {dataset_source} / split={split_name} / n={len(df)}", ""]
    for title, m in (("ARGUMENT TYPE", type_m), ("EFFECTIVENESS", eff_m)):
        lines += [title, f"  accuracy        : {m['accuracy']:.4f}",
                  f"  precision (macro): {m['precision_macro']:.4f}", f"  recall (macro)   : {m['recall_macro']:.4f}",
                  f"  F1 (macro)       : {m['f1_macro']:.4f}", f"  F1 (weighted)    : {m['f1_weighted']:.4f}", ""]
    (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    return result


def load_runs(root=None):
    root = root or config.EVAL_DIR
    runs = []
    for p in sorted(root.glob("*/metrics.json")):
        r = load_json(p)
        if r:
            runs.append(r)
    return runs


def comparison_table(runs=None):
    """DataFrame of all saved (measured) runs. 'same_test_set' compares test-set fingerprints."""
    runs = runs if runs is not None else load_runs()
    rows = []
    for r in runs:
        rows.append({"run": r["run_id"], "engine": r["engine_kind"], "model": r["model"],
                     "dataset": r["dataset_source"], "split": r["split"], "n": r["n_samples"],
                     "type_accuracy": round(r["type"]["accuracy"], 4), "type_macro_F1": round(r["type"]["f1_macro"], 4),
                     "eff_accuracy": round(r["effectiveness"]["accuracy"], 4),
                     "eff_macro_F1": round(r["effectiveness"]["f1_macro"], 4),
                     "test_set_id": r["test_fingerprint"]})
    return pd.DataFrame(rows)


def get_eval_frame(source, directory, split, max_samples, seed=config.SEED):
    df, rep = get_dataset(source, directory)
    if split == "all":
        eval_df = df
    else:
        _, _, eval_df = split_dataset(df, seed=seed)
        if eval_df.empty:
            eval_df = df
    if max_samples:
        eval_df = subsample(eval_df, max_samples, seed)
    return eval_df, rep


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate an engine against ground-truth labels.")
    ap.add_argument("--engine", choices=["heuristic", "pretrained", "finetuned", "bert"], default="heuristic")
    ap.add_argument("--dataset", choices=["sample", "feedback-prize"], default="sample")
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--split", choices=["test", "all"], default="all")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--model-id", default=config.DEFAULT_MODEL_ID)
    ap.add_argument("--quant", choices=["auto", "4bit", "none"], default="auto")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    ensure_dirs()
    try:
        df, rep = get_eval_frame(args.dataset, args.dataset_dir, args.split, args.max_samples)
        if args.engine == "bert":
            from baseline import BertBaseline
            predictor = BertBaseline.load()
        else:
            from model import create_engine
            from predict import ArgumentAnalyzer
            predictor = ArgumentAnalyzer(create_engine(args.engine, args.model_id, None, args.quant, None, args.force))
        res = evaluate_predictor(predictor, df, args.engine, args.dataset, args.split,
                                 progress_cb=lambda f, m: print(f"\r{m}", end="", flush=True))
        print()
    except (DataLoaderError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"\nERROR: {exc}")
        return 1
    print(open(f"{res['output_dir']}/report.txt", encoding="utf-8").read())
    print("Saved to", res["output_dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
