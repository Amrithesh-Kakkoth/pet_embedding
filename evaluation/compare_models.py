import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""Compare evaluation results across pet face embedding models.

Usage:
    python compare_models.py \
        --models "MobileNetV3-small:eval_cat" "MobileFaceNet:eval_cat_mfn" \
        --output comparison.txt

    python compare_models.py \
        --models "MNv3-cat:eval_cat" "MFN-cat:eval_cat_mfn" \
            "MNv3-dog:eval_dog" "MFN-dog:eval_dog_mfn"
"""

import argparse
import json
import os


def load_eval_results(eval_dir: Path) -> dict:
    """Load eval_results.json from an evaluation directory."""
    path = eval_dir / "eval_results.json"
    if not path.exists():
        raise FileNotFoundError(f"No eval_results.json in {eval_dir}")
    with open(path) as f:
        return json.load(f)


def load_model_info(eval_dir: Path) -> dict:
    """Load config.json and compute model size info."""
    info = {}

    # Try to find config.json from the model output dir
    # eval_dir might be eval_cat, model dir is outputs_cat
    config_path = eval_dir / "config.json"
    if not config_path.exists():
        # Try parent-level output dirs with similar naming
        parent = eval_dir.parent
        name = eval_dir.name
        # eval_cat -> outputs_cat, eval_dog_mfn -> outputs_dog_mfn
        candidate = parent / name.replace("eval_", "outputs_")
        config_path = candidate / "config.json"

    if config_path.exists():
        with open(config_path) as f:
            info["config"] = json.load(f)

    # Check for ONNX file size
    for search_dir in [eval_dir, eval_dir.parent / eval_dir.name.replace("eval_", "outputs_")]:
        onnx_path = search_dir / "pet_embedder.onnx"
        if onnx_path.exists():
            info["onnx_size_mb"] = os.path.getsize(onnx_path) / (1024 * 1024)
            break

    return info


def fmt(val, fmt_str=".4f"):
    """Format a value, handling missing/nan."""
    if val is None:
        return "N/A"
    if isinstance(val, str):
        return val
    try:
        return f"{val:{fmt_str}}"
    except (ValueError, TypeError):
        return str(val)


def pct(val):
    """Format as percentage."""
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def print_comparison_table(models: dict[str, dict], model_info: dict[str, dict]):
    """Print a formatted comparison table."""
    names = list(models.keys())
    col_width = max(20, max(len(n) for n in names) + 2)

    def header():
        h = f"{'Metric':<35}"
        for name in names:
            h += f"{name:>{col_width}}"
        return h

    def row(label, values, fmt_fn=fmt):
        r = f"{label:<35}"
        for v in values:
            r += f"{fmt_fn(v):>{col_width}}"
        return r

    def separator():
        return "-" * (35 + col_width * len(names))

    print("\n" + "=" * (35 + col_width * len(names)))
    print(f"{'PET FACE EMBEDDING MODEL COMPARISON':^{35 + col_width * len(names)}}")
    print("=" * (35 + col_width * len(names)))

    # --- Model Info ---
    print(f"\n{header()}")
    print(separator())

    # Backbone
    vals = []
    for name in names:
        info = model_info.get(name, {})
        cfg = info.get("config", {})
        vals.append(cfg.get("backbone", "unknown"))
    print(row("Backbone", vals, fmt_fn=str))

    # ONNX size
    vals = [model_info.get(n, {}).get("onnx_size_mb") for n in names]
    print(row("ONNX size (MB)", vals, fmt_fn=lambda v: fmt(v, ".1f")))

    # --- Retrieval ---
    print(f"\n{'RETRIEVAL':^{35 + col_width * len(names)}}")
    print(separator())
    print(header())
    print(separator())

    for metric in ["recall@1", "recall@5", "recall@10", "MRR", "NDCG@10"]:
        vals = [models[n].get("retrieval", {}).get(metric) for n in names]
        print(row(metric, vals, fmt_fn=pct if "recall" in metric else fmt))

    # --- Clustering ---
    print(f"\n{'CLUSTERING':^{35 + col_width * len(names)}}")
    print(separator())
    print(header())
    print(separator())

    for metric in ["NMI", "ARI", "purity"]:
        vals = [models[n].get("clustering", {}).get(metric) for n in names]
        print(row(metric, vals))

    # --- Ranking ---
    print(f"\n{'RANKING':^{35 + col_width * len(names)}}")
    print(separator())
    print(header())
    print(separator())

    for metric in ["ROC-AUC", "Spearman", "EER"]:
        vals = [models[n].get("ranking", {}).get(metric) for n in names]
        print(row(metric, vals))

    # --- Triplet ---
    print(f"\n{'TRIPLET':^{35 + col_width * len(names)}}")
    print(separator())
    print(header())
    print(separator())

    for metric in ["triplet_violation_rate", "average_margin"]:
        vals = [models[n].get("triplet", {}).get(metric) for n in names]
        label = "Violation rate" if "violation" in metric else "Avg margin"
        print(row(label, vals, fmt_fn=pct if "violation" in metric else fmt))

    # --- Overfitting ---
    print(f"\n{'OVERFITTING CHECK':^{35 + col_width * len(names)}}")
    print(separator())
    print(header())
    print(separator())

    for metric in ["train_recall@1", "test_recall@1", "gap"]:
        vals = [models[n].get("overfitting", {}).get(metric) for n in names]
        label = {"train_recall@1": "Train recall@1",
                 "test_recall@1": "Test recall@1",
                 "gap": "Gap (overfit indicator)"}[metric]
        print(row(label, vals, fmt_fn=pct))

    # --- Robustness ---
    print(f"\n{'ROBUSTNESS (similarity drop)':^{35 + col_width * len(names)}}")
    print(separator())
    print(header())
    print(separator())

    degradations = ["gaussian_blur", "gaussian_noise", "jpeg_compression",
                     "brightness_shift", "partial_occlusion"]
    for deg in degradations:
        vals = [models[n].get("robustness", {}).get(deg, {}).get("similarity_drop")
                for n in names]
        print(row(deg.replace("_", " ").title(), vals,
                  fmt_fn=lambda v: fmt(v, ".4f")))

    print(f"\n{'ROBUSTNESS (recall@1 drop)':^{35 + col_width * len(names)}}")
    print(separator())
    print(header())
    print(separator())

    for deg in degradations:
        vals = [models[n].get("robustness", {}).get(deg, {}).get("recall_drop")
                for n in names]
        print(row(deg.replace("_", " ").title(), vals,
                  fmt_fn=lambda v: pct(v) if v is not None else "N/A"))

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare pet face embedding model evaluations")
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Model entries as 'Name:eval_dir' (e.g. 'MobileNetV3:eval_cat')")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional file to save comparison output")
    args = parser.parse_args()

    models = {}
    model_info = {}
    for entry in args.models:
        if ":" not in entry:
            print(f"Error: expected 'Name:eval_dir', got '{entry}'")
            return
        name, eval_dir = entry.split(":", 1)
        eval_path = Path(eval_dir)
        try:
            models[name] = load_eval_results(eval_path)
            model_info[name] = load_model_info(eval_path)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            models[name] = {}
            model_info[name] = {}

    if args.output:
        import sys
        original_stdout = sys.stdout
        with open(args.output, "w") as f:
            sys.stdout = f
            print_comparison_table(models, model_info)
            sys.stdout = original_stdout
        print(f"Comparison saved to {args.output}")
        # Also print to screen
        print_comparison_table(models, model_info)
    else:
        print_comparison_table(models, model_info)


if __name__ == "__main__":
    main()
