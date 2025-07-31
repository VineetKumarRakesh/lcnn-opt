#!/usr/bin/env python3
# scripts/generate_report.py

import os
import glob
import json
import pandas as pd

def extract_best_val(csv_path):
    """Read the last validation accuracy from a CSV log."""
    df = pd.read_csv(csv_path)
    return float(df['val_acc'].iloc[-1])

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 1) Learning-rate sweep (logs named lr_<lr>_<Model>.csv)
    lr_data = {}
    lr_values = set()
    for path in glob.glob("logs/lr_*.csv"):
        fname = os.path.basename(path).replace(".csv", "")
        _, lr_str, model = fname.split("_", 2)
        lr = float(lr_str)
        acc = extract_best_val(path)
        lr_values.add(lr)
        lr_data.setdefault(model, {})[lr] = acc

    lr_values = sorted(lr_values)
    lr_sweep = {
        "lr_values": lr_values,
        "accuracies": {
            model: [lr_data[model].get(lr, None) for lr in lr_values]
            for model in lr_data
        }
    }
    with open("results/lr_sweep.json", "w") as f:
        json.dump(lr_sweep, f, indent=2)

    # 2) Augmentation sweep (logs named aug_<tag>_<Model>.csv)
    aug_data = {}
    aug_tags = set()
    for path in glob.glob("logs/aug_*.csv"):
        fname = os.path.basename(path).replace(".csv", "")
        _, tag, model = fname.split("_", 2)
        acc = extract_best_val(path)
        aug_tags.add(tag)
        aug_data.setdefault(model, {})[tag] = acc

    aug_tags = sorted(aug_tags)
    aug_sweep = {
        "augmentations": aug_tags,
        "accuracies": {
            model: [aug_data[model].get(tag, None) for tag in aug_tags]
            for model in aug_data
        }
    }
    with open("results/aug_sweep.json", "w") as f:
        json.dump(aug_sweep, f, indent=2)

    # 3) Final results from per-model run CSVs *and* eval JSONs
    final = {}
    # First, load each *_run.csv as the “optimized” setting
    for path in glob.glob("logs/*_run.csv"):
        fname = os.path.basename(path).replace(".csv", "")
        model = fname.rsplit("_run", 1)[0]
        acc = extract_best_val(path)
        final.setdefault(model, {})["optimized"] = acc

    # Then, load the eval JSON (for baseline / SOTA / epochs / gpu_hours metadata)
    for path in glob.glob("results/*_eval.json"):
        with open(path) as f:
            data = json.load(f)
        model = data["model"]
        entry = final.setdefault(model, {})
        entry["baseline"]  = data.get("baseline", data["top1"])
        entry["sota"]      = data.get("sota", None)
        entry["epochs"]    = data.get("epochs", None)
        entry["gpu_hours"] = data.get("gpu_hours", None)

    with open("results/final_results.json", "w") as f:
        json.dump(final, f, indent=2)

    # 4) (Optional) Emit LaTeX tables for your paper
    def make_table(df, caption, label, out_path):
        tex = df.to_latex(
            index=False,
            caption=caption,
            label=label,
            float_format="%.1f"
        )
        with open(out_path, "w") as f:
            f.write(tex)

    # Table 2: LR ablation
    df_lr = pd.DataFrame({"Learning Rate": lr_sweep["lr_values"], **lr_sweep["accuracies"]})
    make_table(
        df_lr,
        "Top-1 accuracy (\\%) for different initial learning rates.",
        "tab:ablation_lr",
        "reports/table2_lr_ablation.tex"
    )

    # Table 3: Aug ablation
    df_aug = pd.DataFrame({"Augmentation": aug_sweep["augmentations"], **aug_sweep["accuracies"]})
    make_table(
        df_aug,
        "Top-1 accuracy (\\%) for different data augmentation strategies.",
        "tab:ablation_aug",
        "reports/table3_aug_ablation.tex"
    )

    # Table 4: Final results
    df_final = pd.DataFrame([
        {"Model": m, **final[m]} for m in sorted(final)
    ])
    make_table(
        df_final,
        "Top-1 accuracy (\\%), training epochs, and approximate GPU hours for each model under baseline, optimized, and reported SOTA settings.",
        "tab:final_results",
        "reports/table4_final_results.tex"
    )

    print("Generated JSON and LaTeX tables in results/ and reports/")

if __name__ == "__main__":
    main()
