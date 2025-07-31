#!/usr/bin/env python3
"""
scripts/plot_analysis.py

Analysis script to generate all key plots from experiment outputs:
- Accuracy vs. Epoch & Loss vs. Epoch for each training log
- Confusion Matrix heatmaps for each model
- Accuracy vs. Learning Rate (if lr_sweep.json exists)
- Final Accuracy Comparison (if final_results.json exists)
"""
import os
import os, sys
# add project root to PYTHONPATH so that local modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import glob
import json
import numpy as np

from utils.plotting import (
    plot_metrics,
    plot_confusion_matrix,
    plot_accuracy_vs_lr,
    plot_final_comparison
)


def main():
    # Directories
    logs_dir = "logs"
    results_dir = "results"
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Plot metrics for each training log
    for csv_file in glob.glob(os.path.join(logs_dir, "*.csv")):
        exp_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_subdir = os.path.join(plots_dir, exp_name)
        acc_path, loss_path = plot_metrics(csv_file, output_subdir)
        print(f"Generated metrics plots for {exp_name}:\n  {acc_path}\n  {loss_path}")

    # 2. Plot confusion matrices for each model
    for cm_file in glob.glob(os.path.join(results_dir, "*_confusion_matrix.npy")):
        model_name = os.path.basename(cm_file).replace("_confusion_matrix.npy", "")
        cm = np.load(cm_file)
        out_path = os.path.join(plots_dir, f"{model_name}_confusion_matrix.png")
        plot_confusion_matrix(cm, out_path, normalize=True)
        print(f"Saved confusion matrix plot for {model_name}: {out_path}")

    # 3. Accuracy vs Learning Rate (optional)
    lr_json = os.path.join(results_dir, "lr_sweep.json")
    if os.path.exists(lr_json):
        with open(lr_json, 'r') as f:
            lr_data = json.load(f)
        lr_values = lr_data.get('lr_values', [])
        accuracies = lr_data.get('accuracies', {})  # {model_name: [acc_values]}
        out_lr = os.path.join(plots_dir, "accuracy_vs_lr.png")
        plot_accuracy_vs_lr(lr_values, accuracies, out_lr)
        print(f"Generated Accuracy vs LR plot: {out_lr}")

    # 4. Final accuracy comparison (optional)
    final_json = os.path.join(results_dir, "final_results.json")
    if os.path.exists(final_json):
        with open(final_json, 'r') as f:
            final_data = json.load(f)
        models = list(final_data.keys())
        baseline_acc = [final_data[m]['baseline'] for m in models]
        optimized_acc = [final_data[m]['optimized'] for m in models]
        out_comp = os.path.join(plots_dir, "final_accuracy_comparison.png")
        plot_final_comparison(models, baseline_acc, optimized_acc, out_comp)
        print(f"Generated final accuracy comparison plot: {out_comp}")

    print("All requested plots generated.")

if __name__ == '__main__':
    main()
