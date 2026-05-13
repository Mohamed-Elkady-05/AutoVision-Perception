"""Plot confusion matrices from results/*_metrics.json files.

Usage: python scripts/plot_confusion_matrices.py
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_matrix(matrix, output_path: Path, title: str):
    matrix = np.array(matrix)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    results_dir = Path("results")
    for metrics_file in results_dir.glob("*_metrics.json"):
        data = json.load(open(metrics_file))
        if "confusion_matrix" not in data:
            print(f"No confusion matrix in {metrics_file}")
            continue
        name = metrics_file.stem.replace("_metrics", "")
        out_png = results_dir / f"{name}_confusion_matrix.png"
        print(f"Plotting {metrics_file} -> {out_png}")
        plot_matrix(data["confusion_matrix"], out_png, title=f"{name} Confusion Matrix")


if __name__ == "__main__":
    main()
