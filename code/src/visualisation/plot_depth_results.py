from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_rmse(df):
    plt.figure(figsize=(7.5, 5))

    for (model, attack), group in df.groupby(["model", "attack"]):
        group = group.sort_values("epsilon")

        plt.plot(
            group["epsilon"],
            group["rmse"],
            marker="o",
            linewidth=2,
            label=f"{model} - {attack.upper()}"
        )

    plt.xlabel(r"$\epsilon$")
    plt.ylabel("RMSE")

    plt.title(
        "Root Mean Square Error"
    )

    # caption: mIoU degradation under FGSM and PGD attacks with increasing L_\infty perturbation magnitude.

    plt.legend(
        fontsize=9
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_abs_rel(df):
    plt.figure(figsize=(7.5, 5))

    for (model, attack), group in df.groupby(["model", "attack"]):
        group = group.sort_values("epsilon")

        plt.plot(
            group["epsilon"],
            group["abs_rel"],
            marker="o",
            linewidth=2,
            label=f"{model} - {attack.upper()}"
        )

    plt.xlabel(r"$\epsilon$")
    plt.ylabel("AbsRel")

    plt.title(
        "Absolute Relative Error"
    )

    # caption: mIoU degradation under FGSM and PGD attacks with increasing L_\infty perturbation magnitude.

    plt.legend(
        fontsize=9
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()


project_root = Path(__file__).resolve().parents[2]

result_path = project_root / "output/depth_results/depth_results.csv"

# load results
df = pd.read_csv(result_path)

# rename models for presentation
df["model"] = df["model"].replace({
    "dinov2-vitl14": "DINOv2 ViT-L/14",
    "dinov3-vitl16": "DINOv3 ViT-L/16"
})

plot_rmse(df)
plot_abs_rel(df)
