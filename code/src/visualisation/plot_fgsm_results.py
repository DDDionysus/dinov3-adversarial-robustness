from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import ast
from pathlib import Path


def extract_metrics(df):

    for model in ["dinov2_cls", "dinov3_cls"]:
        metrics = df[model].apply(ast.literal_eval)

        df[f"{model}_cls_similarity"] = metrics.apply(
            lambda x: x["cls_similarity"]
        )

        df[f"{model}_patch_mean"] = metrics.apply(
            lambda x: x["patch_similarity_mean"]
        )

        df[f"{model}_patch_std"] = metrics.apply(
            lambda x: x["patch_similarity_std"]
        )

    return df


def plot_results(csv_path, attack):

    df = pd.read_csv(csv_path)

    df = extract_metrics(df)

    # Average across all images for each epsilon
    grouped = (
        df.groupby("epsilon")
        .agg({
            "dinov2_cls_cls_similarity": "mean",
            "dinov3_cls_cls_similarity": "mean",

            "dinov2_cls_patch_mean": "mean",
            "dinov3_cls_patch_mean": "mean",

            "dinov2_cls_patch_std": "mean",
            "dinov3_cls_patch_std": "mean",
        })
        .reset_index()
    )

    # -------------------------
    # CLS similarity
    # -------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        grouped["epsilon"],
        grouped["dinov2_cls_cls_similarity"],
        marker="o",
        label="DINOv2"
    )

    plt.plot(
        grouped["epsilon"],
        grouped["dinov3_cls_cls_similarity"],
        marker="o",
        label="DINOv3"
    )

    # plt.ylim(0, 1.05)

    plt.xlabel(f"{attack} epsilon")
    plt.ylabel("CLS cosine similarity")
    plt.title(f"CLS Representation Robustness under {attack}")

    plt.legend()
    plt.grid(True)

    """plt.savefig(
        OUTPUT_DIR / "cls_similarity.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()"""

    plt.show()

    # -------------------------
    # Patch similarity
    # -------------------------

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        grouped["epsilon"],
        grouped["dinov2_cls_patch_mean"],
        yerr=grouped["dinov2_cls_patch_std"],
        marker="o",
        label="DINOv2",
        capsize=4
    )

    plt.errorbar(
        grouped["epsilon"],
        grouped["dinov3_cls_patch_mean"],
        yerr=grouped["dinov3_cls_patch_std"],
        marker="o",
        label="DINOv3",
        capsize=4
    )

    # plt.ylim(0, 1.05)

    plt.xlabel(f"{attack} epsilon")
    plt.ylabel("Patch cosine similarity")
    plt.title(f"Patch Representation Robustness under {attack}")

    plt.legend()
    plt.grid(True)

    """plt.savefig(
        OUTPUT_DIR / "patch_similarity.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()"""

    plt.show()


if __name__ == "__main__":
    attack = "smaller_test"

    project_root = Path(__file__).resolve().parents[2]

    csv_path = project_root / f"output/results/{attack}_imagenet_L_subset.csv"

    plot_results(csv_path, attack)
