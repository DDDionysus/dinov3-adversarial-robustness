from pathlib import Path
import ast

import pandas as pd
import matplotlib.pyplot as plt


def output_inspection():
    """
    Validate the completeness of chunked PGD results produced on HPC.
    Check the expected number of chunk files, unique images, total rows, missing
    values, and duplicated image-epsilon pairs.
    """

    project_root = Path(__file__).resolve().parents[2]

    results_dir = project_root / "output" / "hpc_results/pgd"

    files = sorted(results_dir.glob("pgd_imagenet_L_chunk_*.csv"))

    print("Number of chunk files:", len(files))

    if not files:
        raise FileNotFoundError(f"No PGD chunk files found in {results_dir}")

    dataframes = []

    for file in files:
        df = pd.read_csv(file)

        print(f"{file.name}: {len(df)} rows")

        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)

    print("\n===== Combined dataset =====")
    print("Total rows:", len(df))

    print("Unique images:", df["image"].nunique())

    print("Epsilon values:")
    print(sorted(df["epsilon"].unique()))

    print("\nRows per epsilon:")
    print(df["epsilon"].value_counts().sort_index())

    print("\nMissing values:")
    print(df.isna().sum())

    duplicates = df.duplicated(
        subset=["image", "epsilon"],
        keep=False,
    )

    print("\nDuplicate (image, epsilon) rows:", duplicates.sum())

    expected_rows = 50000 * 5

    print("\n===== Expected values =====")
    print("Expected chunk files: 50")
    print("Expected unique images: 50000")
    print("Expected total rows:", expected_rows)

    if len(files) == 50:
        print("Chunk count: PASS")
    else:
        print("Chunk count: FAIL")

    if df["image"].nunique() == 50000:
        print("Unique image count: PASS")
    else:
        print("Unique image count: FAIL")

    if len(df) == expected_rows:
        print("Total row count: PASS")
    else:
        print("Total row count: FAIL")

    if duplicates.sum() == 0:
        print("Duplicates: PASS")
    else:
        print("Duplicates: FAIL")


def parsing_results():
    """
    Combine chunked outputs and compute dataset-level summary metrics.
    """

    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "output" / "hpc_results/pgd_nyu_hpc"

    files = sorted(results_dir.glob("pgd_nyu_L_chunk_*.csv"))

    dataframes = [pd.read_csv(file) for file in files]
    df = pd.concat(dataframes, ignore_index=True)

    # Parse dictionary columns
    df["dinov2_metrics"] = df["dinov2_cls"].apply(ast.literal_eval)
    df["dinov3_metrics"] = df["dinov3_cls"].apply(ast.literal_eval)

    # Extract individual metrics
    df["dinov2_cls_similarity"] = df["dinov2_metrics"].apply(
        lambda x: x["cls_similarity"]
    )
    df["dinov2_patch_mean"] = df["dinov2_metrics"].apply(
        lambda x: x["patch_similarity_mean"]
    )
    df["dinov2_patch_std"] = df["dinov2_metrics"].apply(
        lambda x: x["patch_similarity_std"]
    )

    df["dinov3_cls_similarity"] = df["dinov3_metrics"].apply(
        lambda x: x["cls_similarity"]
    )
    df["dinov3_patch_mean"] = df["dinov3_metrics"].apply(
        lambda x: x["patch_similarity_mean"]
    )
    df["dinov3_patch_std"] = df["dinov3_metrics"].apply(
        lambda x: x["patch_similarity_std"]
    )

    # Aggregate by epsilon
    summary = df.groupby("epsilon").agg(
        dinov2_cls_mean=("dinov2_cls_similarity", "mean"),
        dinov2_cls_std=("dinov2_cls_similarity", "std"),
        dinov3_cls_mean=("dinov3_cls_similarity", "mean"),
        dinov3_cls_std=("dinov3_cls_similarity", "std"),

        dinov2_patch_mean=("dinov2_patch_mean", "mean"),
        dinov2_patch_mean_std=("dinov2_patch_mean", "std"),
        dinov3_patch_mean=("dinov3_patch_mean", "mean"),
        dinov3_patch_mean_std=("dinov3_patch_mean", "std"),

        dinov2_patch_std_mean=("dinov2_patch_std", "mean"),
        dinov3_patch_std_mean=("dinov3_patch_std", "mean"),
    ).reset_index()

    print(summary.to_string(index=False))

    output_path = results_dir / "pgd_imagenet_L_summary.csv"
    summary.to_csv(output_path, index=False)

    print(f"\nSaved summary to: {output_path}")


def plotting_results():
    """
    Plot dataset-level CLS-token and patch-token robustness under PGD.
    """

    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "output" / "hpc_results/pgd"

    summary_path = results_dir / "pgd_imagenet_L_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Summary file not found: {summary_path}"
        )

    summary = pd.read_csv(summary_path)

    # CLS-token similarity plot
    plt.figure(figsize=(6, 4))

    plt.plot(
        summary["epsilon"],
        summary["dinov2_cls_mean"],
        marker="o",
        label="DINOv2 ViT-L/14",
    )

    plt.fill_between(
        summary["epsilon"],
        summary["dinov2_cls_mean"] - summary["dinov2_cls_std"],
        summary["dinov2_cls_mean"] + summary["dinov2_cls_std"],
        alpha=0.15,
    )

    plt.plot(
        summary["epsilon"],
        summary["dinov3_cls_mean"],
        marker="o",
        label="DINOv3 ViT-L/16",
    )

    plt.fill_between(
        summary["epsilon"],
        summary["dinov3_cls_mean"] - summary["dinov3_cls_std"],
        summary["dinov3_cls_mean"] + summary["dinov3_cls_std"],
        alpha=0.15,
    )

    plt.ylim(-0.1, 1.02)

    plt.xlabel("Perturbation budget (ε)")
    plt.ylabel("Mean cosine similarity")
    plt.title("CLS-token robustness under PGD")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    cls_output_path = results_dir / "pgd_cls_similarity.png"

    plt.savefig(
        cls_output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    # Patch-token similarity plot
    plt.figure(figsize=(6, 4))

    plt.plot(
        summary["epsilon"],
        summary["dinov2_patch_mean"],
        marker="o",
        label="DINOv2 ViT-L/14",
    )

    plt.fill_between(
        summary["epsilon"],
        summary["dinov2_patch_mean"] - summary["dinov2_patch_mean_std"],
        summary["dinov2_patch_mean"] + summary["dinov2_patch_mean_std"],
        alpha=0.15,
    )

    plt.plot(
        summary["epsilon"],
        summary["dinov3_patch_mean"],
        marker="o",
        label="DINOv3 ViT-L/16",
    )

    plt.fill_between(
        summary["epsilon"],
        summary["dinov3_patch_mean"] - summary["dinov3_patch_mean_std"],
        summary["dinov3_patch_mean"] + summary["dinov3_patch_mean_std"],
        alpha=0.15,
    )

    plt.ylim(0.4, 1.02)

    plt.xlabel("Perturbation budget (ε)")
    plt.ylabel("Mean cosine similarity")
    plt.title("Patch-token robustness under PGD")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    patch_output_path = results_dir / "pgd_patch_similarity.png"

    plt.savefig(
        patch_output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    print(f"Saved CLS plot to: {cls_output_path}")
    print(f"Saved patch plot to: {patch_output_path}")


if __name__ == "__main__":
    # output_inspection()
    parsing_results()
    # plotting_results()
