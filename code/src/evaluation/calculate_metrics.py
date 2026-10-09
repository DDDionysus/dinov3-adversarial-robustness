from pathlib import Path
import ast

import pandas as pd


def summarise_raw_results(path):
    """
    Summarise raw representation-level robustness results by perturbation budget.
    The metric dictionaries stored in the CSV are parsed into separate CLS-token
    and patch-token metrics, then averaged across images for each epsilon value.

    Params:
        path: str or pathlib.Path
            Path to CSV containing raw representation-level robustness results.

    Returns:
        pandas.DataFrame
            Dataset-level meawn robustness metrics grouped by epsilon.
    """

    df = pd.read_csv(path)

    df["dinov2_metrics"] = df["dinov2_cls"].apply(ast.literal_eval)
    df["dinov3_metrics"] = df["dinov3_cls"].apply(ast.literal_eval)

    df["dinov2_cls_similarity"] = df["dinov2_metrics"].apply(
        lambda x: x["cls_similarity"]
    )
    df["dinov3_cls_similarity"] = df["dinov3_metrics"].apply(
        lambda x: x["cls_similarity"]
    )

    df["dinov2_patch_mean"] = df["dinov2_metrics"].apply(
        lambda x: x["patch_similarity_mean"]
    )
    df["dinov3_patch_mean"] = df["dinov3_metrics"].apply(
        lambda x: x["patch_similarity_mean"]
    )

    df["dinov2_patch_std"] = df["dinov2_metrics"].apply(
        lambda x: x["patch_similarity_std"]
    )
    df["dinov3_patch_std"] = df["dinov3_metrics"].apply(
        lambda x: x["patch_similarity_std"]
    )

    return df.groupby("epsilon").agg(
        dinov2_cls_mean=("dinov2_cls_similarity", "mean"),
        dinov3_cls_mean=("dinov3_cls_similarity", "mean"),

        dinov2_patch_mean=("dinov2_patch_mean", "mean"),
        dinov3_patch_mean=("dinov3_patch_mean", "mean"),

        dinov2_patch_std_mean=("dinov2_patch_std", "mean"),
        dinov3_patch_std_mean=("dinov3_patch_std", "mean"),
    ).reset_index()


def print_filtered(df, name):
    """
    Print the lowest and highest perturbation budget results used in the cross-dataset
    summary table.
    """

    columns = [
        "epsilon",
        "dinov2_cls_mean",
        "dinov3_cls_mean",
        "dinov2_patch_mean",
        "dinov3_patch_mean",
        "dinov2_patch_std_mean",
        "dinov3_patch_std_mean",
    ]

    filtered = df[
        df["epsilon"].isin([0.001, 0.05])
    ][columns]

    print(f"\n{name}")
    print(filtered.to_string(index=False))


def main():
    project_root = Path(__file__).resolve().parents[2]

    imagenet_fgsm = (
        project_root
        / "output/hpc_results/smaller_test/fgsm_imagenet_L_summary.csv"
    )
    imagenet_pgd = (
        project_root
        / "output/hpc_results/pgd/pgd_imagenet_L_summary.csv"
    )

    voc_fgsm = project_root / "output/results/fgsm_voc_L.csv"
    voc_pgd = project_root / "output/results/pgd_voc_L.csv"

    nyu_fgsm = project_root / "output/results/fgsm_nyu_L.csv"
    # nyu_pgd = project_root / "output/results/pgd_nyu_L.csv"

    imagenet_fgsm_df = pd.read_csv(imagenet_fgsm)
    imagenet_pgd_df = pd.read_csv(imagenet_pgd)

    voc_fgsm_df = summarise_raw_results(voc_fgsm)
    voc_pgd_df = summarise_raw_results(voc_pgd)
    nyu_fgsm_df = summarise_raw_results(nyu_fgsm)

    print_filtered(imagenet_fgsm_df, "ImageNet FGSM")
    print_filtered(imagenet_pgd_df, "ImageNet PGD")
    print_filtered(voc_fgsm_df, "VOC FGSM")
    print_filtered(voc_pgd_df, "VOC PGD")
    print_filtered(nyu_fgsm_df, "NYU FGSM")


if __name__ == "__main__":
    main()
