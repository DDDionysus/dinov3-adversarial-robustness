from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


project_root = Path(__file__).resolve().parents[2]

result_path = project_root / "output/seg_results/seg_results.csv"

# load results
df = pd.read_csv(result_path)

# rename models for presentation
df["model"] = df["model"].replace({
    "dinov2-vitl14": "DINOv2 ViT-L/14",
    "dinov3-vitl16": "DINOv3 ViT-L/16"
})

# convert mIoU to percentage
df["mIoU"] = df["mIoU"] * 100

plt.figure(figsize=(9, 6))

for (model, attack), group in df.groupby(["model", "attack"]):

    group = group.sort_values("epsilon")

    plt.plot(
        group["epsilon"],
        group["mIoU"],
        marker="o",
        linewidth=2,
        label=f"{model} - {attack.upper()}"
    )

plt.xlabel(r"$\epsilon$")
plt.ylabel("mIoU (%)")

plt.ylim(0, 100)

plt.title(
    "Semantic segmentation robustness under FGSM and PGD"
)

# caption: mIoU degradation under FGSM and PGD attacks with increasing L_\infty perturbation magnitude.

plt.legend(
    fontsize=9
)

plt.grid(True)
plt.tight_layout()
plt.show()
