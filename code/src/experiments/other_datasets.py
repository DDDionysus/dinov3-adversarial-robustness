from pathlib import Path

from src.utils.device import get_device
from src.utils.dataset import get_dataset
from src.models.loader import load_model
from src.experiments.generate_rfgsm import fgsm_comparison
from src.experiments.generate_pgd import pgd_comparison


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    # edit here
    # dataset_names = ["imagenet", "voc", "nyu"]

    dataset_names = ["nyu"]

    device = get_device()
    print("Using device: ", device)

    # load models
    dinov2 = load_model("dinov2_vitl14", device)
    dinov3 = load_model("dinov3_vitl16", device)

    for dataset_name in dataset_names:
        fgsm_output_path = project_root / f"output/results/fgsm_{dataset_name}_L.csv"
        pgd_output_path = project_root / f"output/results/pgd_{dataset_name}_L.csv"

        dataset = get_dataset(dataset_name)

        fgsm_comparison(dinov2, dinov3, device, dataset, fgsm_output_path)
        pgd_comparison(dinov2, dinov3, device, dataset, pgd_output_path)

    print("All done!")
