from pathlib import Path
import math
import os
from torch.utils.data import Subset

from src.utils.device import get_device
from src.models.hpc_loader import load_model
from src.experiments.generate_rfgsm import fgsm_comparison
from src.experiments.generate_pgd import pgd_comparison
from src.datasets.imagenet import ImageNetValDataset
from src.datasets.nyudepth import NYUDepthDataset
from src.datasets.voc import VOCSegmentationDataset


def get_dataset(dataset_name="imagenet"):
    """
    Load a dataset for representation-level robustness evaluation.
    All datasets are returned in representation mode, where only the RGB image and
    sample identifier are required.

    Params:
        dataset_name: str
            Name of the dataset to load

    Returns:
        torch.utils.data.Dataset
            Dataset wrapper configured for representation-level evaluation.
    """

    project_root = Path("/scratch_tide/hw2825")

    dataset_path = {
        "imagenet": project_root / "datasets/subset/val_subset",
        "voc": project_root / "datasets/VOC2012",
        "nyu": project_root / "datasets/NYU_Depth_v2/nyu_depth_v2_labeled.mat"
    }

    if dataset_name not in dataset_path:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if dataset_name == "voc":
        return VOCSegmentationDataset(dataset_path[dataset_name], split="val", representation=True)

    if dataset_name == "nyu":
        return NYUDepthDataset(dataset_path[dataset_name], representation=True)

    return ImageNetValDataset(dataset_path[dataset_name])


if __name__ == "__main__":
    project_root = Path("/scratch_tide/hw2825")

    # edit here
    # dataset_names = ["voc", "nyu"]

    dataset_names = ["nyu"]

    device = get_device()
    print("Using device: ", device)

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    # load models
    dinov2 = load_model("dinov2_vitl14", device)
    dinov3 = load_model("dinov3_vitl16", device)

    for dataset_name in dataset_names:
        # fgsm_output_path = project_root / f"output/results/fgsm_{dataset_name}_L.csv"
        dataset = get_dataset(dataset_name)

        chunk_size = math.ceil(len(dataset) / num_tasks)

        start_idx = task_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(dataset))

        indices = list(range(start_idx, end_idx))
        dataset_chunk = Subset(dataset, indices)

        print(
            f"Task {task_id}/{num_tasks - 1}: "
            f"Processing dataset indices {start_idx} to {end_idx - 1}"
        )

        pgd_output_path = (
                project_root
                / f"output/results/pgd_{dataset_name}_L_chunk_{task_id:02d}.csv"
        )

        # fgsm_comparison(dinov2, dinov3, device, dataset, fgsm_output_path)
        pgd_comparison(dinov2, dinov3, device, dataset_chunk, pgd_output_path)

    print("All done!")
