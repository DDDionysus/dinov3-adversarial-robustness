import os
from pathlib import Path
from torch.utils.data import Subset

from src.utils.device import get_device
from src.utils.dataset import get_dataset
from src.utils.transforms import get_normalise
from src.models.loader import load_model
from src.experiments.generate_rfgsm import rfgsm_perturbation
from src.experiments.generate_pgd import pgd_perturbation
from src.experiments.evaluate_perturbation import evaluate_perturbation


# experiment settings
model_name = "dinov3_vit7b16"
dataset_name = "imagenet"
random_seed = 42


# Supplied by the HPC submission script
stage = os.environ["EXPERIMENT_STAGE"]

data_root = Path(
    os.environ["IMAGENET_VAL_PATH"]
)

checkpoint_path = Path(
    os.environ["DINOV3_7B_CHECKPOINT"]
)

output_root = Path(
    os.environ["EXPERIMENT_OUTPUT_DIR"]
)


valid_stages = {
    "generate_rfgsm",
    "generate_pgd",
    "evaluate_rfgsm",
    "evaluate_pgd",
}

if stage not in valid_stages:
    raise ValueError(
        f"Unknown EXPERIMENT_STAGE: {stage}. "
        f"Expected one of {sorted(valid_stages)}"
    )


# Output locations
rfgsm_dir = output_root / "perturbations/rfgsm"

pgd_dir = output_root / "perturbations/pgd"

rfgsm_result = output_root / "results/rfgsm_dinov3.csv"

pgd_result = output_root / "results/pgd_dinov3.csv"


# load model
device = get_device()
print("Using device:", device)
print("Stage:", stage)
print("Dataset:", data_root)
print("Checkpoint:", checkpoint_path)
print("Output root:", output_root)

dinov3 = load_model(model_name, device, checkpoint_path)

if stage in {"generate_rfgsm", "generate_pgd"}:
    dataset = get_dataset(dataset_name, data_root)

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    indices = list(range(task_id, len(dataset), num_tasks))

    dataset_chunk = Subset(dataset, indices)

    print(
        f"Task {task_id}/{num_tasks}: "
        f"processing {len(dataset_chunk)} "
        f"of {len(dataset)} images"
    )

    # generate and save perturbation
    if stage == "generate_rfgsm":
        rfgsm_perturbation(dinov3, model_name, device, dataset_chunk, rfgsm_dir, random_seed)
    elif stage == "generate_pgd":
        pgd_perturbation(dinov3, model_name, device, dataset_chunk, pgd_dir, random_seed)

elif stage in {"evaluate_rfgsm", "evaluate_pgd"}:
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        raise ValueError(
            "Evaluation must run as a single job, "
            "not as a Slurm array."
        )

    normalisation = get_normalise()

    if stage == "evaluate_rfgsm":
        evaluate_perturbation(
            model=dinov3,
            device=device,
            normalisation=normalisation,
            input_dir=rfgsm_dir / model_name,
            output_path=rfgsm_result,
        )
    elif stage == "evaluate_pgd":
        evaluate_perturbation(
            model=dinov3,
            device=device,
            normalisation=normalisation,
            input_dir=pgd_dir / model_name,
            output_path=pgd_result,
        )
