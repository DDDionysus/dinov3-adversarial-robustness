from src.utils.device import get_device
from src.utils.dataset import get_dataset
from src.utils.transforms import get_normalise
from src.models.loader import load_model
from src.experiments.generate_rfgsm import rfgsm_perturbation
from src.experiments.generate_pgd import pgd_perturbation
from src.experiments.evaluate_perturbation import evaluate_perturbation


dataset_name = "imagenet"
random_seed = 42

v2_model = "dinov2_vitl14"
v3_model = "dinov3_vitl16"

device = get_device()
print("Using device: ", device)

# load models
dinov2 = load_model(v2_model, device)
dinov3 = load_model(v3_model, device)

dataset = get_dataset(dataset_name)

rfgsm_dir = "output/perturbations/sanity_test/rfgsm"
pgd_dir = "output/perturbations/sanity_test/pgd"

# generate and save perturbation
rfgsm_perturbation(dinov2, v2_model, device, dataset, rfgsm_dir, random_seed)
rfgsm_perturbation(dinov3, v3_model, device, dataset, rfgsm_dir, random_seed)
pgd_perturbation(dinov2, v2_model, device, dataset, pgd_dir, random_seed)
pgd_perturbation(dinov3, v3_model, device, dataset, pgd_dir, random_seed)

# Load saved tensors and evaluate them
normalisation = get_normalise()

evaluate_perturbation(
    model=dinov2,
    device=device,
    normalisation=normalisation,
    input_dir=f"{rfgsm_dir}/{v2_model}",
    output_path="output/results/sanity_test/rfgsm_dinov2.csv",
)

evaluate_perturbation(
    model=dinov3,
    device=device,
    normalisation=normalisation,
    input_dir=f"{rfgsm_dir}/{v3_model}",
    output_path="output/results/sanity_test/rfgsm_dinov3.csv",
)

evaluate_perturbation(
    model=dinov2,
    device=device,
    normalisation=normalisation,
    input_dir=f"{pgd_dir}/{v2_model}",
    output_path="output/results/sanity_test/pgd_dinov2.csv",
)

evaluate_perturbation(
    model=dinov3,
    device=device,
    normalisation=normalisation,
    input_dir=f"{pgd_dir}/{v3_model}",
    output_path="output/results/sanity_test/pgd_dinov3.csv",
)
