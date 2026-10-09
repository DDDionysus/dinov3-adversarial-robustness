from pathlib import Path
from tqdm import tqdm

from src.experiments.generate_perturbation import generate_perturbation
from src.utils.transforms import get_transform, get_normalise


def rfgsm_perturbation(model, model_name, device, dataset, output_path, seed=42):
    epsilon = [
        1 / 255,
        2 / 255,
        4 / 255,
        8 / 255,
    ]

    objective = ["cls", "patch", "gram"]

    # transform: resize and centre crop, without normalisation
    transform = get_transform(resolution=448)
    normalisation = get_normalise()

    output_path = Path(output_path)

    for idx in tqdm(range(len(dataset)), desc="Generating R-FGSM perturbations"):
        img, img_id = dataset[idx]

        for o in objective:
            for e in epsilon:
                sample_path = output_path / model_name / o / f"eps_{e * 255:g}/{Path(img_id).stem}.pt"

                generate_perturbation(
                    model=model,
                    device=device,
                    image=img,
                    image_id=img_id,
                    transform=transform,
                    normalisation=normalisation,
                    model_name=model_name,
                    attack_type="rfgsm",
                    objective=o,
                    epsilon=e,
                    seed=seed,
                    output_path=sample_path,
                )
