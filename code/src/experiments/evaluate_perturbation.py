from pathlib import Path
import torch
from tqdm import tqdm

from src.experiments.evaluate_model import evaluate_model
from src.utils.save_as import save_csv


def evaluate_perturbation(model, device, normalisation, input_dir, output_path):
    """

    """

    input_dir = Path(input_dir)
    perturbation_files = sorted(input_dir.rglob("*.pt"))

    if not perturbation_files:
        raise ValueError(
            f"No perturbation files found in {input_dir}"
        )

    results = []

    for perturbation_path in tqdm(perturbation_files, desc="Evaluating saved perturbations"):
        sample = torch.load(perturbation_path, map_location=device)

        x_clean = sample["x_clean"].to(device)
        x_adv = sample["x_adv"].to(device)
        metadata = sample["metadata"]

        scores = evaluate_model(model, device, x_clean, x_adv, normalisation)

        results.append({
            **metadata,
            **scores,
            "tensor_path": str(perturbation_path),
        })

    save_csv(results, output_path)

    print(f"Saved evaluation results to {output_path}")
