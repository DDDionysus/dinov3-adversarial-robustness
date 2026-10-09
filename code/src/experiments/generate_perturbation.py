from pathlib import Path
import torch

from src.attacks.fgsm import rfgsm_attack
from src.attacks.pgd import pgd_attack


def generate_perturbation(
        model,
        device,
        image,
        image_id,
        transform,
        normalisation,
        model_name,
        attack_type,
        objective,
        epsilon,
        seed,
        output_path,
        alpha=None,
        steps=None,
):
    """

    """

    # PIL image -> [1, C, H, W], pixel values in [0, 1]
    x_clean = transform(image).unsqueeze(0).to(device)

    if attack_type == "rfgsm":
        x_adv = rfgsm_attack(
            model=model,
            normalisation=normalisation,
            x=x_clean,
            epsilon=epsilon,
            seed=seed,
            objective=objective,
        )
    elif attack_type == "pgd":
        if alpha is None or steps is None:
            raise ValueError(
                "alpha and steps must be provided for PGD"
            )

        x_adv = pgd_attack(
            model=model,
            normalisation=normalisation,
            x=x_clean,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            seed=seed,
            objective=objective,
        )
    else:
        raise ValueError(
            f"Unknown attack type: {attack_type}"
        )

    # Validate the saved pixel-domain inputs
    if not torch.isfinite(x_adv).all():
        raise ValueError("Generated adversarial tensor contains NaN or Inf")

    if x_clean.min().item() < 0.0 or x_clean.max().item() > 1.0:
        raise ValueError("x_clean is outside the [0, 1] pixel range")

    if x_adv.min().item() < 0.0 or x_adv.max().item() > 1.0:
        raise ValueError("x_adv is outside the [0, 1] pixel range")

    linf = (x_adv - x_clean).abs().max().item()

    if linf > epsilon + 1e-6:
        raise ValueError(
            f"Perturbation exceeds epsilon: "
            f"Linf={linf}, epsilon={epsilon}"
        )

    sample = {
        "x_clean": x_clean.detach().cpu(),
        "x_adv": x_adv.detach().cpu(),
        "metadata": {
            "image_id": image_id,
            "model": model_name,
            "attack": attack_type,
            "objective": objective,
            "epsilon": float(epsilon),
            "seed": int(seed),
            "alpha": None if alpha is None else float(alpha),
            "steps": None if steps is None else int(steps),
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(sample, output_path)

    return sample
