from pathlib import Path

import torch


def get_checkpoint(model_name):
    """
    Get path to pretrained DINOv2 model checkpoint.

    Params:
        model_name: str
            Name of pretrained model.

    Returns:
        Path:
            Path to pretrained DINOv2 model checkpoint.
    """

    project_root = Path(__file__).resolve().parents[3]
    checkpoints = {
        "dinov2_vits14": project_root / "checkpoints/backbones/dinov2/dinov2_vits14_pretrain.pth",
        "dinov2_vitl14": "/scratch_tide/hw2825/checkpoints/dinov2_vitl14_pretrain.pth"

        # "dinov2_vitg": project_root / ""
    }

    if model_name not in checkpoints:
        raise ValueError(
            f"Unsupported DINOv2 model: {model_name}"
        )

    return checkpoints[model_name]


def load_dinov2(model_name, device):
    """
    Load pretrained DINOv2 model.

    Params:
        model_name: str
            Name of pretrained model.
        device: str
            Device to use for inference (cpu/cuda/mps).

    Returns:
        torch.nn.Module:
            Loaded DINOv2 model in evaluation mode.
    """

    weight = get_checkpoint(model_name)

    model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2",
        model=model_name,
        weights=str(weight),
    )

    model = model.to(device)
    model.eval()

    return model
