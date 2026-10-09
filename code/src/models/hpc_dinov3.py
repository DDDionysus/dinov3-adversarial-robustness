from pathlib import Path

import torch


# REPO_DIR = "C:/Users/10031/.cache/torch/hub/facebookresearch_dinov3_main"


def get_checkpoint(model_name):
    """
    Get path to pretrained DINOv3 model checkpoint.

    Params:
        model_name: str
            Name of pretrained model.

    Returns:
        Path:
            Path to pretrained DINOv3 model checkpoint.
    """

    project_root = Path(__file__).resolve().parents[3]
    checkpoints = {
        "dinov3_vits16": project_root / "checkpoints/backbones/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "dinov3_vitl16": "/scratch_tide/hw2825/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",

        # "dinov3_vit7b16": project_root / ""
    }

    if model_name not in checkpoints:
        raise ValueError(
            f"Unsupported DINOv3 model: {model_name}"
        )

    return checkpoints[model_name]


def load_dinov3(model_name, device):
    """
    Load pretrained DINOv3 model.

    Params:
        model_name: str
            Name of pretrained model.
        device: str
            Device to use for inference (cpu/cuda/mps).

    Returns:
        torch.nn.Module:
            Loaded DINOv3 model in evaluation mode.
    """

    """if weight is not None:
        weight = Path(weight).resolve().as_uri()"""

    weight = get_checkpoint(model_name)

    model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov3",
        # repo_or_dir=REPO_DIR,
        model=model_name,
        weights=str(weight),
    )

    model = model.to(device)
    model.eval()

    return model
