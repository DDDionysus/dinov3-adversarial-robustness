from pathlib import Path

import torch
from torchvision.models import get_weight


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
        "dinov3_vitl16": project_root / "checkpoints/backbones/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "dinov3_vit7b16": project_root / "checkpoints/backbones/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    }

    if model_name not in checkpoints:
        raise ValueError(
            f"Unsupported DINOv3 model: {model_name}"
        )

    return checkpoints[model_name]


def get_downstream_checkpoint(task):
    """

    """

    project_root = Path(__file__).resolve().parents[3]
    checkpoints = {
        "classification": project_root / "checkpoints/weights/weights/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth",
        "segmentation": project_root / "checkpoints/weights/weights/dinov3_vit7b16_ade20k_m2f_head.pth",
        "depth": project_root / "checkpoints/weights/weights/dinov3_vit7b16_synthmix_dpt_head.pth",
    }

    if task not in checkpoints:
        raise ValueError(f"Unsupported downstream task: {task}")

    return checkpoints[task]


def load_dinov3(model_name, device, checkpoint_path=None):
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

    if checkpoint_path is None:
        checkpoint_path = get_checkpoint(model_name)

    model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov3",
        # repo_or_dir=REPO_DIR,
        model=model_name,
        weights=str(checkpoint_path),
    )

    model = model.to(device)
    model.eval()
    # Keep parameters fixed while allowing gradients with respect to the input.
    model.requires_grad_(False)

    return model


def load_dinov3_downstream(task, device):
    """

    """

    backbone_weight = get_weight("dinov3_vit7b16")
    downstream_weight = get_downstream_checkpoint(task)

    downstream_models = {
        "classification": "dinov3_vit7b16_lc",
        "segmentation": "dinov3_vit7b16_ms",
        "depth": "dinov3_vit7b16_dd",
    }

    model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov3",
        model=downstream_models[task],
        weights=str(downstream_weight),
        backbone_weights=str(backbone_weight),
    )

    model = model.to(device)
    model.eval()
    # Keep parameters fixed while allowing gradients with respect to the input.
    model.requires_grad_(False)

    return model
