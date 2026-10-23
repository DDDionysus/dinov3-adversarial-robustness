import torch

from src.models.checkpoint_paths import checkpoint_file
from src.models.hub_source import hub_source_kwargs


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

    checkpoints = {
        "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m.pth",
        "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m.pth",
        "dinov3_vit7b16": "dinov3_vit7b16_pretrain_lvd1689m.pth",
    }

    if model_name not in checkpoints:
        raise ValueError(
            f"Unsupported DINOv3 model: {model_name}"
        )

    return checkpoint_file(checkpoints[model_name])


def get_downstream_checkpoint(task):
    """

    """

    checkpoints = {
        "classification": "dinov3_vit7b16_imagenet1k_linear_head.pth",
        "segmentation": "dinov3_vit7b16_ade20k_m2f_head.pth",
        "depth": "dinov3_vit7b16_synthmix_dpt_head.pth",
    }

    if task not in checkpoints:
        raise ValueError(f"Unsupported downstream task: {task}")

    return checkpoint_file(checkpoints[task])


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
        **hub_source_kwargs("dinov3"),
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

    downstream_models = {
        "classification": "dinov3_vit7b16_lc",
        "segmentation": "dinov3_vit7b16_ms",
        "depth": "dinov3_vit7b16_dd",
    }
    if task not in downstream_models:
        raise ValueError(f"Unsupported downstream task: {task}")

    backbone_weight = get_checkpoint("dinov3_vit7b16")
    downstream_weight = get_downstream_checkpoint(task)

    model = torch.hub.load(
        **hub_source_kwargs("dinov3"),
        model=downstream_models[task],
        weights=str(downstream_weight),
        backbone_weights=str(backbone_weight),
    )

    model = model.to(device)
    model.eval()
    # Keep parameters fixed while allowing gradients with respect to the input.
    model.requires_grad_(False)

    return model
