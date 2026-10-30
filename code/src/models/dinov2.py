import torch

from src.models.checkpoint_paths import checkpoint_file
from src.models.hub_source import hub_source_kwargs


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

    checkpoints = {
        "dinov2_vits14": "dinov2_vits14_pretrain.pth",
        "dinov2_vitl14": "dinov2_vitl14_pretrain.pth",
    }

    if model_name not in checkpoints:
        raise ValueError(
            f"Unsupported DINOv2 model: {model_name}"
        )

    return checkpoint_file(checkpoints[model_name])


def load_dinov2(model_name, device, checkpoint_path=None):
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

    if checkpoint_path is None:
        checkpoint_path = get_checkpoint(model_name)

    model = torch.hub.load(
        **hub_source_kwargs("dinov2"),
        model=model_name,
        weights=str(checkpoint_path),
    )

    model = model.to(device)
    model.eval()
    # Keep parameters fixed while allowing gradients with respect to the input.
    model.requires_grad_(False)

    return model
