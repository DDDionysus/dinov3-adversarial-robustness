from src.models.dinov2 import load_dinov2
from src.models.dinov3 import load_dinov3


def load_model(model_name, device, checkpoint_path=None):
    """
    Load requested model variant.
    """

    if model_name.startswith("dinov2"):
        return load_dinov2(model_name, device, checkpoint_path=checkpoint_path)
    elif model_name.startswith("dinov3"):
        return load_dinov3(model_name, device, checkpoint_path=checkpoint_path)

    else:
        raise ValueError(f"Invalid model name: {model_name}")
