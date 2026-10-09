from src.models.hpc_dinov2 import load_dinov2
from src.models.hpc_dinov3 import load_dinov3


def load_model(model_name, device):
    """
    Load requested model variant.
    """

    if model_name.startswith("dinov2"):
        return load_dinov2(model_name, device)
    elif model_name.startswith("dinov3"):
        return load_dinov3(model_name, device)

    else:
        raise ValueError(f"Invalid model name: {model_name}")
