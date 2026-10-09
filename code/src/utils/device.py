import torch


def get_device():
    """
    Select the available computation device.

    Returns:
        str:
            Device identifier: 'cuda', 'mps' or 'cpu'.
    """

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"
