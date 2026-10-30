"""Locate locally stored DINO checkpoints without embedding user-specific paths."""

import os
from pathlib import Path


def checkpoint_directory() -> Path:
    """Use DINO_CHECKPOINT_DIR or the project's sibling checkpoints directory.

    On Borg, the repository and checkpoints are siblings inside dino_robustness/.
    The environment override also supports other layouts and local tests.
    """
    override = os.environ.get("DINO_CHECKPOINT_DIR")
    if override:
        return Path(override).expanduser()

    repo_root = Path(__file__).resolve().parents[3]
    return repo_root.parent / "checkpoints"


def checkpoint_file(filename: str) -> Path:
    """Return an existing checkpoint or fail with a useful path diagnostic."""
    path = checkpoint_directory() / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. "
            "Set DINO_CHECKPOINT_DIR to the directory containing your .pth files."
        )
    return path
