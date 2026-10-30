"""Resolve official DINO PyTorch Hub sources for reproducible local loading.

When DINO_HUB_ROOT is configured, no GitHub Hub source fallback is allowed:
source code must exist at <DINO_HUB_ROOT>/<dinov2|dinov3>.
Without the override, retain the legacy GitHub Hub behavior for old scripts.
Production launchers should always set DINO_HUB_ROOT explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path


def hub_source_kwargs(project: str) -> dict[str, str]:
    """Return checked ``repo_or_dir`` and ``source`` arguments for torch.hub.load.

    This only validates source-tree structure; the official Hub implementation
    performs the actual checkpoint loading. No network call occurs here.
    """
    if project not in {"dinov2", "dinov3"}:
        raise ValueError(f"Unsupported DINO source project: {project!r}")

    root_value = os.environ.get("DINO_HUB_ROOT")
    if root_value is None:
        # Backward compatibility for legacy workstation entry points. HPC jobs
        # must configure DINO_HUB_ROOT to enforce local-only source loading.
        return {"repo_or_dir": f"facebookresearch/{project}", "source": "github"}

    if not root_value.strip():
        raise ValueError("DINO_HUB_ROOT must be a nonempty directory")
    root = Path(root_value).expanduser().resolve()
    checkout = root / project
    if not checkout.is_dir():
        raise FileNotFoundError(f"Missing local {project} checkout: {checkout}")
    if not (checkout / "hubconf.py").is_file():
        raise FileNotFoundError(f"Missing official Hub entry point: {checkout / 'hubconf.py'}")
    if not (checkout / project / "__init__.py").is_file():
        raise FileNotFoundError(f"Missing official Python package: {checkout / project}")

    return {"repo_or_dir": str(checkout), "source": "local"}
