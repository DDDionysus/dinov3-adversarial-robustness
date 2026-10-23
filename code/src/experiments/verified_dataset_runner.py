"""Verified 448-pixel inputs for the canonical representation-attack runner.

This module is intentionally *not* a downstream task evaluator. It checks
frozen image and optional target bytes before preparing each RGB input and
passing it to run_representation_image. Segmentation/depth target resampling,
valid-pixel masks and official evaluation heads will be implemented separately.

The unnormalized [0,1] image is used for attack constraints and archives;
ImageNet RGB normalization is applied only at each backbone forward pass.
"""

from __future__ import annotations

from dataclasses import asdict
from itertools import islice
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch

from src.experiments.dataset_manifests import verify_manifest
from src.experiments.experiment_provenance import (
    _identity, iter_verified_samples, make_preprocessing_spec,
)
from src.experiments.run_representation_image import AttackSpec, run_representation_image
from src.utils.metric_records import MetricJSONLWriter
from src.utils.perturbation_archive import PerturbationShardWriter
from src.utils.spatial_map_archive import SpatialMapShardWriter


def validate_preprocessing_spec(spec: dict) -> None:
    """Fail closed if a frozen spec was altered or contains unsupported fields."""
    if not isinstance(spec, dict):
        raise TypeError("preprocessing must be a versioned dictionary")
    required = {
        "schema_version", "input_mode", "input_dtype", "input_range", "image_size_hw",
        "resize_mode", "interpolation", "antialias", "normalization_mean",
        "normalization_std", "target_handling", "preprocessing_id",
    }
    if set(spec) != required:
        raise ValueError("Preprocessing specification has missing or unexpected fields")
    try:
        reference = make_preprocessing_spec(
            resize_mode=spec["resize_mode"],
            interpolation=spec["interpolation"],
            rgb_normalization_mean=tuple(spec["normalization_mean"]),
            rgb_normalization_std=tuple(spec["normalization_std"]),
            target_handling=spec["target_handling"],
        )
    except (TypeError, KeyError) as exc:
        raise ValueError("Malformed preprocessing specification") from exc
    if reference != spec:
        raise ValueError("Preprocessing specification or checksum does not match implementation")


def preprocess_rgb(path: str | Path, spec: dict) -> torch.Tensor:
    """Return [1,3,448,448] float32 *unnormalized* pixels, no GPU work.

    - stretch: resize width and height to 448;
    - short_side_center_crop: preserve aspect, set shorter side to 448,
      then center crop 448 pixels using Pillow's deterministic pixel geometry.
    All resizing uses Pillow bilinear/bicubic with antialiasing.
    """
    validate_preprocessing_spec(spec)
    side = 448
    method = {"bilinear": Image.Resampling.BILINEAR,
              "bicubic": Image.Resampling.BICUBIC}[spec["interpolation"]]
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        w, h = rgb.size
        if w < 1 or h < 1:
            raise ValueError("Invalid RGB source dimensions")
        if spec["resize_mode"] == "stretch":
            rgb = rgb.resize((side, side), resample=method)
        else:
            if w <= h:
                nw, nh = side, int(side * h / w)
            else:
                nw, nh = int(side * w / h), side
            rgb = rgb.resize((nw, nh), resample=method)
            left = int(round((nw - side) / 2.0))
            top = int(round((nh - side) / 2.0))
            rgb = rgb.crop((left, top, left + side, top + side))
        pixels = np.asarray(rgb, dtype=np.uint8)
    # Recast and divide by 255 in float32, as recorded in provenance.
    chw = torch.from_numpy(pixels.copy()).permute(2, 0, 1).unsqueeze(0)
    result = chw.to(torch.float32).div_(255.0)
    if result.shape != (1, 3, side, side):
        raise RuntimeError("Unexpected preprocessed image shape")
    return result


def normalizer_from_spec(spec: dict):
    """Normalize model inputs, without changing archived pixel-domain data."""
    validate_preprocessing_spec(spec)
    mean = tuple(spec["normalization_mean"])
    std = tuple(spec["normalization_std"])

    def normalize(batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 4 or batch.shape[1] != 3:
            raise ValueError("Expected RGB batch [B,3,H,W]")
        if not batch.is_floating_point():
            raise ValueError("Normalize only floating-point tensors")
        means = batch.new_tensor(mean).view(1, 3, 1, 1)
        stds = batch.new_tensor(std).view(1, 3, 1, 1)
        return (batch - means) / stds

    return normalize


def _check_provenance(manifest: dict, provenance: dict, *, manifest_role: str,
                      checkpoint_role: str, preprocessing: dict) -> tuple[str, str]:
    """Bind every sample to immutable experiment, manifest and checkpoint IDs."""
    verify_manifest(manifest)
    validate_preprocessing_spec(preprocessing)
    if not isinstance(provenance, dict):
        raise ValueError("Missing experiment provenance")
    body = {k: v for k, v in provenance.items() if k != "experiment_id"}
    if provenance.get("schema_version") != 1:
        raise ValueError("Unsupported experiment provenance schema")
    if provenance.get("experiment_id") != _identity(body):
        raise ValueError("Experiment provenance checksum mismatch")
    if provenance.get("preprocessing") != preprocessing:
        raise ValueError("Preprocessing differs from frozen experiment")
    if provenance.get("dataset_manifests", {}).get(manifest_role) != manifest["manifest_id"]:
        raise ValueError("Dataset selection differs from frozen experiment")
    checkpoint = provenance.get("checkpoints", {}).get(checkpoint_role)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", checkpoint["sha256"]) is None:
        raise ValueError("Checkpoint is not identified in frozen experiment")
    return checkpoint["sha256"], provenance["experiment_id"]


def run_verified_representation_samples(
    *, model: torch.nn.Module, manifest: dict, dataset_root: Path,
    manifest_role: str, checkpoint_role: str, provenance: dict,
    model_name: str, specs: list[AttackSpec], device: str | torch.device,
    attacks: PerturbationShardWriter, metrics: MetricJSONLWriter,
    spatial_maps: SpatialMapShardWriter | None = None,
    source_paths: dict[str, Path] | None = None,
    image_ids: set[str] | None = None, max_samples: int | None = None,
) -> dict[str, int]:
    """Run a frozen, verified dataset selection using the single-image engine.

    Source manifest and *per-image* RGB/target SHA-256 are checked, including on
    restart, before any model forward pass on that image. The caller owns the
    writer partition and checkpoint/model loading. Dataset partitions must be
    disjoint and have stable archive prefixes and JSONL paths across restarts.
    """
    if max_samples is not None and (not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples < 1):
        raise ValueError("max_samples must be a positive integer or None")
    preprocessing = provenance.get("preprocessing") if isinstance(provenance, dict) else None
    checkpoint_id, protocol_id = _check_provenance(
        manifest, provenance, manifest_role=manifest_role,
        checkpoint_role=checkpoint_role, preprocessing=preprocessing,
    )
    if not isinstance(specs, list) or not specs or not all(isinstance(s, AttackSpec) for s in specs):
        raise ValueError("Expected a nonempty list of attack specifications")
    # An image-specific run may request a subset of the frozen attack plan,
    # but must never add an attack absent from the provenance record.
    approved = provenance.get("attack_settings")
    if not isinstance(approved, list) or not approved:
        raise ValueError("Frozen provenance has no attack settings")
    for spec in specs:
        if asdict(spec) not in approved:
            raise ValueError("Attack configuration is absent from frozen provenance")
    normalize = normalizer_from_spec(preprocessing)
    records = iter_verified_samples(manifest, dataset_root, source_paths=source_paths, image_ids=image_ids)
    if max_samples is not None:
        records = islice(records, max_samples)

    result = {"images_processed": 0, "already_scored": 0, "reused_attack": 0,
              "generated_attack": 0, "evaluated": 0, "spatial_maps_saved": 0}
    for sample, rgb_path, _verified_target in records:
        x_clean = preprocess_rgb(rgb_path, preprocessing).to(device)
        counts = run_representation_image(
            model=model, x_clean=x_clean, normalisation=normalize,
            dataset=manifest["dataset"], image_id=sample["image_id"],
            model_name=model_name, checkpoint_id=checkpoint_id,
            preprocessing=preprocessing["preprocessing_id"],
            protocol_id=protocol_id, specs=specs,
            attacks=attacks, metrics=metrics, spatial_maps=spatial_maps,
        )
        result["images_processed"] += 1
        for k, v in counts.items():
            result[k] += v
        del x_clean
    return result
