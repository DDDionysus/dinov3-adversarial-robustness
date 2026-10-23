"""Immutable experiment identity and dataset-file integrity preflight.

This module describes experiment settings; it does *not* implement image
preprocessing or execute models. A future dataset runner must call
``iter_verified_samples`` before loading examples, apply the exact documented
transform, and pass the resulting provenance IDs to metric/attack writers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Iterator

from src.experiments.dataset_manifests import sha256_file, verify_manifest

PROVENANCE_SCHEMA = 1
PREPROCESSING_SCHEMA = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _identity(value: dict) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _verified_path(root: Path, relative: str) -> Path:
    """Accept only dataset-root-relative files, including after symlink resolution."""
    if not isinstance(relative, str) or not relative:
        raise ValueError("Expected a nonempty dataset-relative path")
    parts = PurePosixPath(relative)
    if (parts.is_absolute() or relative.startswith("~") or "\\" in relative or
            any(part in {".", ".."} for part in relative.split("/")) or
            any(not part for part in relative.split("/"))):
        raise ValueError(f"Unsafe dataset path: {relative!r}")
    root = Path(root).resolve(strict=True)
    target = root.joinpath(*parts.parts).resolve(strict=True)
    if not target.is_relative_to(root):
        raise ValueError(f"Dataset path escapes root: {relative!r}")
    if not target.is_file():
        raise ValueError(f"Dataset entry is not a regular file: {relative!r}")
    return target


def verify_sample_files(sample: dict, dataset_root: Path) -> tuple[Path, Path | None]:
    """Verify *actual bytes* of RGB and optional target before GPU inference.

    A manifest's own checksum is independent of file contents. This separate
    check must be invoked again on resumed jobs, once per sample (not once
    per epsilon). No large RGB/depth tensors are retained by this function.
    """
    if not isinstance(sample, dict) or not isinstance(sample.get("image_id"), str) or not sample["image_id"]:
        raise ValueError("Invalid dataset sample")
    image_hash = sample.get("image_sha256")
    if not isinstance(image_hash, str) or not _SHA256.fullmatch(image_hash):
        raise ValueError("Missing or invalid frozen image SHA-256")
    image_path = _verified_path(dataset_root, sample.get("image"))
    if sha256_file(image_path) != image_hash:
        raise ValueError(f"Dataset image differs from frozen manifest: {sample['image_id']}")

    has_target = "target" in sample
    if has_target != ("target_sha256" in sample):
        raise ValueError("Target and target_sha256 must either both exist or both be absent")
    target_path = None
    if has_target:
        target_hash = sample["target_sha256"]
        if not isinstance(target_hash, str) or not _SHA256.fullmatch(target_hash):
            raise ValueError("Missing or invalid frozen target SHA-256")
        target_path = _verified_path(dataset_root, sample["target"])
        if sha256_file(target_path) != target_hash:
            raise ValueError(f"Dataset target differs from frozen manifest: {sample['image_id']}")
    return image_path, target_path


def verify_manifest_sources(manifest: dict, source_paths: dict[str, Path] | None = None) -> None:
    """Also verify external ImageNet ground-truth / NYUv2 extraction CSV.

    These files are not located relative to all dataset roots, so the caller
    must explicitly provide their paths rather than assuming a directory.
    """
    verify_manifest(manifest)
    expected = {key: value for key, value in manifest.get("source", {}).items()
                if key.endswith("_sha256")}
    provided = source_paths or {}
    if set(provided) != set(expected):
        raise ValueError(f"Required source paths: {sorted(expected)}")
    for key, digest in expected.items():
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"Invalid source hash: {key}")
        if sha256_file(Path(provided[key])) != digest:
            raise ValueError(f"Source file differs from frozen manifest: {key}")


def iter_verified_samples(
    manifest: dict,
    dataset_root: Path,
    *,
    source_paths: dict[str, Path] | None = None,
    image_ids: set[str] | None = None,
) -> Iterator[tuple[dict, Path, Path | None]]:
    """Yield selected samples only after integrity checks, for a future runner.

    One manifest/source verification is performed on starting the iterator;
    each yielded sample is independently checked immediately before use.
    """
    verify_manifest_sources(manifest, source_paths)
    rows = manifest["samples"]
    if image_ids is not None:
        if not isinstance(image_ids, set) or any(not isinstance(item, str) for item in image_ids):
            raise TypeError("image_ids must be a set of sample ID strings")
        found = {row["image_id"] for row in rows}
        if not image_ids <= found:
            raise ValueError("Requested sample IDs not in frozen manifest")
    for sample in rows:
        if image_ids is None or sample["image_id"] in image_ids:
            image, target = verify_sample_files(sample, dataset_root)
            yield sample, image, target


def make_preprocessing_spec(
    *,
    resize_mode: str,
    interpolation: str,
    rgb_normalization_mean: tuple[float, float, float],
    rgb_normalization_std: tuple[float, float, float],
    target_handling: str,
) -> dict:
    """Record an *explicit* 448x448 protocol; no implicit transform defaults.

    The actual transform is intentionally not implemented here. We must verify
    the Borg runner performs exactly these operations before GPU experiments.
    """
    if resize_mode not in {"stretch", "short_side_center_crop"}:
        raise ValueError("Unsupported RGB resize mode")
    if interpolation not in {"bilinear", "bicubic"}:
        raise ValueError("Unsupported RGB interpolation")
    if (not isinstance(target_handling, str) or not target_handling.strip() or
            any(placeholder in target_handling.lower() for placeholder in ("tbd", "unspecified", "unknown"))):
        raise ValueError("Target handling must be explicitly documented, without placeholders")
    mean, std = rgb_normalization_mean, rgb_normalization_std
    for values, label in ((mean, "mean"), (std, "std")):
        if len(values) != 3 or any(not isinstance(x, (int, float)) or
                                  isinstance(x, bool) or not math.isfinite(x) for x in values):
            raise ValueError(f"RGB normalization {label} must have three finite numbers")
    if any(value <= 0 for value in std):
        raise ValueError("RGB normalization std must be positive")
    body = {
        "schema_version": PREPROCESSING_SCHEMA,
        "input_mode": "RGB", "input_dtype": "float32",
        "input_range": "[0,1] (uint8 pixel value divided by 255)",
        "image_size_hw": [448, 448], "resize_mode": resize_mode,
        "interpolation": interpolation, "antialias": True,
        "normalization_mean": list(map(float, mean)),
        "normalization_std": list(map(float, std)),
        "target_handling": target_handling,
    }
    return {**body, "preprocessing_id": _identity(body)}


def _git_revision(repo_root: Path) -> str:
    root = Path(repo_root)
    def git(*args: str) -> str:
        proc = subprocess.run(["git", "-C", str(root), *args],
                              check=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        return proc.stdout.strip()
    try:
        revision = git("rev-parse", "--verify", "HEAD")
        if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
            raise ValueError("Invalid Git revision")
        if git("status", "--porcelain", "--untracked-files=normal"):
            raise ValueError("Working tree must be clean for reproducible provenance")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Cannot resolve Git revision") from exc
    return revision


def _validate_attack_settings(conditions: list[dict]) -> None:
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("Expected nonempty attack settings")
    seen = set()
    for attack in conditions:
        if not isinstance(attack, dict) or set(attack) != {"method", "objective", "epsilon", "seed", "alpha", "steps"}:
            raise ValueError("Attack settings must explicitly specify method, objective, epsilon, seed, alpha, steps")
        if attack["method"] not in {"pgd", "rfgsm"} or attack["objective"] not in {"cls", "patch", "gram"}:
            raise ValueError("Invalid attack method or objective")
        epsilon, alpha, steps, seed = (attack[field] for field in ("epsilon", "alpha", "steps", "seed"))
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("Attack seed must be integer")
        if (not isinstance(epsilon, (float, int)) or isinstance(epsilon, bool) or
                not math.isfinite(epsilon) or not 0 < epsilon <= 1):
            raise ValueError("Invalid attack epsilon")
        if attack["method"] == "pgd":
            if (not isinstance(alpha, (float, int)) or isinstance(alpha, bool) or
                    not math.isfinite(alpha) or not 0 < alpha <= epsilon or
                    not isinstance(steps, int) or isinstance(steps, bool) or steps < 1):
                raise ValueError("Invalid PGD alpha or steps")
        elif alpha is not None or steps is not None:
            raise ValueError("R-FGSM must have null alpha and steps")
        key = _canonical(attack)
        if key in seen:
            raise ValueError("Duplicate attack settings")
        seen.add(key)


def build_experiment_provenance(
    *,
    manifests: dict[str, dict],
    checkpoints: dict[str, Path],
    preprocessing: dict,
    attack_settings: list[dict],
    evaluation_protocol: dict,
    git_repo_root: Path,
) -> dict:
    """Generate immutable identifiers from source manifests, weights and code.

    Caller must also call ``iter_verified_samples`` before actual inference.
    Full checkpoint SHA-256 hashing is performed once when building this
    artifact, *not* on every attack or model forward pass.
    """
    if not manifests or not checkpoints or not attack_settings or not evaluation_protocol:
        raise ValueError("Provenance needs manifests, checkpoints, attacks and protocol")
    for manifest in manifests.values():
        verify_manifest(manifest)
    _validate_attack_settings(attack_settings)
    if not isinstance(preprocessing, dict) or preprocessing.get("schema_version") != PREPROCESSING_SCHEMA:
        raise ValueError("Invalid preprocessing specification")
    preprocessing_body = {key: value for key, value in preprocessing.items() if key != "preprocessing_id"}
    if preprocessing.get("preprocessing_id") != _identity(preprocessing_body):
        raise ValueError("Preprocessing specification was modified")
    if preprocessing_body.get("image_size_hw") != [448, 448]:
        raise ValueError("Experiment requires explicit 448x448 preprocessing")
    weights = {}
    for role, path in sorted(checkpoints.items()):
        if not isinstance(role, str) or not role:
            raise ValueError("Invalid checkpoint role")
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        weights[role] = {
            "filename": path.name, "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    body = {
        "schema_version": PROVENANCE_SCHEMA,
        "git_revision": _git_revision(git_repo_root),
        "dataset_manifests": {key: row["manifest_id"] for key, row in sorted(manifests.items())},
        "checkpoints": weights,
        "preprocessing": preprocessing,
        "attack_settings": attack_settings,
        "evaluation_protocol": evaluation_protocol,
    }
    result = {**body, "experiment_id": _identity(body)}
    _canonical(result)  # Reject NaN, non-JSON settings immediately.
    return result


def write_experiment_provenance(path: Path, provenance: dict) -> None:
    """Never overwrite an existing experiment configuration with new content."""
    if not isinstance(provenance, dict) or provenance.get("schema_version") != PROVENANCE_SCHEMA:
        raise ValueError("Invalid experiment provenance")
    body = {key: value for key, value in provenance.items() if key != "experiment_id"}
    if provenance.get("experiment_id") != _identity(body):
        raise ValueError("Experiment provenance checksum mismatch")
    payload = json.dumps(provenance, sort_keys=True, ensure_ascii=False,
                         indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write the complete file first, then atomically publish it without
    # overwriting a previous experiment's provenance. Survives SIGKILL
    # during JSON serialization without leaving a truncated final path.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                          dir=path.parent, prefix=path.name + ".",
                                          suffix=".tmp", delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp_path, path)  # Atomic publish, fails if path already exists.
        except FileExistsError:
            if path.read_text(encoding="utf-8") != payload:
                raise FileExistsError(f"Frozen experiment provenance differs: {path}")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
