"""Compact, restartable archive for pixel-domain adversarial perturbations.

This module only stores/loads attacks; experiment runners will be wired to it
in a later commit. Values use signed int8 *fixed-point* quantization, not fp8.

The original clean image is never saved. To replay an attack, reproduce the
same unnormalized [0, 1] float32 CHW tensor and verify its SHA-256 checksum.
Use a deterministic ``prefix`` for each non-overlapping work partition, not
a Slurm job ID (which changes on resubmission). On startup, committed shards
with that prefix are indexed, allowing completed attacks to be skipped. A
writer context also flushes completed attacks on ordinary Python exceptions.
SIGKILL, node loss or power loss can still discard the current buffer; for
expensive 7B attacks use ``shard_size=1`` or ``2`` initially, or call
``flush()`` after each costly unit of work. Restart does not automatically
resume a loop: the future runner must call ``contains`` BEFORE generating an
attack and re-use the same manifest, preprocessing, metadata and prefix.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np

FORMAT_VERSION = 1
_INT8_SCALE = 127
_REQUIRED_METADATA = (
    "dataset", "image_id", "model", "attack", "objective",
    "epsilon", "seed", "preprocessing", "checkpoint_id",
)


def _pixel_tensor(value, name: str) -> np.ndarray:
    """Convert a torch/numpy CHW pixel tensor to contiguous float32 on CPU."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"{name} must have CHW shape (3, H, W), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"{name} must be unnormalized pixel values in [0, 1]")
    return np.ascontiguousarray(arr)


def clean_input_sha256(x_clean) -> str:
    """Hash *preprocessed*, unnormalized CHW float32 inputs, not source JPEGs."""
    return hashlib.sha256(_pixel_tensor(x_clean, "x_clean").tobytes()).hexdigest()


def _metadata_json(metadata: dict) -> str:
    try:
        return json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must contain JSON-safe finite values") from error


def _validated_metadata(metadata: dict) -> dict:
    """Canonicalize the inputs used to identify an adversarial experiment."""
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")
    missing = [key for key in _REQUIRED_METADATA if key not in metadata]
    if missing:
        raise ValueError(f"Missing attack metadata fields: {missing}")
    if any(not isinstance(metadata[key], str) or not metadata[key] for key in
           ("dataset", "image_id", "model", "attack", "objective", "preprocessing", "checkpoint_id")):
        raise ValueError("dataset, image_id, model, attack, objective, preprocessing, checkpoint_id must be nonempty strings")

    try:
        epsilon = float(metadata["epsilon"])
        seed = int(metadata["seed"])
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("epsilon and seed must be numeric") from error
    if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be finite and in (0, 1]")
    if not isinstance(metadata["seed"], (int, np.integer)) or isinstance(metadata["seed"], bool):
        raise ValueError("seed must be an integer")

    canonical = dict(metadata, epsilon=epsilon, seed=seed)
    _metadata_json(canonical)
    return canonical


def attack_key_for_input(x_clean, metadata: dict) -> str:
    """Stable attack identity, usable to skip *before* expensive PGD."""
    canonical = _validated_metadata(metadata)
    identity_json = _metadata_json({
        "metadata": canonical,
        "clean_input_sha256": clean_input_sha256(x_clean),
    })
    return hashlib.sha256(identity_json.encode("utf-8")).hexdigest()


def quantize_attack(x_clean, x_adv, metadata: dict) -> tuple[np.ndarray, dict]:
    """Represent x_adv - x_clean as epsilon-scaled, signed int8 values.

    Return (q_delta, record), where record contains enough information to
    check clean-input identity and reconstruct the quantized attacked input.
    """
    canonical = _validated_metadata(metadata)
    clean = _pixel_tensor(x_clean, "x_clean")
    adv = _pixel_tensor(x_adv, "x_adv")
    if clean.shape != adv.shape:
        raise ValueError("x_clean and x_adv must have identical CHW shapes")
    epsilon = canonical["epsilon"]
    delta = adv - clean
    actual_linf = float(np.max(np.abs(delta)))
    if actual_linf > epsilon + 1e-6:
        raise ValueError(f"Attack exceeds L_inf budget: {actual_linf} > {epsilon}")

    q = np.rint(delta.astype(np.float64) * (_INT8_SCALE / epsilon))
    q = np.clip(q, -_INT8_SCALE, _INT8_SCALE).astype(np.int8)
    decoded = np.clip(clean + q.astype(np.float32) * np.float32(epsilon / _INT8_SCALE), 0.0, 1.0)
    max_error = float(np.max(np.abs(decoded - adv)))
    if max_error > epsilon / (2 * _INT8_SCALE) + 3e-7:
        raise AssertionError(f"Unexpected quantization error: {max_error}")

    record = dict(canonical)
    record["shape_chw"] = list(clean.shape)
    record["clean_input_sha256"] = hashlib.sha256(clean.tobytes()).hexdigest()
    record["original_linf"] = actual_linf
    record["quantization_max_error"] = max_error
    record["storage_encoding"] = "signed_int8_delta_epsilon_over_127"
    # Derive a stable key for interrupted-run recovery. Metadata should not
    # contain volatile fields such as timestamps or run durations.
    record["attack_key"] = attack_key_for_input(clean, canonical)
    _metadata_json(record)
    return q, record


def reconstruct_attack(x_clean, q_delta: np.ndarray, record: dict) -> np.ndarray:
    """Reconstruct the quantized attacked tensor; reject wrong clean inputs."""
    clean = _pixel_tensor(x_clean, "x_clean")
    q = np.asarray(q_delta)
    if q.dtype != np.int8 or q.shape != clean.shape:
        raise ValueError("q_delta must be signed int8 and match clean CHW shape")
    if list(clean.shape) != record["shape_chw"]:
        raise ValueError("Clean input shape does not match archive")
    if hashlib.sha256(clean.tobytes()).hexdigest() != record["clean_input_sha256"]:
        raise ValueError("Clean input checksum mismatch: check image and preprocessing")
    if record.get("storage_encoding") != "signed_int8_delta_epsilon_over_127":
        raise ValueError("Unsupported perturbation encoding")
    epsilon = float(record["epsilon"])
    if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        raise ValueError("Invalid epsilon stored in archive")
    attacked = np.clip(clean + q.astype(np.float32) * np.float32(epsilon / _INT8_SCALE), 0, 1)
    return attacked.astype(np.float32, copy=False)


def load_shard(path: str | Path) -> list[tuple[dict, np.ndarray]]:
    """Return (record, q_delta) pairs; no pickle or clean images are loaded."""
    with np.load(path, allow_pickle=False) as data:
        if int(data["format_version"]) != FORMAT_VERSION:
            raise ValueError("Unsupported archive format version")
        records = json.loads(str(data["metadata_json"].item()))
        deltas = data["q_deltas"]
    if deltas.dtype != np.int8 or deltas.ndim != 4 or len(records) != len(deltas):
        raise ValueError("Invalid perturbation shard")
    return list(zip(records, deltas))


def load_shard_metadata(path: str | Path) -> list[dict]:
    """Read identities without loading/decompressing perturbation arrays."""
    with np.load(path, allow_pickle=False) as data:
        if int(data["format_version"]) != FORMAT_VERSION:
            raise ValueError("Unsupported archive format version")
        return json.loads(str(data["metadata_json"].item()))


class PerturbationShardWriter:
    """Incremental shard writer with duplicate skipping and atomic commits.

    Each stable prefix belongs to one non-overlapping work partition. Existing
    shards with that prefix are indexed on startup. On restart the caller must
    iterate the same manifest and call ``contains`` before each expensive
    attack. Avoid parallel writers to the same prefix: this class does not
    provide a distributed lock or global cross-prefix deduplication.

    Validated records are flushed even when the context exits with a Python
    exception. For abrupt termination, only already committed shards survive.
    The default shard size is 2 for cautious initial runs. Use 1 for maximum
    per-attack durability or explicitly call ``flush`` after an expensive
    unit of work. Larger sizes reduce file count for full-scale runs.
    """

    def __init__(self, directory: str | Path, prefix: str, shard_size: int = 2):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix):
            raise ValueError("prefix must be a filesystem-safe worker identifier")
        if not isinstance(shard_size, int) or shard_size < 1:
            raise ValueError("shard_size must be a positive integer")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_size = shard_size
        self._buffer: list[tuple[np.ndarray, dict]] = []
        self._keys: set[str] = set()
        self._locations: dict[str, tuple[Path, int]] = {}
        self._next_index = 0
        pattern = re.compile(rf"{re.escape(prefix)}-([0-9]{{6}})\.npz")
        for path in self.directory.glob(f"{prefix}-*.npz"):
            match = pattern.fullmatch(path.name)
            if match:
                self._next_index = max(self._next_index, int(match.group(1)) + 1)
                for offset, record in enumerate(load_shard_metadata(path)):
                    key = record["attack_key"]
                    if key in self._keys:
                        raise ValueError(f"Duplicate attack key in existing shards: {key}")
                    self._keys.add(key)
                    self._locations[key] = (path, offset)

    def contains(self, x_clean, metadata: dict) -> bool:
        """Check for completed attacks before doing a costly forward/backward pass."""
        return attack_key_for_input(x_clean, metadata) in self._keys

    def load(self, attack_key: str) -> tuple[dict, np.ndarray]:
        """Read an already committed attack by key, without regeneration.

        Buffered attacks are not durable; callers must flush before replay.
        """
        if attack_key not in self._locations:
            raise KeyError(f"Attack not committed in this prefix: {attack_key}")
        path, offset = self._locations[attack_key]
        record, q = load_shard(path)[offset]
        if record.get("attack_key") != attack_key:
            raise ValueError("Archive attack index mismatch")
        return record, q

    def add(self, x_clean, x_adv, metadata: dict) -> bool:
        """Add a new attack, returning False when already archived."""
        q, record = quantize_attack(x_clean, x_adv, metadata)
        key = record["attack_key"]
        if key in self._keys:
            return False
        if self._buffer and self._buffer[0][0].shape != q.shape:
            self.flush()
        self._buffer.append((q, record))
        self._keys.add(key)
        if len(self._buffer) >= self.shard_size:
            self.flush()
        return True

    def flush(self) -> Path | None:
        """Write a complete shard, never overwriting an existing shard."""
        if not self._buffer:
            return None
        destination = self.directory / f"{self.prefix}-{self._next_index:06d}.npz"
        deltas = np.stack([q for q, _ in self._buffer])
        records = [record for _, record in self._buffer]
        fd, temporary = tempfile.mkstemp(prefix=f".{self.prefix}-", suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(
                    stream,
                    format_version=np.asarray(FORMAT_VERSION, dtype=np.uint8),
                    q_deltas=deltas,
                    metadata_json=np.asarray(_metadata_json(records)),
                )
                stream.flush()
                os.fsync(stream.fileno())
            # Atomic creation on the same filesystem. A competing writer
            # cannot replace or silently overwrite an existing shard.
            os.link(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        for offset, record in enumerate(records):
            self._locations[record["attack_key"]] = (destination, offset)
        self._next_index += 1
        self._buffer.clear()
        return destination

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Only successfully validated attacks enter _buffer. Preserve those
        # even when a later experiment raises: do not discard expensive PGD.
        # We deliberately return False so the original exception propagates.
        self.flush()
        return False
