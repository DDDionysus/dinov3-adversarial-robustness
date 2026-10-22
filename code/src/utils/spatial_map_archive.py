"""Compact per-patch spatial map archives, keyed by committed attack_key.

Maps describe the exact reconstructed int8 attack, not the transient float32
PGD output. Two maps are saved per attack: float32 corresponding-patch cosine
similarity and float16 RMS change of each row of the normalized patch Gram matrix.
Store neither original images, P x D embeddings nor P x P Gram matrices.

One writer per stable, nonoverlapping partition prefix; no global locking.
Buffers are flushed on normal/exceptional context exit and on shard size.
After an abrupt kill, missing maps are regenerated from the *already archived*
attack at restart; the expensive PGD optimization is never repeated.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

FORMAT_VERSION = 2
_ENCODING = "float32_patch_cosine_float16_gram_row_rmse_v2"
_KEY_PATTERN = re.compile(r"[0-9a-f]{64}")


def _digest(cosine: np.ndarray, gram_row: np.ndarray) -> str:
    return hashlib.sha256(cosine.tobytes() + gram_row.tobytes()).hexdigest()


def _validated_maps(attack_key: str, maps: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    if not isinstance(attack_key, str) or _KEY_PATTERN.fullmatch(attack_key) is None:
        raise ValueError("attack_key must be a 64-character lowercase hexadecimal digest")
    if not isinstance(maps, dict) or set(maps) != {"patch_similarity", "gram_row_rmse"}:
        raise ValueError("Expected patch_similarity and gram_row_rmse maps")
    cosine = np.asarray(maps["patch_similarity"])
    gram_row = np.asarray(maps["gram_row_rmse"])
    if cosine.ndim != 2 or cosine.shape != gram_row.shape or cosine.size < 4:
        raise ValueError("Both maps must be matching 2D patch grids with at least 4 patches")
    if cosine.shape[0] != cosine.shape[1]:
        raise ValueError("Current protocol requires a square patch grid")
    if cosine.size > 16384:
        raise ValueError("Unexpectedly large spatial map")
    if not np.issubdtype(cosine.dtype, np.floating) or not np.issubdtype(gram_row.dtype, np.floating):
        raise ValueError("Spatial maps must be real floating-point arrays")
    if not np.isfinite(cosine).all() or not np.isfinite(gram_row).all():
        raise ValueError("Spatial maps must be finite")
    if np.any(cosine < -1.00001) or np.any(cosine > 1.00001):
        raise ValueError("Patch cosine values must fall in [-1, 1]")
    if np.any(gram_row < 0):
        raise ValueError("Gram-row RMS values must be nonnegative")

    # Small numerical deviations outside [-1,1] may occur in float32 cosine.
    # Preserve near-unity differences (e.g. 0.9998 vs. 1.0).
    cosine32 = np.clip(cosine, -1.0, 1.0).astype(np.float32)
    gram16 = gram_row.astype(np.float16)
    if not np.isfinite(gram16).all():
        raise ValueError("Gram-row values exceed float16's range")
    row = {
        "attack_key": attack_key,
        "grid_hw": [int(cosine.shape[0]), int(cosine.shape[1])],
        "encoding": _ENCODING,
        "sha256": _digest(cosine32, gram16),
    }
    return cosine32, gram16, row


def _load_metadata(path: Path) -> list[dict]:
    with np.load(path, allow_pickle=False) as data:
        if int(data["format_version"]) != FORMAT_VERSION:
            raise ValueError("Unknown spatial map format")
        rows = json.loads(str(data["metadata_json"].item()))
    if not isinstance(rows, list):
        raise ValueError("Invalid spatial map metadata")
    for row in rows:
        if (not isinstance(row, dict) or
                set(row) != {"attack_key", "grid_hw", "encoding", "sha256"} or
                _KEY_PATTERN.fullmatch(row["attack_key"]) is None or
                row["encoding"] != _ENCODING):
            raise ValueError("Malformed spatial map metadata")
    return rows


def _load_shard(path: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if int(data["format_version"]) != FORMAT_VERSION:
            raise ValueError("Unknown spatial map format")
        rows = json.loads(str(data["metadata_json"].item()))
        cosines = data["patch_similarity"]
        grams = data["gram_row_rmse"]
    if (not isinstance(rows, list) or cosines.dtype != np.float32 or
            grams.dtype != np.float16 or cosines.ndim != 3 or
            cosines.shape != grams.shape or len(rows) != len(cosines)):
        raise ValueError("Malformed spatial map shard")
    for row, cosine, gram in zip(rows, cosines, grams):
        if (row["encoding"] != _ENCODING or list(cosine.shape) != row["grid_hw"] or
                row["sha256"] != _digest(cosine, gram)):
            raise ValueError("Spatial map checksum/shape mismatch")
    return rows, cosines, grams


class SpatialMapShardWriter:
    """Incremental restartable float32 cosine / float16 Gram-row map writer.

    With shards > 1, a sudden kill may discard buffered maps; because attacks
    themselves are committed first, missing maps can be reconstructed by
    replaying those attacks without repeating PGD. Use shard_size=1 if each
    computed map must be immediately durable during initial 7B validation.
    """

    def __init__(self, directory: str | Path, prefix: str, shard_size: int = 8):
        if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix):
            raise ValueError("prefix must be filesystem-safe")
        if not isinstance(shard_size, int) or isinstance(shard_size, bool) or shard_size < 1:
            raise ValueError("shard_size must be a positive integer")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_size = shard_size
        self._buffer: list[tuple[np.ndarray, np.ndarray, dict]] = []
        self._keys: set[str] = set()
        self._locations: dict[str, tuple[Path, int]] = {}
        # Existing shard payloads are verified lazily, once per writer lifetime.
        # Shards are immutable after publication; do not mutate live archives.
        self._verified_shards: set[Path] = set()
        self._next_index = 0
        pattern = re.compile(rf"{re.escape(prefix)}-([0-9]{{6}})\.npz")
        for path in sorted(self.directory.glob(f"{prefix}-*.npz")):
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            self._next_index = max(self._next_index, int(match.group(1)) + 1)
            for offset, row in enumerate(_load_metadata(path)):
                key = row["attack_key"]
                if key in self._keys:
                    raise ValueError("Duplicate attack_key across map shards")
                self._keys.add(key)
                self._locations[key] = (path, offset)

    def _verify_saved_shard(self, path: Path) -> None:
        """Reject payload corruption before a stored key can be skipped.

        One digest check covers every map in the shard. Once checked, cache
        the result for this writer instance to avoid rereading shard data on
        every identity lookup; reopening the writer revalidates on access.
        """
        if path in self._verified_shards:
            return
        rows, _, _ = _load_shard(path)
        for offset, row in enumerate(rows):
            if self._locations.get(row["attack_key"]) != (path, offset):
                raise ValueError("Spatial map shard index mismatch")
        self._verified_shards.add(path)

    def contains(self, attack_key: str) -> bool:
        if not isinstance(attack_key, str) or _KEY_PATTERN.fullmatch(attack_key) is None:
            raise ValueError("Invalid attack_key")
        if attack_key in self._locations:
            self._verify_saved_shard(self._locations[attack_key][0])
        return attack_key in self._keys

    def add(self, attack_key: str, maps: dict) -> bool:
        cosine, gram, row = _validated_maps(attack_key, maps)
        if attack_key in self._keys:
            if attack_key in self._locations:
                existing = self.load(attack_key)
                existing_cos, existing_gram = existing["patch_similarity"], existing["gram_row_rmse"]
            else:
                existing_cos, existing_gram, _ = next(
                    entry for entry in self._buffer if entry[2]["attack_key"] == attack_key
                )
            if not np.array_equal(cosine, existing_cos) or not np.array_equal(gram, existing_gram):
                raise ValueError("Conflicting spatial maps for previously saved attack_key")
            return False
        if self._buffer and self._buffer[0][0].shape != cosine.shape:
            self.flush()
        self._buffer.append((cosine, gram, row))
        self._keys.add(attack_key)
        if len(self._buffer) >= self.shard_size:
            self.flush()
        return True

    def load(self, attack_key: str) -> dict[str, np.ndarray]:
        if attack_key not in self._locations:
            raise KeyError("Map is not committed for attack_key")
        path, offset = self._locations[attack_key]
        rows, cosines, grams = _load_shard(path)
        if rows[offset]["attack_key"] != attack_key:
            raise ValueError("Spatial map index mismatch")
        return {"patch_similarity": cosines[offset].copy(),
                "gram_row_rmse": grams[offset].copy()}

    def flush(self) -> Path | None:
        if not self._buffer:
            return None
        destination = self.directory / f"{self.prefix}-{self._next_index:06d}.npz"
        records = [row for _, _, row in self._buffer]
        cosines = np.stack([c for c, _, _ in self._buffer])
        grams = np.stack([g for _, g, _ in self._buffer])
        metadata_json = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.prefix}-", suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, format_version=np.asarray(FORMAT_VERSION, dtype=np.uint8),
                                    patch_similarity=cosines, gram_row_rmse=grams,
                                    metadata_json=np.asarray(metadata_json))
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        for offset, row in enumerate(records):
            self._locations[row["attack_key"]] = destination, offset
        self._next_index += 1
        self._buffer.clear()
        return destination

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.flush()
        return False
