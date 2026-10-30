"""Durable per-image evaluation records and scientifically correct summaries.

This is an I/O and aggregation layer, NOT a model evaluator. A later commit
will extract CLS/patch/Gram, classification, segmentation and depth statistics
from model predictions and create records using ``make_metric_record``.

Write one JSONL file per stable, non-overlapping work partition. Each record
is flushed and fsynced before ``add`` returns, avoiding expensive recomputation
if a later job fails. Only one writer may hold a file at a time (fcntl lock).
Do not change partition assignments on resume; aggregate across files with
``read_metric_records`` to detect duplicate sample/condition evaluations.

JSONL deliberately avoids pyarrow/pandas as new runtime dependencies. Values
must be small, finite scalars or short numeric lists (e.g. per-class counts):
no images, predictions, embeddings or full pixel arrays are stored here.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np

SCHEMA_VERSION = 1
_TASKS = {"representation", "classification", "segmentation", "depth"}
_IDENTITY_FIELDS = (
    "dataset", "image_id", "model", "checkpoint_id", "task",
    "protocol_id", "preprocessing", "condition_id", "attack_key",
)
_SCOPE_FIELDS = (
    "dataset", "model", "checkpoint_id", "task", "protocol_id",
    "preprocessing", "condition_id",
)
_SAMPLE_FIELDS = tuple(field for field in _IDENTITY_FIELDS
                       if field not in ("condition_id", "attack_key"))
_MAX_LIST_LENGTH = 1024  # ADE20K's 150 per-class counts fit; pixel maps do not.


def _json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _metric_value(value, name: str):
    if isinstance(value, np.ndarray):
        raise TypeError(f"{name} cannot be an array; save only aggregate metrics")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, complex):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_LIST_LENGTH:
            raise ValueError(f"{name} has too many elements")
        return [_metric_scalar(item, name) for item in value]
    raise TypeError(f"{name} must be a finite scalar or short numeric list")


def _metric_scalar(value, name: str):
    item = _metric_value(value, name)
    if isinstance(item, list):
        raise TypeError(f"{name} must be a flat list")
    return item


def _identity(row: dict) -> dict:
    return {key: row[key] for key in _IDENTITY_FIELDS}


def _validate_identity(identity: dict) -> None:
    """Validate an experiment identity without requiring calculated metrics."""
    if set(identity) != set(_IDENTITY_FIELDS):
        raise ValueError("Identity must specify all experiment identity fields")
    for field in _IDENTITY_FIELDS:
        if field != "attack_key" and (not isinstance(identity[field], str) or not identity[field]):
            raise ValueError(f"{field} must be a nonempty string")
    if identity["task"] not in _TASKS:
        raise ValueError("Unknown task")
    if identity["condition_id"] == "clean":
        if identity["attack_key"] is not None:
            raise ValueError("Clean records must not reference an attack")
    elif not isinstance(identity["attack_key"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", identity["attack_key"]):
        raise ValueError("Attacked records require their 64-character archive attack_key")


def _record_id(identity: dict) -> str:
    _validate_identity(identity)
    return hashlib.sha256(_json({"version": SCHEMA_VERSION, **identity}).encode()).hexdigest()


def _validate_record(row: dict) -> dict:
    if not isinstance(row, dict):
        raise TypeError("Metric record must be a dictionary")
    required = {"schema_version", "record_id", "metrics", *_IDENTITY_FIELDS}
    if set(row) != required:
        raise ValueError(f"Metric record fields must be {sorted(required)}")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported metric schema version")
    identity = _identity(row)
    digest = _record_id(identity)
    if not isinstance(row["metrics"], dict) or not row["metrics"]:
        raise ValueError("metrics must be a nonempty dictionary")
    metrics = {}
    for name, value in row["metrics"].items():
        if not isinstance(name, str) or not name:
            raise ValueError("Metric names must be nonempty strings")
        metrics[name] = _metric_value(value, name)
    if row["record_id"] != digest:
        raise ValueError("Metric record identity checksum mismatch")
    return {**row, "metrics": metrics}

def make_metric_record(*, dataset: str, image_id: str, model: str,
                       checkpoint_id: str, task: str, protocol_id: str,
                       preprocessing: str, condition_id: str, metrics: dict,
                       attack_key: str | None = None) -> dict:
    """Construct a versioned record; ``condition_id`` must encode all settings.

    Example condition_id: 'pgd.cls.eps_2_255.seed42.steps10'. For a clean
    prediction use 'clean' and attack_key=None. Include data selection, target
    resolution, crop, interpolation, alignment/masking in protocol_id or a
    versioned referenced specification; changing that requires a new ID.
    """
    row = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset, "image_id": image_id, "model": model,
        "checkpoint_id": checkpoint_id, "task": task,
        "protocol_id": protocol_id, "preprocessing": preprocessing,
        "condition_id": condition_id, "attack_key": attack_key,
        "metrics": metrics,
    }
    row["record_id"] = _record_id(_identity(row))
    return _validate_record(row)


def _slot(row: dict) -> tuple:
    # Protect against two different attack_keys in the same condition/image.
    return tuple(row[field] for field in _SAMPLE_FIELDS) + (row["condition_id"],)


def _index_one(row: dict, by_id: dict, by_slot: dict) -> bool:
    row = _validate_record(row)
    key = row["record_id"]
    if key in by_id:
        if by_id[key] != _json(row):
            raise ValueError(f"Conflicting metrics for existing record {key}")
        return False
    slot = _slot(row)
    if slot in by_slot:
        raise ValueError(f"Multiple attacks for the same sample/condition: {slot}")
    by_id[key] = _json(row)
    by_slot[slot] = key
    return True


class MetricJSONLWriter:
    """Append one fully validated, fsynced record at a time; restart-safe.

    The same partition must use the same path on restart. A nonblocking file
    lock prevents two simultaneous writers to one file. A crash-truncated final
    line is removed on restart; malformed complete lines fail loudly.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = open(self.path, "a+b")
        self._by_id: dict[str, str] = {}
        self._by_slot: dict[tuple, str] = {}
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._stream.seek(0)
            while True:
                start = self._stream.tell()
                line = self._stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    # Only the final, non-newline-terminated record is unsafe.
                    self._stream.truncate(start)
                    self._stream.flush()
                    os.fsync(self._stream.fileno())
                    break
                row = json.loads(line)
                if not _index_one(row, self._by_id, self._by_slot):
                    raise ValueError("Duplicate records already present in metric file")
            self._stream.seek(0, os.SEEK_END)
        except BaseException:
            self.close()
            raise

    def contains(self, record: dict) -> bool:
        """Check a complete record, rejecting conflicts with committed results."""
        row = _validate_record(record)
        previous = self._by_id.get(row["record_id"])
        if previous is not None:
            if previous != _json(row):
                raise ValueError("Previously committed metrics differ for this record")
            return True
        if _slot(row) in self._by_slot:
            raise ValueError("Sample/condition already evaluated with another attack_key")
        return False

    def contains_identity(self, *, dataset: str, image_id: str, model: str,
                          checkpoint_id: str, task: str, protocol_id: str,
                          preprocessing: str, condition_id: str,
                          attack_key: str | None = None) -> bool:
        """Check before computing predictions; no metrics dictionary is needed.

        Uses the exact identity that make_metric_record() uses. A sample and
        condition already stored with a *different* attack_key is an error,
        not permission to recompute and silently replace the attack.
        This lookup checks only this writer's JSONL partition.
        """
        identity = {
            "dataset": dataset, "image_id": image_id, "model": model,
            "checkpoint_id": checkpoint_id, "task": task,
            "protocol_id": protocol_id, "preprocessing": preprocessing,
            "condition_id": condition_id, "attack_key": attack_key,
        }
        key = _record_id(identity)
        if key in self._by_id:
            return True
        if _slot(identity) in self._by_slot:
            raise ValueError("Sample/condition already evaluated with another attack_key")
        return False

    def add(self, record: dict) -> bool:
        row = _validate_record(record)
        key = row["record_id"]
        # Check without recording the identity until data is durable.
        if key in self._by_id:
            if self._by_id[key] != _json(row):
                raise ValueError("Previously committed metrics differ; change protocol_id")
            return False
        if _slot(row) in self._by_slot:
            raise ValueError("Sample/condition already evaluated with another attack_key")
        self._stream.write((_json(row) + "\n").encode("utf-8"))
        self._stream.flush()
        os.fsync(self._stream.fileno())
        _index_one(row, self._by_id, self._by_slot)
        return True

    def close(self):
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def read_metric_records(paths) -> list[dict]:
    """Load result partitions; reject duplicates across workers/files.

    Accepted inputs: an iterable of paths, or one path. Do not blindly
    concatenate or average metrics from potentially overlapping partitions.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rows = []
    by_id, by_slot = {}, {}
    for path in sorted(map(Path, paths)):
        with open(path, "rb") as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.endswith(b"\n"):
                    raise ValueError(f"Incomplete metric record: {path}:{lineno}")
                row = _validate_record(json.loads(line))
                if not _index_one(row, by_id, by_slot):
                    raise ValueError(f"Duplicate metric record across files: {path}:{lineno}")
                rows.append(row)
    return rows


def _consistent_condition(rows: list[dict]) -> tuple:
    if not rows:
        raise ValueError("No metric records")
    scope = tuple(rows[0][field] for field in _SCOPE_FIELDS)
    slots = set()
    for row in rows:
        _validate_record(row)
        if tuple(row[field] for field in _SCOPE_FIELDS) != scope:
            raise ValueError("Do not combine different datasets/models/protocols/conditions")
        sample = tuple(row[field] for field in _SAMPLE_FIELDS)
        if sample in slots:
            raise ValueError("Duplicate sample in metric distribution")
        slots.add(sample)
    return scope


def _distribution(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Distribution requires nonempty finite scalar values")
    p10, p25, p50, p75, p90 = np.percentile(array, [10, 25, 50, 75, 90])
    return {
        "n": int(array.size), "mean": float(array.mean()),
        "std": float(array.std(ddof=0)), "min": float(array.min()),
        "p10": float(p10), "p25": float(p25), "median": float(p50),
        "p75": float(p75), "p90": float(p90), "max": float(array.max()),
    }


def metric_distribution(rows: list[dict], metric: str) -> dict:
    """Mean, spread, percentiles for ONE dataset/model/protocol/condition."""
    _consistent_condition(rows)
    values = []
    for row in rows:
        value = row["metrics"].get(metric)
        if isinstance(value, list) or not isinstance(value, (float, int, bool)):
            raise ValueError(f"Missing or non-scalar metric {metric}")
        values.append(float(value))
    return _distribution(values)


def paired_metric_distribution(clean: list[dict], attacked: list[dict],
                               metric: str, *, lower_is_better: bool = False) -> dict:
    """Describe per-image adv-clean deltas; fail on missing/misaligned pairs.

    ``fraction_worsened`` is computed using the metric's direction. No
    bootstrap assumptions or unpaired group averages are hidden in this API.
    """
    clean_scope = _consistent_condition(clean)
    attacked_scope = _consistent_condition(attacked)
    if clean_scope[:-1] != attacked_scope[:-1]:
        raise ValueError("Clean/attacked evaluation protocols or models differ")
    if clean_scope[-1] != "clean" or attacked_scope[-1] == "clean":
        raise ValueError("Expected a clean condition and one attack condition")
    c_map = {tuple(r[k] for k in _SAMPLE_FIELDS): r for r in clean}
    a_map = {tuple(r[k] for k in _SAMPLE_FIELDS): r for r in attacked}
    if c_map.keys() != a_map.keys():
        raise ValueError("Clean/attacked sample IDs do not match")
    deltas = []
    for key in sorted(a_map):
        c, a = c_map[key]["metrics"].get(metric), a_map[key]["metrics"].get(metric)
        if not isinstance(c, (bool, int, float)) or not isinstance(a, (bool, int, float)):
            raise ValueError(f"Missing or non-scalar paired metric {metric}")
        deltas.append(float(a) - float(c))
    stats = _distribution(deltas)
    values = np.asarray(deltas)
    stats["fraction_worsened"] = float(np.mean(values > 0 if lower_is_better else values < 0))
    return stats


def dataset_segmentation_miou(rows: list[dict]) -> dict:
    """Compute dataset mIoU using SUMMED class intersections/unions.

    Metric fields are per-image ``intersection`` and ``union`` integer lists.
    Absent classes (zero union over the entire dataset) are excluded.
    """
    _consistent_condition(rows)
    if rows[0]["task"] != "segmentation":
        raise ValueError("Expected segmentation records")
    total_i = total_u = None
    for row in rows:
        i = np.asarray(row["metrics"].get("intersection"))
        u = np.asarray(row["metrics"].get("union"))
        if i.ndim != 1 or i.size == 0 or i.dtype.kind not in "iu" or u.shape != i.shape or u.dtype.kind not in "iu":
            raise ValueError("intersection and union must be equally sized integer lists")
        if (i < 0).any() or (u < i).any():
            raise ValueError("Expected 0 <= intersection <= union")
        if total_i is None:
            total_i = np.zeros(i.shape, dtype=np.int64)
            total_u = np.zeros(u.shape, dtype=np.int64)
        if i.shape != total_i.shape:
            raise ValueError("Segmentation class count differs between images")
        total_i += i
        total_u += u
    present = total_u > 0
    if not present.any():
        raise ValueError("No evaluable segmentation classes")
    return {
        "images": len(rows), "present_classes": int(present.sum()),
        "miou": float(np.mean(total_i[present] / total_u[present])),
        "class_iou": [(float(i / u) if u else None) for i, u in zip(total_i, total_u)],
    }


def dataset_depth_metrics(rows: list[dict]) -> dict:
    """Pixel-weighted depth scores from per-image additive sufficient stats.

    Fields: ``valid_pixel_count``, ``abs_rel_sum``, ``sq_error_sum``,
    ``delta1_count``. Apply the agreed valid-pixel mask, crop, and (if used)
    scale alignment BEFORE computing these values. Specify that protocol in
    protocol_id; this method does not silently align predictions.
    """
    _consistent_condition(rows)
    if rows[0]["task"] != "depth":
        raise ValueError("Expected depth records")
    n = 0
    ar = se = 0.0
    d1 = 0
    for row in rows:
        m = row["metrics"]
        count = m.get("valid_pixel_count")
        passed = m.get("delta1_count")
        if (not isinstance(count, int) or isinstance(count, bool) or count < 0 or
                not isinstance(passed, int) or isinstance(passed, bool) or not 0 <= passed <= count):
            raise ValueError("Invalid depth pixel counts")
        abs_rel_sum, sq_error_sum = m.get("abs_rel_sum"), m.get("sq_error_sum")
        if (not isinstance(abs_rel_sum, (int, float)) or not isinstance(sq_error_sum, (int, float)) or
                abs_rel_sum < 0 or sq_error_sum < 0):
            raise ValueError("Invalid depth error sums")
        n += count
        d1 += passed
        ar += abs_rel_sum
        se += sq_error_sum
    if n == 0:
        raise ValueError("No valid depth pixels")
    return {"images": len(rows), "valid_pixels": n,
            "abs_rel": float(ar / n), "rmse": float(math.sqrt(se / n)),
            "delta1": float(d1 / n)}
