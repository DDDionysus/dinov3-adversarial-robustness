"""Portable, content-verified manifests for the agreed DINO robustness datasets.

This module only SELECTS and FREEZES samples. The later dataset runner must
load these manifests, apply the versioned preprocessing protocol, and map
ImageNet ILSVRC labels to the official head's output index ordering.

Paths in manifests are POSIX paths relative to the corresponding dataset root;
no workstation/Borg absolute paths, job IDs, or timestamps enter identities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
from collections import Counter, defaultdict


MANIFEST_SCHEMA = 1
SELECTION_SCHEME = "sha256-class-seed-python-random-v1"


def _canonical(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _manifest(dataset: str, split: str, selection: dict, samples: list[dict],
              source: dict) -> dict:
    result = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset": dataset,
        "split": split,
        "selection": selection,
        "source": source,
        "sample_count": len(samples),
        "samples": samples,
    }
    result["manifest_id"] = "sha256:" + _digest_bytes(_canonical(result))
    return result


def verify_manifest(manifest: dict) -> None:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("Unsupported dataset manifest schema")
    body = {key: value for key, value in manifest.items() if key != "manifest_id"}
    if manifest.get("manifest_id") != "sha256:" + _digest_bytes(_canonical(body)):
        raise ValueError("Dataset manifest checksum mismatch")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples or manifest.get("sample_count") != len(samples):
        raise ValueError("Invalid manifest sample count")
    ids = [row["image_id"] for row in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs in dataset manifest")


def _require_file(root: Path, relative: str) -> Path:
    # A manifest must not contain paths capable of escaping its dataset root.
    posix = PurePosixPath(relative)
    if (not relative or posix.is_absolute() or ".." in posix.parts or
            "." in posix.parts or "\\" in relative):
        raise ValueError(f"Unsafe dataset-relative path: {relative!r}")
    path = root.joinpath(*posix.parts)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path


def imagenet_manifests(val_dir: Path, label_file: Path, *, seed: int,
                       classes: int = 1000, examples_per_class: int = 50) -> tuple[dict, dict]:
    """Select two/class for 7B; the first of each pair is the ViT-L subset.

    Class IDs are the original 1-based ILSVRC2012 IDs, NOT assumed to be the
    DINOv3 classifier's 0-based output indices. The full validation ground
    truth is required even though most images are not selected.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Subset seed must be a nonnegative integer")
    if classes < 1 or examples_per_class < 2:
        raise ValueError("Invalid ImageNet class counts")
    val_dir, label_file = Path(val_dir), Path(label_file)
    text = label_file.read_text(encoding="utf-8")
    try:
        labels = [int(line.strip()) for line in text.splitlines()]
    except ValueError as exc:
        raise ValueError("ImageNet labels must be integer ILSVRC class IDs") from exc
    if (len(labels) != classes * examples_per_class or
            Counter(labels) != Counter({i: examples_per_class for i in range(1, classes + 1)})):
        raise ValueError("ImageNet validation labels are not the expected balanced split")

    grouped = defaultdict(list)
    for index, label in enumerate(labels, start=1):
        grouped[label].append(index)

    selections = {1: [], 2: []}
    for class_id in range(1, classes + 1):
        material = _canonical({"scheme": SELECTION_SCHEME, "seed": seed, "class_id": class_id})
        rng = random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))
        chosen = rng.sample(grouped[class_id], 2)
        selected_rows = []
        for index in chosen:
            relative = f"ILSVRC2012_val_{index:08d}.JPEG"
            path = _require_file(val_dir, relative)
            selected_rows.append({
                "image_id": path.stem,
                "image": relative,
                "label_ilsvrc2012": class_id,
                "image_sha256": sha256_file(path),
            })
        selections[2].extend(selected_rows)
        selections[1].append(selected_rows[0])

    source = {"ground_truth_sha256": sha256_file(label_file),
              "label_convention": "ILSVRC2012 validation ground truth, 1-based class IDs"}
    full = _manifest("imagenet1k", "validation",
                     {"strategy": "two_per_class", "seed": seed,
                      "selection_scheme": SELECTION_SCHEME},
                     sorted(selections[2], key=lambda row: row["image_id"]), source)
    small = _manifest("imagenet1k", "validation",
                      {"strategy": "nested_one_per_class", "seed": seed,
                       "selection_scheme": SELECTION_SCHEME,
                       "parent_manifest_id": full["manifest_id"]},
                      sorted(selections[1], key=lambda row: row["image_id"]), source)
    return full, small


def ade20k_manifest(root: Path, *, expected_count: int = 2000) -> dict:
    """Use all ADE20K validation RGB/mask pairs and verify content hashes."""
    root = Path(root)
    image_dir, annotation_dir = root / "images/validation", root / "annotations/validation"
    images = sorted(image_dir.glob("*.jpg"))
    masks = sorted(annotation_dir.glob("*.png"))
    if len(images) != expected_count or len(masks) != expected_count:
        raise ValueError(f"ADE20K needs {expected_count} images and masks")
    if {p.stem for p in images} != {p.stem for p in masks}:
        raise ValueError("ADE20K image and annotation IDs do not match")
    samples = []
    for image in images:
        ir = f"images/validation/{image.name}"
        ar = f"annotations/validation/{image.stem}.png"
        mask = _require_file(root, ar)
        samples.append({"image_id": image.stem, "image": ir, "target": ar,
                        "image_sha256": sha256_file(image),
                        "target_sha256": sha256_file(mask)})
    return _manifest("ade20k", "validation", {"strategy": "all"}, samples,
                     {"root_layout": "ADEChallengeData2016"})


def nyuv2_manifest(test_root: Path, *, expected_count: int = 654) -> dict:
    """Use all 654 extracted NYUv2 labeled test samples, not the 1,449-frame MAT."""
    test_root = Path(test_root)
    csv_file = test_root / "manifest.csv"
    with csv_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"test_order", "source_matlab_index", "rgb_path", "depth_path"}
    if not rows or set(rows[0]) != required or len(rows) != expected_count:
        raise ValueError("Invalid NYUv2 test manifest layout or count")
    try:
        rows.sort(key=lambda row: int(row["test_order"]))
        orders = [int(row["test_order"]) for row in rows]
        original_ids = [int(row["source_matlab_index"]) for row in rows]
    except ValueError as exc:
        raise ValueError("Invalid NYUv2 sample indices") from exc
    # Accept either 0-based or 1-based contiguous source extraction order.
    if orders not in (list(range(expected_count)), list(range(1, expected_count + 1))):
        raise ValueError("NYUv2 test order must be contiguous")
    if len(set(original_ids)) != len(original_ids) or any(not 1 <= i <= 1449 for i in original_ids):
        raise ValueError("NYUv2 source MATLAB indices must be unique and in 1..1449")
    samples = []
    for row, source_id in zip(rows, original_ids):
        rgb_rel, depth_rel = row["rgb_path"], row["depth_path"]
        if (not rgb_rel.startswith("rgb/") or not depth_rel.startswith("depth/") or
                not rgb_rel.endswith(".png") or not depth_rel.endswith(".npy") or
                PurePosixPath(rgb_rel).stem != PurePosixPath(depth_rel).stem or
                PurePosixPath(rgb_rel).stem != f"{source_id:06d}"):
            raise ValueError("NYUv2 RGB/depth paths or MATLAB indices do not correspond")
        rgb, depth = _require_file(test_root, rgb_rel), _require_file(test_root, depth_rel)
        samples.append({
            "image_id": f"nyuv2_{source_id:06d}",
            "source_matlab_index": source_id,
            "image": rgb_rel,
            "target": depth_rel,
            "image_sha256": sha256_file(rgb),
            "target_sha256": sha256_file(depth),
        })
    return _manifest("nyuv2", "test", {"strategy": "all"}, samples,
                     {"extraction_manifest_sha256": sha256_file(csv_file),
                      "depth_convention": "aligned depths in metres; source MATLAB 1-based IDs"})


def write_manifest(path: Path, manifest: dict) -> None:
    """Write deterministically; refuse to silently replace a frozen manifest."""
    verify_manifest(manifest)
    path = Path(path)
    payload = json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Existing frozen manifest differs: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze hashed dataset subsets for DINO robustness")
    parser.add_argument("--imagenet-val", type=Path, required=True)
    parser.add_argument("--imagenet-labels", type=Path, required=True)
    parser.add_argument("--ade20k-root", type=Path, required=True)
    parser.add_argument("--nyuv2-test", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    large, small = imagenet_manifests(args.imagenet_val, args.imagenet_labels, seed=args.seed)
    manifests = {
        "imagenet_7b_2_per_class.json": large,
        "imagenet_vitl_1_per_class.json": small,
        "ade20k_val_all.json": ade20k_manifest(args.ade20k_root),
        "nyuv2_test_all.json": nyuv2_manifest(args.nyuv2_test),
    }
    # Compute/validate *all* before writing any output files.
    for filename, manifest in manifests.items():
        write_manifest(args.output_dir / filename, manifest)
        print(f"{filename}: {manifest['sample_count']} samples, {manifest['manifest_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
