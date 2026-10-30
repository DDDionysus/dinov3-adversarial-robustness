"""Execute one frozen, verified representation experiment partition.

This is a command-line adapter around ``run_verified_representation_samples``.
It does NOT evaluate official downstream classification, segmentation or depth
heads. All configuration comes from a previously frozen experiment provenance
and dataset manifest, not from ad hoc values in the launcher.

Example (paths shown for illustration; first create the frozen JSON documents):

PYTHONPATH=code python -m src.experiments.run_frozen_representation \\
  --manifest /path/to/imagenet_7b_2_per_class.json \\
  --manifest-role imagenet_7b --dataset-root /path/to/imagenet/val \\
  --source-file ground_truth_sha256=/path/to/ILSVRC2012_validation_ground_truth.txt \\
  --provenance /path/to/experiment.json --checkpoint-role dinov3_7b \\
  --checkpoint /path/to/dinov3_vit7b16_pretrain_lvd1689m.pth \\
  --model dinov3_vit7b16 --hub-root /path/to/external \\
  --output-root /path/to/outputs --device cuda \\
  --max-samples 1 --max-conditions 1

An explicit --verify-only mode checks the selected on-disk source files without
loading a model, using the GPU, or writing output artifacts.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import torch

from src.experiments.dataset_manifests import sha256_file, verify_manifest
from src.experiments.experiment_provenance import (
    _git_revision, verify_manifest_sources, iter_verified_samples,
)
from src.experiments.run_representation_image import AttackSpec
from src.experiments.verified_dataset_runner import (
    _check_provenance, run_verified_representation_samples,
)
from src.models.hub_source import hub_source_kwargs
from src.models.loader import load_model
from src.utils.metric_records import MetricJSONLWriter
from src.utils.perturbation_archive import PerturbationShardWriter
from src.utils.spatial_map_archive import SpatialMapShardWriter


# Explicitly restrict this executable to our three studied backbone variants.
# These are representation models, not the downstream heads.
MODEL_CHECKPOINTS = {
    "dinov2_vitl14": "dinov2_vitl14_pretrain.pth",
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m.pth",
    "dinov3_vit7b16": "dinov3_vit7b16_pretrain_lvd1689m.pth",
}


def _read_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _source_paths(raw: list[str]) -> dict[str, Path]:
    """Parse --source-file HASH_FIELD=PATH (e.g. ground_truth_sha256=...)."""
    paths = {}
    for item in raw:
        key, separator, value = item.partition("=")
        if (not separator or not value or not re.fullmatch(r"[a-z_]+_sha256", key)):
            raise ValueError("Use --source-file hash_field=/absolute/path")
        if key in paths:
            raise ValueError(f"Repeated source-file checksum field: {key}")
        paths[key] = Path(value)
    return paths


def _image_ids_for_partition(manifest: dict, partition_index: int,
                             partition_count: int) -> set[str]:
    """Fixed modulo sharding over the immutable manifest order.

    Partition count must be kept constant for all runs of an experiment;
    a frozen plan in the output directory enforces this for active jobs.
    """
    if (not isinstance(partition_count, int) or isinstance(partition_count, bool) or
            partition_count < 1 or not isinstance(partition_index, int) or
            isinstance(partition_index, bool) or not 0 <= partition_index < partition_count):
        raise ValueError("Require 0 <= partition_index < partition_count")
    return {
        sample["image_id"] for i, sample in enumerate(manifest["samples"])
        if i % partition_count == partition_index
    }


def _publish_partition_plan(destination: Path, *, manifest_id: str,
                            partition_count: int, sample_count: int) -> None:
    """Prevent accidental changes in Slurm partitioning on a resumed run."""
    plan = {"format_version": 1, "partition_scheme": "manifest-order-modulo-v1",
            "manifest_id": manifest_id, "partition_count": partition_count,
            "sample_count": sample_count}
    payload = json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=destination.parent, prefix=".plan-",
                                         delete=False) as stream:
            tmp = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(tmp, destination)
        except FileExistsError:
            if destination.read_text(encoding="utf-8") != payload:
                raise ValueError(
                    "Frozen partition plan differs: do not change worker count "
                    "or manifest under an existing output directory"
                )
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _validate_checkpoint(checkpoint: Path, provenance: dict, role: str,
                         model_name: str) -> Path:
    """Hash the exact checkpoint bytes once before any model construction."""
    if model_name not in MODEL_CHECKPOINTS:
        raise ValueError(f"Unsupported backbone: {model_name}")
    info = provenance.get("checkpoints", {}).get(role)
    if not isinstance(info, dict):
        raise ValueError(f"Missing checkpoint role in provenance: {role}")
    path = Path(checkpoint).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("Checkpoint is not a regular file")
    expected = MODEL_CHECKPOINTS[model_name]
    if path.name != expected or info.get("filename") != expected:
        raise ValueError("Checkpoint filename is inconsistent with model/provenance")
    if path.stat().st_size != info.get("size_bytes"):
        raise ValueError("Checkpoint size differs from frozen provenance")
    if sha256_file(path) != info.get("sha256"):
        raise ValueError("Checkpoint SHA-256 differs from frozen provenance")
    return path


def _verify_model_source_revision(provenance: dict, model_name: str) -> None:
    """Validate the exact local source the Hub loader will execute.

    Caller must already be inside _configured_local_source, which forbids a
    GitHub download fallback. This check runs even in --verify-only mode.
    """
    project = "dinov2" if model_name.startswith("dinov2") else "dinov3"
    revisions = provenance.get("model_source_revisions")
    expected = revisions.get(project) if isinstance(revisions, dict) else None
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{40,64}", expected) is None:
        raise ValueError(f"Missing pinned official {project} source revision")
    checkout = Path(hub_source_kwargs(project)["repo_or_dir"])
    actual = _git_revision(checkout)
    if actual != expected:
        raise ValueError(f"Official {project} source revision differs from frozen provenance")


def _verify_git_revision(provenance: dict) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    if _git_revision(repo_root) != provenance.get("git_revision"):
        raise ValueError("Repository Git revision differs from frozen provenance")


def _selected_specs(provenance: dict, max_conditions: int | None) -> list[AttackSpec]:
    rows = provenance.get("attack_settings")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Frozen provenance has no attack settings")
    if max_conditions is not None:
        if max_conditions < 1:
            raise ValueError("max_conditions must be positive")
        rows = rows[:max_conditions]
    return [AttackSpec(**record) for record in rows]


def _output_directory(output_root: Path, provenance: dict, manifest: dict,
                      model_name: str) -> Path:
    # Full identifiers ensure disjoint archives for different experiments,
    # model variants and dataset selections without user-defined names.
    eid = provenance["experiment_id"]
    mid = manifest["manifest_id"]
    if any(re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None for value in (eid, mid)):
        raise ValueError("Invalid frozen experiment or manifest identifier")
    return Path(output_root) / eid[7:] / model_name / mid[7:]


@contextmanager
def _configured_local_source(hub_root: Path | None, model_name: str):
    """Require local Hub source code, restoring caller environment afterwards."""
    previous = os.environ.get("DINO_HUB_ROOT")
    try:
        if hub_root is not None:
            os.environ["DINO_HUB_ROOT"] = str(Path(hub_root).expanduser().resolve())
        if not os.environ.get("DINO_HUB_ROOT", "").strip():
            raise ValueError("Set DINO_HUB_ROOT or pass --hub-root")
        hub_source_kwargs("dinov2" if model_name.startswith("dinov2") else "dinov3")
        yield
    finally:
        if previous is None:
            os.environ.pop("DINO_HUB_ROOT", None)
        else:
            os.environ["DINO_HUB_ROOT"] = previous


def execute(args: argparse.Namespace) -> dict:
    """Validate frozen inputs and run one deterministic partition."""
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max_samples must be positive")
    if (isinstance(args.attack_shard_size, bool) or args.attack_shard_size < 1 or
            isinstance(args.map_shard_size, bool) or args.map_shard_size < 1):
        raise ValueError("Archive shard sizes must be positive integers")
    manifest = _read_json(args.manifest)
    provenance = _read_json(args.provenance)
    verify_manifest(manifest)
    # Do not trust the manifest name supplied by the CLI: the role must agree
    # with the immutable provenance before any output files or GPU work.
    _check_provenance(manifest, provenance, manifest_role=args.manifest_role,
                      checkpoint_role=args.checkpoint_role,
                      preprocessing=provenance.get("preprocessing"))
    _verify_git_revision(provenance)
    # Check what torch.hub will actually execute, including on verify-only
    # runs, before hashing large checkpoint files or constructing any model.
    with _configured_local_source(args.hub_root, args.model):
        _verify_model_source_revision(provenance, args.model)
    source_paths = _source_paths(args.source_file)
    verify_manifest_sources(manifest, source_paths)
    specs = _selected_specs(provenance, args.max_conditions)
    selected_ids = _image_ids_for_partition(manifest, args.partition_index,
                                             args.partition_count)
    if not selected_ids:
        raise ValueError("Selected partition contains no samples")
    # Validate the actual checkpoint file, not only its manifest metadata.
    checkpoint = _validate_checkpoint(args.checkpoint, provenance,
                                      args.checkpoint_role, args.model)

    if args.verify_only:
        sample_count = 0
        for _sample, _image, _target in iter_verified_samples(
                manifest, args.dataset_root, source_paths=source_paths,
                image_ids=selected_ids):
            sample_count += 1
            if args.max_samples is not None and sample_count >= args.max_samples:
                break
        return {"verified_samples": sample_count, "model_loaded": False,
                "experiment_id": provenance["experiment_id"]}

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; run on a Borg GPU node")
    # Refuse network-based Hub source loading, even if local files are absent.
    with _configured_local_source(args.hub_root, args.model):
        root = _output_directory(args.output_root, provenance, manifest, args.model)
        _publish_partition_plan(root / "partition-plan.json",
                                manifest_id=manifest["manifest_id"],
                                partition_count=args.partition_count,
                                sample_count=manifest["sample_count"])
        worker_name = f"part-{args.partition_index:04d}-of-{args.partition_count:04d}"
        worker = root / worker_name

        model = load_model(args.model, args.device, checkpoint_path=checkpoint)
        with ExitStack() as stack:
            # No shared writer paths across workers. Fixed worker ID on restart.
            attacks = stack.enter_context(PerturbationShardWriter(
                worker / "attacks", prefix="attack", shard_size=args.attack_shard_size))
            metrics = stack.enter_context(MetricJSONLWriter(worker / "metrics.jsonl"))
            maps = stack.enter_context(SpatialMapShardWriter(
                worker / "spatial_maps", prefix="map", shard_size=args.map_shard_size))
            result = run_verified_representation_samples(
                model=model, manifest=manifest, dataset_root=args.dataset_root,
                manifest_role=args.manifest_role, checkpoint_role=args.checkpoint_role,
                provenance=provenance, model_name=args.model, specs=specs,
                device=args.device, attacks=attacks, metrics=metrics,
                spatial_maps=maps, source_paths=source_paths,
                image_ids=selected_ids, max_samples=args.max_samples,
            )
    return {**result, "experiment_id": provenance["experiment_id"],
            "partition": worker_name, "output_directory": str(worker)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-role", required=True,
                        help="Role key of the manifest in frozen provenance")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-file", action="append", default=[], metavar="HASH_FIELD=PATH",
                        help="Required for ImageNet label file or NYUv2 extraction CSV")
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--checkpoint-role", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODEL_CHECKPOINTS), required=True)
    parser.add_argument("--hub-root", type=Path, default=None,
                        help="Root containing official dinov2/ and dinov3/ checkouts")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per partition for small smoke tests")
    parser.add_argument("--max-conditions", type=int, default=None,
                        help="Use first N frozen attacks for small smoke tests")
    parser.add_argument("--attack-shard-size", type=int, default=1,
                        help="Int8 attacks per atomic shard; 1 maximizes immediate durability, "
                             "24 reduces filesystem pressure for production")
    parser.add_argument("--map-shard-size", type=int, default=1,
                        help="Spatial maps per atomic shard; missing buffered maps are replayable")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify frozen source files and checkpoint; do not load model or write outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max_samples must be positive")
    print(json.dumps(execute(args), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
