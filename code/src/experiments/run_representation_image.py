"""Restartable one-image representation-attack workflow.

This is the common computation core for a future dataset/Slurm runner, not a
command-line entry point or downstream-head evaluator. Input is one already
preprocessed, *unnormalized* RGB image [1,3,H,W], valued in [0,1]. Store
and evaluate the reconstructed int8 perturbation, not the original float32
attack tensor, so the metrics describe the exact archived experiment.

The caller owns both writers; one stable, non-overlapping dataset partition
per writer/prefix. Never derive partition names from transient Slurm job IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import torch

from src.attacks.fgsm import rfgsm_attack
from src.attacks.pgd import pgd_attack
from src.evaluation.representation_metrics import representation_metrics
from src.features.extraction import extract_features
from src.utils.metric_records import MetricJSONLWriter, make_metric_record
from src.utils.perturbation_archive import (
    PerturbationShardWriter, attack_key_for_input,
    reconstruct_attack,
)


_SEED_SCHEME = "sha256-image-seed-v1"


def derive_image_seed(global_seed: int, *, dataset: str, image_id: str) -> int:
    """Get a stable per-image PyTorch seed from the experiment's global seed.

    Identical global seeds and sample IDs produce identical initial noise across
    objectives and models; distinct sample IDs do not reuse a random-start
    stream. Python's randomized hash() is intentionally not used here.
    """
    if not isinstance(global_seed, int) or isinstance(global_seed, bool):
        raise ValueError("global_seed must be an integer")
    if any(not isinstance(value, str) or not value for value in (dataset, image_id)):
        raise ValueError("dataset and image_id must be non-empty strings")
    identity = {
        "scheme": _SEED_SCHEME,
        "global_seed": global_seed,
        "dataset": dataset,
        "image_id": image_id,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    # A nonnegative 63-bit integer is supported by torch.Generator.manual_seed.
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class AttackSpec:
    """One random-start representation attack; no implicit defaults for PGD."""

    method: str
    objective: str
    epsilon: float
    seed: int  # Global experiment seed; not the per-image generator seed.
    alpha: float | None = None
    steps: int | None = None

    def metadata(self, *, dataset, image_id, model, checkpoint_id,
                 preprocessing, protocol_id) -> dict:
        if self.method not in {"rfgsm", "pgd"}:
            raise ValueError(f"Unsupported attack method: {self.method}")
        if self.objective not in {"cls", "patch", "gram"}:
            raise ValueError(f"Unsupported objective: {self.objective}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if not math.isfinite(float(self.epsilon)) or not 0 < self.epsilon <= 1:
            raise ValueError("epsilon must be in (0, 1]")
        if self.method == "pgd":
            if (self.alpha is None or not math.isfinite(float(self.alpha)) or
                    self.alpha <= 0 or self.alpha > self.epsilon):
                raise ValueError("PGD alpha must be in (0, epsilon]")
            if not isinstance(self.steps, int) or isinstance(self.steps, bool) or self.steps < 1:
                raise ValueError("PGD steps must be a positive integer")
        elif self.alpha is not None or self.steps is not None:
            raise ValueError("R-FGSM does not accept alpha or steps")

        image_seed = derive_image_seed(self.seed, dataset=dataset, image_id=image_id)
        return {
            "dataset": dataset, "image_id": image_id, "model": model,
            "checkpoint_id": checkpoint_id, "preprocessing": preprocessing,
            "protocol_id": protocol_id, "attack": self.method,
            "objective": self.objective, "epsilon": float(self.epsilon),
            "global_seed": self.seed, "seed": image_seed,
            "seed_derivation": _SEED_SCHEME,
            "alpha": None if self.alpha is None else float(self.alpha),
            "steps": self.steps,
        }


def attack_condition_id(metadata: dict) -> str:
    """Stable readable condition plus full configuration digest.

    Include the global seed, not the image-specific derived seed, so one
    attack condition groups together *all* images in a dataset. Alpha and
    steps distinguish PGD variants. The archive separately records the exact
    derived seed; no timestamp or job ID participates.
    """
    params = {field: metadata[field] for field in (
        "attack", "objective", "epsilon", "global_seed", "alpha", "steps")}
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{metadata['attack']}.{metadata['objective']}.eps{metadata['epsilon']:.8g}.{digest}"


def _identity(metadata: dict, condition_id: str, attack_key: str | None) -> dict:
    return {
        "dataset": metadata["dataset"], "image_id": metadata["image_id"],
        "model": metadata["model"], "checkpoint_id": metadata["checkpoint_id"],
        "task": "representation", "protocol_id": metadata["protocol_id"],
        "preprocessing": metadata["preprocessing"],
        "condition_id": condition_id, "attack_key": attack_key,
    }


def run_representation_image(
    *, model: torch.nn.Module, x_clean: torch.Tensor, normalisation,
    dataset: str, image_id: str, model_name: str, checkpoint_id: str,
    preprocessing: str, protocol_id: str, specs: list[AttackSpec],
    attacks: PerturbationShardWriter, metrics: MetricJSONLWriter,
) -> dict[str, int]:
    """Process all specified attacks for one image and safely resume failures.

    * Check metric identity and archive identity before any model forward pass.
    * Compute clean features only once per image when needed.
    * Reuse cached clean features as fixed attack targets (no gradients).
    * Flush each generated attack BEFORE evaluating/reconstructing it.
    * Evaluate only the quantized, replayable image and fsync metric records.

    This function is single-image and sequential; a later commit will provide
    deterministic dataset partitions and a Slurm command-line entry point.
    """
    if not isinstance(x_clean, torch.Tensor) or x_clean.ndim != 4 or x_clean.shape[0] != 1 or x_clean.shape[1] != 3:
        raise ValueError("Expected x_clean with shape [1,3,H,W]")
    if x_clean.dtype != torch.float32 or not torch.isfinite(x_clean).all():
        raise ValueError("x_clean must be a finite float32 pixel tensor")
    if torch.any(x_clean < 0) or torch.any(x_clean > 1):
        raise ValueError("x_clean must be unnormalized pixels in [0,1]")
    if x_clean.requires_grad:
        raise ValueError("x_clean must not require gradients")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Use a frozen, evaluation-mode backbone")
    if not isinstance(specs, list):
        raise TypeError("specs must be a list of AttackSpec")

    # Hash CPU/float32 pixels exactly as the archive does; no hidden resizing.
    clean_chw = x_clean[0].detach().cpu().float().contiguous()
    example_meta = {
        "dataset": dataset, "image_id": image_id, "model": model_name,
        "checkpoint_id": checkpoint_id, "preprocessing": preprocessing,
        "protocol_id": protocol_id,
    }
    plan = []
    identities = set()
    for spec in specs:
        if not isinstance(spec, AttackSpec):
            raise TypeError("All attack specs must be AttackSpec instances")
        metadata = spec.metadata(**example_meta)
        key = attack_key_for_input(clean_chw, metadata)
        condition = attack_condition_id(metadata)
        identity = _identity(metadata, condition, key)
        if condition in identities:
            raise ValueError(f"Duplicate attack condition: {condition}")
        identities.add(condition)
        already_scored = metrics.contains_identity(**identity)
        archived = attacks.contains(clean_chw, metadata)
        if already_scored and not archived:
            raise ValueError("Metric exists without its corresponding perturbation archive")
        plan.append((spec, metadata, key, identity, archived, already_scored))

    pending = [item for item in plan if not item[5]]
    result = {"already_scored": len(plan) - len(pending),
              "reused_attack": 0, "generated_attack": 0, "evaluated": 0}
    if not pending:
        return result

    device = x_clean.device
    with torch.no_grad():
        clean_cls, clean_patch = extract_features(model, normalisation(x_clean))
        clean_features = (clean_cls.detach(), clean_patch.detach())

    for spec, metadata, key, identity, archived, _ in pending:
        if archived:
            result["reused_attack"] += 1
        else:
            if spec.method == "rfgsm":
                x_adv = rfgsm_attack(
                    model, normalisation, x_clean, spec.epsilon, metadata["seed"],
                    spec.objective, clean_features=clean_features,
                )
            else:
                x_adv = pgd_attack(
                    model, normalisation, x_clean, spec.epsilon, spec.alpha,
                    spec.steps, metadata["seed"], spec.objective,
                    clean_features=clean_features,
                )
            attacks.add(clean_chw, x_adv[0], metadata)
            # Even for a buffered writer, a successful attack is now durable.
            attacks.flush()
            del x_adv
            result["generated_attack"] += 1

        record, q_delta = attacks.load(key)
        # Reject replay under a mismatched experiment setting even if the
        # stored clean-input checksum matches.
        if any(record.get(field) != value for field, value in metadata.items()):
            raise ValueError("Archived attack metadata differs from requested condition")
        reconstructed = reconstruct_attack(clean_chw, q_delta, record)
        x_adv_replayed = torch.from_numpy(np.ascontiguousarray(reconstructed)).unsqueeze(0).to(device)
        with torch.no_grad():
            adv_cls, adv_patch = extract_features(model, normalisation(x_adv_replayed))
            values = representation_metrics(*clean_features, adv_cls, adv_patch)
        delta = x_adv_replayed - x_clean
        values.update({
            "attack_linf_reconstructed": float(delta.abs().max().item()),
            "attack_l2_reconstructed": float(torch.linalg.vector_norm(delta).item()),
            "attack_original_linf": float(record["original_linf"]),
            "attack_quantization_max_error": float(record["quantization_max_error"]),
        })
        metrics.add(make_metric_record(**identity, metrics=values))
        result["evaluated"] += 1
        del x_adv_replayed, adv_cls, adv_patch, delta

    return result
