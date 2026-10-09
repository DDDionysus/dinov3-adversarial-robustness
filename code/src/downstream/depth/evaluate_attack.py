from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.nyudepth import NYUDepthDataset
from src.models.loader import load_model
from src.downstream.depth.decoder import LinearDepthHead
from src.features.extraction import extract_dense_features
from src.utils.device import get_device
from src.utils.transforms import get_transform, get_depth_transform
from src.utils.upsampling import upsampling
from src.utils.metrics import update_depth_metrics, compute_rmse, compute_abs_rel
from src.utils.embed_dim import get_embed_dim
from src.attacks.fgsm import fgsm_depth_attack
from src.attacks.pgd import pgd_depth_attack


def evaluate_attack_metrics(model, head, device, loader, attack_type, epsilon):
    """
    Evaluate monocular depth robustness under a selected adversarial attack.
    Adversarial examples are generated through both the frozen DINO backbone and
    the trained depth prediction head using the depth regression loss.
    Performance measured using RMSE and AbsRel.

    Params:
        model: torch.nn.Module
            Frozen DINO backbone.
        head: torch.nn.Module
            Trained depth estimation head.
        device: torch.device
            Device used for evaluation.
        loader: torch.utils.data.DataLoader
            Dataloader used for evaluation.
        attack_type: str
            Adversarial attack to apply: "smaller_test" or "pgd".
        epsilon: float
            Perturbation budget.

    Returns:
        tuple
            Dataset-level RMSE and AbsRel
    """

    # metrics accumulation
    sq_error_sum = 0.0
    abs_rel_sum = 0.0
    valid_count = 0

    alpha = epsilon / 5

    for images, depths in tqdm(loader, desc=f"{attack_type.upper()} eps={epsilon}"):
        images = images.to(device)
        depths = depths.to(device)

        if attack_type == "smaller_test":
            adv_img = fgsm_depth_attack(model, head, images, depths, epsilon)
        elif attack_type == "pgd":
            adv_img = pgd_depth_attack(model, head, images, depths, epsilon, alpha, steps=10)
        else:
            raise ValueError(f"Unsupported attack: {attack_type}")

        # evaluate the generated adversarial input without tracking further gradients
        with torch.no_grad():
            features = extract_dense_features(model, adv_img)

            pred_depth = head(features)
            pred_depth = upsampling(pred_depth, size=depths.shape[-2:])

        sq_error_sum, abs_rel_sum, valid_count = update_depth_metrics(
            pred_depth,
            depths,
            sq_error_sum,
            abs_rel_sum,
            valid_count
        )

    rmse = compute_rmse(sq_error_sum, valid_count)
    abs_rel = compute_abs_rel(abs_rel_sum, valid_count)

    return rmse, abs_rel


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    dataset_path = project_root / "dataset/NYU_Depth_v2/nyu_depth_v2_labeled.mat"

    device = get_device()

    # transform resolution
    res = 448

    dataset = NYUDepthDataset(
        root=dataset_path,
        transform=get_transform(res),
        depth_transform=get_depth_transform(res)
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False
    )

    model_name = "dinov2_vitl14"
    attack = "pgd"

    embed_dim = get_embed_dim(model_name)

    model = load_model(model_name=model_name, device=device)

    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    head = LinearDepthHead(embed_dim=embed_dim).to(device)

    head.load_state_dict(
        torch.load(
            project_root /
            f"checkpoints/downstream/depth/{model_name}/nyu_linear_head_{res}.pth",
            weights_only=True
        )
    )

    head.eval()

    epsilon = [
        0,
        0.001,
        0.005,
        0.01,
        0.02,
        0.05
    ]

    for e in epsilon:
        rmse, abs_rel = evaluate_attack_metrics(
            model,
            head,
            device,
            loader,
            attack_type=attack,
            epsilon=e
        )

        print(
            f"{attack.upper()} epsilon={e}, "
            f"RMSE={rmse}, AbsRel={abs_rel}"
        )
