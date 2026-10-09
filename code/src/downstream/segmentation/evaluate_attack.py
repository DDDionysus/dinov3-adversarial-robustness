from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.voc import VOCSegmentationDataset
from src.models.loader import load_model
from src.downstream.segmentation.decoder import LinearSegmentationHead
from src.features.extraction import extract_dense_features
from src.utils.device import get_device
from src.utils.transforms import get_transform, get_mask_transform
from src.utils.upsampling import upsampling
from src.utils.metrics import update_iou, compute_miou
from src.utils.embed_dim import get_embed_dim
from src.attacks.fgsm import fgsm_seg_attack
from src.attacks.pgd import pgd_seg_attack


def evaluate_attack_miou(model, head, device, loader, attack_type, epsilon):
    """
    Evaluate semantic segmentation robustness under a selected adversarial attack.
    Adversarial examples are generated through both the frozen DINO backbone and
    the trained segmentation prediction head using the cross-entropy loss.
    Performance measured using mean Intersection over Union (mIoU).

    Params:
        model: torch.nn.Module
            Frozen DINO backbone.
        head: torch.nn.Module
            Trained semantic segmentation head.
        device: torch.device
            Device used for evaluation.
        loader: torch.utils.data.DataLoader
            Dataloader used for evaluation.
        attack_type: str
            Adversarial attack to apply: "smaller_test" or "pgd".
        epsilon: float
            Perturbation budget.

    Returns:
        float
            Dataset-level mIoU.
    """

    # accumulate class-wise intersection and union over the full dataset
    intersection = torch.zeros(21, device=device)
    union = torch.zeros(21, device=device)

    alpha = epsilon / 5

    for images, masks in tqdm(loader, desc=f"{attack_type.upper()} eps={epsilon}"):
        images = images.to(device)

        # conver masks from [B, 1, H, W] to class-index format [B, H, W]
        masks = masks.squeeze(1).to(device, dtype=torch.long)

        if attack_type == "smaller_test":
            adv_img = fgsm_seg_attack(model, head, images, masks, epsilon)
        elif attack_type == "pgd":
            adv_img = pgd_seg_attack(model, head, images, masks, epsilon, alpha, steps=10)
        else:
            raise ValueError(f"Unsupported attack: {attack_type}")

        # evaluate the generated adversarial input without tracking further gradients
        with torch.no_grad():
            features = extract_dense_features(model, adv_img)

            logits = head(features)
            logits = upsampling(logits, size=masks.shape[-2:])

            pred = torch.argmax(logits, dim=1)

        intersection, union = update_iou(
            pred,
            masks,
            num_classes=21,
            intersection=intersection,
            union=union
        )

    miou = compute_miou(
        intersection,
        union
    )

    return miou.item()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]

    device = get_device()

    # transform resolution
    res = 448

    dataset = VOCSegmentationDataset(
        root=project_root / "dataset/VOC2012",
        split="val",
        transform=get_transform(res),
        mask_transform=get_mask_transform(res)
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False
    )

    model_name = "dinov3_vitl16"
    attack = "pgd"

    embed_dim = get_embed_dim(model_name)

    model = load_model(model_name=model_name, device=device)

    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    head = LinearSegmentationHead(embed_dim=embed_dim, num_classes=21).to(device)

    head.load_state_dict(
        torch.load(
            project_root /
            f"checkpoints/downstream/seg/{model_name}/voc_linear_head_{res}.pth",
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
        miou = evaluate_attack_miou(
            model,
            head,
            device,
            loader,
            attack_type=attack,
            epsilon=e
        )

        print(
            f"{attack.upper()} epsilon={e}, mIoU={miou}"
        )
