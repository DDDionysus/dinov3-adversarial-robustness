from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.voc import VOCSegmentationDataset
from src.models.loader import load_model
from src.downstream.segmentation.decoder import LinearSegmentationHead
from src.features.extraction import extract_dense_features
from src.utils.device import get_device
from src.utils.transforms import get_transform, get_mask_transform
from src.utils.upsampling import upsampling
from src.utils.metrics import update_iou, compute_miou
from src.utils.embed_dim import get_embed_dim


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

    intersection = torch.zeros(21, device=device)
    union = torch.zeros(21, device=device)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)

            masks = masks.squeeze(1).to(device, dtype=torch.long)

            features = extract_dense_features(model, images)

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

    print("mIoU: ", miou.item())
