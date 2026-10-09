from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.voc import VOCSegmentationDataset
from src.models.loader import load_model
from src.downstream.segmentation.decoder import LinearSegmentationHead
from src.utils.device import get_device
from src.utils.upsampling import upsampling
from src.utils.transforms import get_transform, get_mask_transform
from src.utils.embed_dim import get_embed_dim
from src.features.extraction import extract_dense_features


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    dataset_path = project_root / "dataset/VOC2012"

    device = get_device()

    # transform resolution
    res = 448

    # modify HERE for different models.
    model_name = "dinov2_vitl14"

    embed_dim = get_embed_dim(model_name)

    dataset = VOCSegmentationDataset(
        root=dataset_path,
        split="train",
        transform=get_transform(res),
        mask_transform=get_mask_transform(res)
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    # model = load_dinov3(model_name, device, checkpoint_path)
    model = load_model(model_name, device)

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    head = LinearSegmentationHead(embed_dim=embed_dim, num_classes=21).to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    optimiser = torch.optim.Adam(
        head.parameters(),
        lr=1e-3
    )

    num_epochs = 20

    for epoch in range(num_epochs):
        head.train()

        total_loss = 0

        for images, masks in loader:
            images = images.to(device)
            masks = masks.squeeze(1).to(device, dtype=torch.long)

            features = extract_dense_features(model, images)

            logits = head(features)
            logits = upsampling(logits, res)

            loss = criterion(logits, masks)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    torch.save(head.state_dict(), project_root / f"checkpoints/downstream/seg/{model_name}/voc_linear_head_{res}.pth")
