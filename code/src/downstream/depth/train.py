from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.nyudepth import NYUDepthDataset
from src.models.loader import load_model
from src.downstream.depth.decoder import LinearDepthHead
from src.utils.device import get_device
from src.utils.upsampling import upsampling
from src.utils.transforms import get_transform, get_depth_transform
from src.utils.embed_dim import get_embed_dim
from src.features.extraction import extract_dense_features


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    dataset_path = project_root / "dataset/NYU_Depth_v2/nyu_depth_v2_labeled.mat"

    device = get_device()

    # transform resolution
    res = 448

    # modify HERE for different models.
    model_name = "dinov2_vitl14"

    embed_dim = get_embed_dim(model_name)

    dataset = NYUDepthDataset(
        root=dataset_path,
        transform=get_transform(res),
        depth_transform=get_depth_transform(res)
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    model = load_model(model_name, device)

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    head = LinearDepthHead(embed_dim=embed_dim).to(device)

    criterion = torch.nn.SmoothL1Loss()

    optimiser = torch.optim.Adam(
        head.parameters(),
        lr=1e-3
    )

    num_epochs = 20

    for epoch in range(num_epochs):
        head.train()

        total_loss = 0
        total_abs_rel = 0

        for images, depths in loader:
            images = images.to(device)
            depths = depths.to(device)

            features = extract_dense_features(model, images)

            pred_depth = head(features)
            pred_depth = upsampling(pred_depth, depths.shape[-2:])

            loss = criterion(pred_depth, depths)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()

            # for debugging
            abs_rel = torch.abs(pred_depth - depths) / (depths + 1e-8)

            total_abs_rel += abs_rel.mean().item()

        avg_loss = total_loss / len(loader)
        avg_abs_rel = total_abs_rel / len(loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Abs Rel: {avg_abs_rel:.4f}")

    torch.save(head.state_dict(), project_root / f"checkpoints/downstream/depth/{model_name}/nyu_linear_head_{res}.pth")
