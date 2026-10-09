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
from src.utils.embed_dim import get_embed_dim
from src.utils.metrics import update_depth_metrics, compute_rmse, compute_abs_rel


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

    # metrics accumulation
    sq_error_sum = 0.0
    abs_rel_sum = 0.0
    valid_count = 0

    with torch.no_grad():
        for images, depths in tqdm(loader):
            images = images.to(device)
            depths = depths.to(device)

            features = extract_dense_features(model, images)

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

    print("RMSE:", rmse)
    print("AbsRel:", abs_rel)
