from pathlib import Path
from torch.utils.data import DataLoader

from src.datasets.nyudepth import NYUDepthDataset
from src.utils.transforms import get_transform, get_depth_transform


# quick sanity check
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    data_path = project_root / "dataset/NYU_Depth_v2/nyu_depth_v2_labeled.mat"

    res = 448

    dataset = NYUDepthDataset(
        root=data_path,
        transform=get_transform(res),
        depth_transform=get_depth_transform(res)
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False
    )

    images, depths = next(iter(loader))

    print("images:", images.shape)
    print("depths:", depths.shape)
