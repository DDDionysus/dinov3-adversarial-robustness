from pathlib import Path

from torch.utils.data import DataLoader

from src.datasets.voc import VOCSegmentationDataset
from src.utils.transforms import get_transform, get_mask_transform


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[3]
    dataset_path = project_root / "dataset/VOC2012"

    transform = get_transform(resolution=512)
    mask_transform = get_mask_transform(resolution=512)

    dataset = VOCSegmentationDataset(
        root = dataset_path,
        split="train",
        transform=transform,
        mask_transform=mask_transform,
    )

    image, mask = dataset[0]

    print(image.shape, mask.shape)

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    images, masks = next(iter(loader))

    print(images.shape, masks.shape)

    masks = masks.squeeze(1)

    print(masks.shape)
