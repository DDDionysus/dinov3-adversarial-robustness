from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class ImageNetValDataset(Dataset):
    """
    Dataset wrapper for ImageNet-1K validation images.

    Params:
        root: str or pathlib.Path
            Directory containing ImageNet validation images.

    Returns:
        PIL.Image.Image:
            RGB image.
        str:
            Image filename.
    """

    def __init__(self, root):
        self.root = Path(root)

        # sort filenames to keep the dataset order
        self.images = sorted(
            self.root.glob("*.JPEG")
        )


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image_path = self.images[idx]

        # convert all images to RGB to ensure a 3-channel input
        image = Image.open(image_path).convert("RGB")

        return image, image_path.name
