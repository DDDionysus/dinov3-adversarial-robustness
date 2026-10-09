import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image


class NYUDepthDataset(Dataset):
    """
    Dataset wrapper for NYU Depth V2 RGB images and depth maps.
    This dataset have two modes: In representation mode, it returns the original
    RGB image and a generated sample identifier; Otherwise, it returns the
    transformed RGB image with depth map.

    Params:
        root: str or pathlib.Path
            Path to NYU Depth V2 mat file.
        transform: callable, optional
            Transform applied to RGB images.
        depth_transform: callable, optional
            Transform applied to depth maps.
        representation: bool
            Return images for representation or depth estimation.

    Returns:
        representation=True:
            PIL.Image.Image:
                RGB image
            str:
                Generated sample identifier.

        representation=False:
            PIL.Image.Image:
                Transformed RGB image.
            torch.Tensor:
                Depth map.
    """

    def __init__(self, root, transform=None, depth_transform=None, representation=False):
        self.root = root
        self.transform = transform
        self.depth_transform = depth_transform
        self.representation = representation

        # keep the HDF5 file open to assess image and depth arrays
        self.file = h5py.File(self.root, "r")

        self.images = self.file["images"]
        self.depths = self.file["depths"]

        self.length = self.images.shape[0]


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        image = self.images[idx]
        depth = self.depths[idx]

        # convert HDF5 image data to the HWC format for PIL
        image = Image.fromarray(
            image.transpose(1, 2, 0)
        )

        # representation-level only need RGB image
        if self.representation:
            return image, f"nyu_{idx:04d}"

        # add channel dimension to for me shape [1, H, W]
        depth = torch.from_numpy(depth).float().unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        if self.depth_transform:
            depth = self.depth_transform(depth)

        return image, depth
