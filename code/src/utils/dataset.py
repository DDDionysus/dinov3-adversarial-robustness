from pathlib import Path

from src.datasets.imagenet import ImageNetValDataset
from src.datasets.nyudepth import NYUDepthDataset
from src.datasets.voc import VOCSegmentationDataset


def get_dataset(dataset_name="imagenet", dataset_path=None):
    """
    Load a dataset for representation-level robustness evaluation.
    All datasets are returned in representation mode, where only the RGB image and
    sample identifier are required.

    Params:
        dataset_name: str
            Name of the dataset to load

    Returns:
        torch.utils.data.Dataset
            Dataset wrapper configured for representation-level evaluation.
    """

    project_root = Path(__file__).resolve().parents[2]

    # local default dataset path, not for HPC use
    default_path = {
        "imagenet": project_root / "dataset/subset/val_subset",
        # "imagenet": "C:\\Users\\10031\\irp-hw2825\\code\\input",
        "voc": project_root / "dataset/VOC2012",
        "nyu": project_root / "dataset/NYU_Depth_v2/nyu_depth_v2_labeled.mat"
    }

    if dataset_path is None:
        dataset_path = default_path[dataset_name]
    else:
        dataset_path = Path(dataset_path)

    if dataset_name == "voc":
        return VOCSegmentationDataset(dataset_path, split="val", representation=True)

    if dataset_name == "nyu":
        return NYUDepthDataset(dataset_path, representation=True)

    return ImageNetValDataset(dataset_path)
