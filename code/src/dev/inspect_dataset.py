from pathlib import Path
from PIL import Image


dataset_path = Path("/scratch_root/hw2825/datasets/imagenet/val")

images = list(dataset_path.glob("*.JPEG"))

print(len(images))

for img_path in images[:5]:
    img = Image.open(img_path)
    print(img_path.name, img.size)
