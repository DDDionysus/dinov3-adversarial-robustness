from pathlib import Path

import torch
from PIL import Image
import matplotlib.pyplot as plt

from src.models.dinov3 import load_dinov3
from src.utils.device import get_device
from src.utils.transforms import get_transform
from src.utils.upsampling import upsampling
from src.features.extraction import extract_dense_features, extract_features
from src.downstream.segmentation.decoder import LinearSegmentationHead


def inference(model, image, head):
    """
    Run semantic segmentation inference using dense DINO features and a segmentation
    prediction head.
    """

    patch_tokens = extract_dense_features(model, image)

    print(patch_tokens.shape)

    with torch.no_grad():
        logits = head(patch_tokens)

    print(logits.shape)

    logits = upsampling(logits)

    print(logits.shape)

    prediction = torch.argmax(
        logits,
        dim=1
    )

    print(prediction.shape)

    return prediction


def visual(img, prediction):
    """
    Display the original image alongside its predicted segmentation map.
    """

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(prediction[0].cpu())
    plt.title("Predicted Segmentation")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# a sanity check for the prediction head pipeline
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]

    checkpoint_path = project_root / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    image_path = project_root / "dataset/subset/val_subset/ILSVRC2012_val_00000031.JPEG"

    device = get_device()

    # load dinov3 model
    dinov3 = load_dinov3(checkpoint_path, device)
    dinov3.eval()

    # segmentation head
    head = LinearSegmentationHead(
        embed_dim=384,
        num_classes=21,
    ).to(device)
    head.eval()

    img = Image.open(image_path).convert("RGB")

    transform = get_transform(resolution=512)
    x = transform(img).unsqueeze(0).to(device)

    pred = inference(dinov3, x, head)

    visual(img, pred)
