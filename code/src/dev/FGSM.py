"""
implementation notes or logs:

First experiment could be:
Attack Objective:
make the perturbed image's DINOv3 representation different from the clean image's representation.

pipeline becomes:
clean image x
    |
    v
extract clean global feature / patch features

create adversarial image x_adv using FGSM
    |
    v
extract adversarial global feature / patch features

compare:
global cosine similarity
patch-wise cosine similarity map

The first FGSM version can use a feature-distance loss, for example:
loss = -cosine_similarity(feature_clean, feature_adv)

The FGSM updates the image in the direction that reduces similarity:
x_adv = x + epsilon * sign(gradient)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.utils.device import get_device
from src.utils.transforms import get_transform
from src.features.extraction import extract_features
from src.models.dinov3 import load_dinov3


# global variables
# should remove the device block later, figure out a better solition
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)


def denormalise(x):
    return (x * imagenet_std + imagenet_mean).clamp(0, 1)


def clamp_normalised(x):
    lower = (0.0 - imagenet_mean) / imagenet_std
    upper = (1.0 - imagenet_mean) / imagenet_std
    return torch.max(torch.min(x, upper), lower)


def save_tensor_as_image(x, path):
    image_tensor = x[0].detach().cpu().permute(1, 2, 0).numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(image_tensor)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
    plt.close()


def feature_extraction_from_tensor(model, x):
    global_feature = model(x)
    features = model.forward_features(x)
    patch_tokens = features["x_norm_patchtokens"]

    return global_feature, patch_tokens


def main():
    # transform_resolution = 224
    epsilon = 0.01

    project_root = Path(__file__).resolve().parent.parent

    checkpoint_path = project_root / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    image_path = project_root / "input/test_img.jpg"
    # output_path = project_root / f"output/test_output_{transform_resolution}.jpg"

    # check device
    device = get_device()

    print("Using device:", device)

    # transform - preprocessing
    transform = get_transform(resolution=224)

    model = load_dinov3(checkpoint_path, device)

    # load image
    image = Image.open(image_path).convert("RGB")

    x = transform(image).unsqueeze(0).to(device)

    x_clean, clean_feature, _  = extract_features(model, x)

    clean_feature = clean_feature.detach()

    x_adv = x_clean.clone().detach()
    x_adv.requires_grad_(True)

    adv_feature, _ = feature_extraction_from_tensor(model, x_adv)

    # Objective:
    # reduce cosine similarity between clean feature and adversarial feature.
    similarity = F.cosine_similarity(clean_feature, adv_feature, dim=1).mean()

    loss = similarity

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        # To reduce similarity, move in the negative gradient direction.
        x_adv = x_adv - epsilon * x_adv.grad.sign()
        x_adv = clamp_normalised(x_adv)

    # check attack results
    with torch.no_grad():
        final_adv_feature, _ = feature_extraction_from_tensor(model, x_adv)

    final_similarity = F.cosine_similarity(clean_feature, final_adv_feature, dim=1).item()

    print("-----------------")
    print("Input shape:", x_clean.shape)
    print("Clean feature shape:", clean_feature.shape)
    print("Adversarial feature shape:", final_adv_feature.shape)
    print("Cosine similarity after FGSM:", final_similarity)
    print("-----------------")


if __name__ == '__main__':
    main()
