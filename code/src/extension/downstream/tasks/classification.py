import torch
import torch.nn.functional as F

from src.models.dinov3 import load_dinov3_downstream
from src.utils.device import get_device
from src.utils.transforms import get_transform, get_normalise


def evaluate_clean(model, dataloader, normalisation, device):
    """

    """

    with torch.no_grad():
        model.eval()

        correct = 0
        total = 0
        total_loss = 0.0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(normalisation(images))

            loss = F.cross_entropy(logits, labels)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        return {
            "accuracy": correct / total,
            "loss": total_loss / total,
        }


if __name__ == '__main__':
    device = get_device()

    model = load_dinov3_downstream("classification", device)

    transform = get_transform(resolution=224)
    normalisation = get_normalise()

    dataset, dataloader =

    results = evaluate_clean(model, dataloader, normalisation, device)

    print(results)
