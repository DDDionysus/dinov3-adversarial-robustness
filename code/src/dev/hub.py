import torch
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent

checkpoint_path = project_root / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

model = torch.hub.load(
    "facebookresearch/dinov3",
    "dinov3_vits16",
    weights=str(checkpoint_path),
)

'''
Extra dependencies: 
- torchmetrics
- termcolor
'''
