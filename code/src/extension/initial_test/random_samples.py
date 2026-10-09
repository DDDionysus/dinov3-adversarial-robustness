from pathlib import Path
import random
import shutil


SOURCE_DIR = Path("C:/Users/10031/irp-hw2825/code/dataset/subset/val_subset")
OUTPUT_DIR = Path("/dataset/subset/initial_test")

NUM_SAMPLES = 100
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPEG"}


images = [
    p for p in SOURCE_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
]

print(f"Found {len(images)} images.")

if len(images) < NUM_SAMPLES:
    raise ValueError(
        f"Dataset only contains {len(images)} images, "
        f"cannot sample {NUM_SAMPLES}."
    )


random.seed(SEED)
selected = random.sample(images, NUM_SAMPLES)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for src in selected:
    dst = OUTPUT_DIR / src.name
    shutil.copy2(src, dst)

print(f"Copied {len(selected)} images to:")
print(OUTPUT_DIR)

sample_list = OUTPUT_DIR / "sample_list.txt"

with open(sample_list, "w") as f:
    for path in selected:
        f.write(str(path.relative_to(SOURCE_DIR)) + "\n")

print(f"Sample list saved to: {sample_list}")
