from src.datasets.imagenet import ImageNetValDataset


dataset = ImageNetValDataset("C:/Users/10031/irp-hw2825/code/dataset/subset/val_subset")

print(len(dataset))

img, name = dataset[0]

print(name)
print(img.size)
