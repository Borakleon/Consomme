import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

# JAX-origin ViT checkpoints (orig_in21k_ft_in1k, augreg*) use 0.5 norm + 0.9 crop.
# DeiT and MAE use ImageNet norm + 0.875 crop.
INCEPTION_MEAN = (0.5, 0.5, 0.5)
INCEPTION_STD  = (0.5, 0.5, 0.5)
IMAGENET_MEAN  = (0.485, 0.456, 0.406)
IMAGENET_STD   = (0.229, 0.224, 0.225)


def get_preprocess_params(model_name):
    """Returns (mean, std, crop_pct) for the checkpoint family."""
    name = model_name.lower()
    if "mae" in name:                      # MAE checkpoints use ImageNet norm
        return IMAGENET_MEAN, IMAGENET_STD, 224 / 256
    if name.startswith("vit"):             # JAX-origin ViT (orig / augreg)
        return INCEPTION_MEAN, INCEPTION_STD, 0.9
    return IMAGENET_MEAN, IMAGENET_STD, 224 / 256   # deit, others


def get_dataloader(
    data_dir,
    model_name,
    split="train",
    batch_size=64,
    num_workers=8,
    pin_memory=True,
):
    assert split in ["train", "val"]
    mean, std, crop_pct = get_preprocess_params(model_name)
    print(f"[data] {model_name} | split={split} | mean={mean} std={std} crop_pct={crop_pct:.4f}")

    if split == "train":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        shuffle = True
    else:
        resize = int(round(224 / crop_pct))          # 249 for 0.9, 256 for 0.875
        transform = transforms.Compose([
            transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        shuffle = False

    dataset = datasets.ImageFolder(root=f"{data_dir}/{split}", transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )