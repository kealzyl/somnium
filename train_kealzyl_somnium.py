"""
Kealzyl Somnium — train a Deep Dream feature model from an image folder.

Example dataset:
    dataset/
        nature/image1.jpg
        textures/image2.png
        image3.jpg

The network is initialized from scratch and learns image representations from
augmented views. It can then be used to maximize spatial activations for Deep
Dream generation. This is not a pretrained ImageNet model, so semantic effects
depend on the quality and size of your dataset.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class ImageFolderDataset(Dataset):
    def __init__(self, root: Path, image_size: int):
        self.paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not self.paths:
            raise RuntimeError(f"No images found in {root.resolve()}. Add images to dataset/ or its subfolders.")
        self.transform = transforms.Compose([
            transforms.Resize(image_size + 16),
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.04)], p=0.5),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Skip corrupt files rather than crashing the whole training run.
        for offset in range(len(self.paths)):
            path = self.paths[(index + offset) % len(self.paths)]
            try:
                with Image.open(path) as im:
                    return self.transform(ImageOps.exif_transpose(im).convert("RGB"))
            except Exception:
                continue
        raise RuntimeError("All image files appear unreadable or corrupted.")


class SomniumFeatureNet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(cin, cout, stride=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.GELU(),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.GELU(),
            )

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.block1 = block(32, 48, 2)
        self.block2 = block(48, 96, 2)
        self.block3 = block(96, 128, 2)
        self.block4 = block(128, 160, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(nn.Linear(160, 256), nn.GELU(), nn.Linear(256, 160))

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.projector(self.pool(x).flatten(1))

    def dream_features(self, x):
        features = {}
        x = self.stem(x)
        features["stem"] = x
        x = self.block1(x)
        features["block1"] = x
        x = self.block2(x)
        features["block2"] = x
        x = self.block3(x)
        features["block3"] = x
        x = self.block4(x)
        features["block4"] = x
        return features


def augmented_view(x):
    y = x * random.uniform(0.85, 1.15) + random.uniform(-0.08, 0.08)
    y = y + torch.randn_like(y) * 0.025
    return y.clamp(0, 1)


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Kealzyl Somnium | device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    dataset = ImageFolderDataset(Path(args.dataset), args.image_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        drop_last=False,
    )
    print(f"Images found: {len(dataset)}")

    model = SomniumFeatureNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    amp_enabled = device.type == "cuda" and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "kealzyl_somnium.pth"
    start_epoch, best_loss = 0, float("inf")

    if args.resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_loss = float(ckpt.get("best_loss", best_loss))
        print(f"Resuming after epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss, total_items = 0.0, 0
        start_time = time.time()

        for images in loader:
            images = images.to(device, non_blocking=True)
            view1, view2 = images, augmented_view(images)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                z1, z2 = model(view1), model(view2)
                consistency_loss = F.mse_loss(z1, z2.detach())
                centered = z1 - z1.mean(dim=0, keepdim=True)
                std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
                variance_penalty = F.relu(0.5 - std).mean()
                loss = consistency_loss + 0.1 * variance_penalty

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            n = images.shape[0]
            total_loss += float(loss.detach().float()) * n
            total_items += n

        avg_loss = total_loss / max(1, total_items)
        best_loss = min(best_loss, avg_loss)
        print(f"Epoch {epoch + 1:03d}/{args.epochs} | loss {avg_loss:.6f} | best {best_loss:.6f} | {time.time()-start_time:.1f}s")

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_loss": best_loss,
            "model_name": "Kealzyl Somnium",
            "architecture": "SomniumFeatureNet",
            "image_size": args.image_size,
        }
        torch.save(checkpoint, checkpoint_path)
        if (epoch + 1) % args.save_every == 0:
            periodic = output_dir / f"kealzyl_somnium_epoch_{epoch + 1:03d}.pth"
            torch.save(checkpoint, periodic)
            print(f"Saved: {periodic}")

    weights_path = output_dir / "kealzyl_somnium_weights.pth"
    torch.save({
        "model": model.state_dict(),
        "model_name": "Kealzyl Somnium",
        "architecture": "SomniumFeatureNet",
        "image_size": args.image_size,
    }, weights_path)
    print(f"\nTraining complete.\nLatest checkpoint: {checkpoint_path}\nInference weights: {weights_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train Kealzyl Somnium on images in ./dataset")
    p.add_argument("--dataset", default="dataset", help="Dataset directory (default: dataset)")
    p.add_argument("--output", default="kealzyl_somnium_model", help="Checkpoint output directory")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=0, help="Keep 0 for easiest Windows compatibility")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
