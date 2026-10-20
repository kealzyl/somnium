"""
Kealzyl Somnium — Deep Dream image generator
Loads the custom CNN trained by train_kealzyl_somnium.py and amplifies its
intermediate activations in an input image.

Example:
    python kealzyl_somnium_dream.py --input photo.jpg
    python kealzyl_somnium_dream.py --input photo.jpg --intensity 1.5 --steps 80
    python kealzyl_somnium_dream.py --input photo.jpg --layer block3 --octaves 4

Install:
    python -m pip install torch torchvision pillow
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms


class SomniumFeatureNet(nn.Module):
    """Must match the architecture used in train_kealzyl_somnium.py."""

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
        self.projector = nn.Sequential(
            nn.Linear(160, 256), nn.GELU(), nn.Linear(256, 160)
        )

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


def load_model(weights_path: Path, device: torch.device):
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            "Train the model first, or pass --model with the correct .pth path."
        )

    try:
        checkpoint = torch.load(
            weights_path, map_location=device, weights_only=True
        )
    except TypeError:  # Compatibility with older PyTorch releases
        checkpoint = torch.load(weights_path, map_location=device)

    state = checkpoint.get("model", checkpoint)
    model = SomniumFeatureNet().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_image(path: Path, max_side: int, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_size = image.size

        # Keep aspect ratio while limiting memory usage.
        scale = min(1.0, max_side / max(image.size))
        new_size = (
            max(32, int(image.width * scale)),
            max(32, int(image.height * scale)),
        )
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    return tensor, original_size


def save_image(tensor: torch.Tensor, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().clamp(0, 1).squeeze(0).cpu()
    transforms.ToPILImage()(image).save(output_path, quality=95)


def dream_once(model, image, layer, intensity, steps, step_size, device):
    """
    Gradient-ascent Deep Dream pass. The model is frozen, but gradients are
    calculated for the image so its selected layer's activations are amplified.
    """
    result = image.detach().clone()
    for step in range(steps):
        result = result.detach().requires_grad_(True)
        features = model.dream_features(result)
        activations = features[layer]

        # Encourage strong spatial features without allowing one large value
        # to dominate the objective.
        objective = activations.square().mean()
        gradient = torch.autograd.grad(objective, result, only_inputs=True)[0]

        # Normalize gradient per image for predictable intensity control.
        gradient = gradient / (gradient.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8)
        with torch.no_grad():
            result = result + (step_size * intensity) * gradient
            result.clamp_(0, 1)

        if (step + 1) % 10 == 0 or step == 0 or step == steps - 1:
            print(f"  Step {step + 1:03d}/{steps} | activation objective: {objective.item():.6f}")

    return result.detach()


def multi_scale_dream(model, image, args, device):
    """Process small-to-large image scales to add patterns at multiple sizes."""
    current = image
    original_h, original_w = image.shape[-2:]
    scale_factor = args.octave_scale

    # Construct octave sizes from smallest to largest, preserving aspect ratio.
    sizes = []
    for octave in range(args.octaves - 1, -1, -1):
        scale = scale_factor ** octave
        h = max(32, int(original_h / scale))
        w = max(32, int(original_w / scale))
        sizes.append((h, w))
    sizes = list(dict.fromkeys(sizes))

    detail = torch.zeros_like(image)
    for octave_index, (h, w) in enumerate(sizes, start=1):
        base = F.interpolate(image, size=(h, w), mode="bilinear", align_corners=False)
        if detail.shape[-2:] != (h, w):
            detail = F.interpolate(detail, size=(h, w), mode="bilinear", align_corners=False)
        base = (base + detail * args.octave_blend).clamp(0, 1)

        print(f"\nOctave {octave_index}/{len(sizes)} — {w}x{h}")
        dreamed = dream_once(
            model, base, args.layer, args.intensity, args.steps,
            args.step_size, device
        )
        detail = dreamed - base
        current = dreamed

    return F.interpolate(current, size=(original_h, original_w), mode="bilinear", align_corners=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Deep Dream image using a trained Kealzyl Somnium model."
    )
    parser.add_argument("--input", required=True, help="Input image file")
    parser.add_argument("--output", default="somnium_dream.png", help="Output image path")
    parser.add_argument(
        "--model", default="kealzyl_somnium_model/kealzyl_somnium_weights.pth",
        help="Path to the trained weights file"
    )
    parser.add_argument("--layer", choices=["stem", "block1", "block2", "block3", "block4"], default="block3")
    parser.add_argument("--intensity", type=float, default=1.0, help="Dream strength (try 0.3–2.0)")
    parser.add_argument("--steps", type=int, default=40, help="Gradient ascent steps per octave")
    parser.add_argument("--step-size", type=float, default=0.005, help="Small update size per step")
    parser.add_argument("--octaves", type=int, default=3, help="Number of image scales (1–5 recommended)")
    parser.add_argument("--octave-scale", type=float, default=1.4, help="Scale ratio between octaves")
    parser.add_argument("--octave-blend", type=float, default=0.8, help="How much previous-scale detail to carry over")
    parser.add_argument("--max-side", type=int, default=768, help="Maximum input image dimension")
    parser.add_argument("--cpu", action="store_true", help="Force CPU instead of CUDA")
    args = parser.parse_args()

    if args.intensity < 0:
        parser.error("--intensity must be >= 0")
    if args.steps < 1 or args.octaves < 1:
        parser.error("--steps and --octaves must be >= 1")
    if args.step_size <= 0:
        parser.error("--step-size must be > 0")
    if args.octave_scale <= 1:
        parser.error("--octave-scale must be > 1")
    if args.max_side < 32:
        parser.error("--max-side must be >= 32")
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Kealzyl Somnium Deep Dream | device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = load_model(Path(args.model), device)
    image, _ = load_image(Path(args.input), args.max_side, device)
    print(f"Input size: {image.shape[-1]}x{image.shape[-2]}")
    print(f"Layer: {args.layer} | intensity: {args.intensity}")

    dreamed = multi_scale_dream(model, image, args, device)
    output = Path(args.output)
    save_image(dreamed, output)
    print(f"\nSaved Deep Dream image: {output.resolve()}")


if __name__ == "__main__":
    main()
