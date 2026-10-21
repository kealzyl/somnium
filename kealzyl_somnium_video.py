
"""
Kealzyl Somnium — Video Deep Dream
Dreams video frames using the existing trained CNN, with optional
optical-flow temporal blending and original audio preservation.

Example:
    python kealzyl_somnium_video.py --input clip.mp4
    python kealzyl_somnium_video.py --input clip.mp4 --intensity 1.2
    python kealzyl_somnium_video.py --input clip.mp4 --layer block3 --octaves 3
    python kealzyl_somnium_video.py --input clip.mp4 --temporal-blend 0.2
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from kealzyl_somnium_dream import load_model, dream_once


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video Deep Dream with Kealzyl Somnium."
    )

    parser.add_argument("--input", required=True, help="Input video")
    parser.add_argument("--output", default="somnium_dream.mp4")
    parser.add_argument(
        "--model",
        default="kealzyl_somnium_model/kealzyl_somnium_weights.pth",
    )
    parser.add_argument(
        "--layer",
        choices=["stem", "block1", "block2", "block3", "block4"],
        default="block3",
    )
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=0.005)
    parser.add_argument("--octaves", type=int, default=3)
    parser.add_argument("--octave-scale", type=float, default=1.4)
    parser.add_argument("--octave-blend", type=float, default=0.8)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument(
        "--temporal-blend",
        type=float,
        default=0.15,
        help="Previous-frame influence, from 0 (off) to 0.5",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Do not copy audio from the original video",
    )
    parser.add_argument("--cpu", action="store_true")

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
    if not 0 <= args.temporal_blend <= 0.5:
        parser.error("--temporal-blend must be between 0 and 0.5")

    return args


def resize_frame(frame, max_side):
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))

    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(32, round(w * scale)), max(32, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    return frame


def frame_to_tensor(frame, device):
    # OpenCV uses BGR; the model expects RGB.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    tensor = torch.from_numpy(
        np.asarray(image, dtype=np.float32).copy()
    ).permute(2, 0, 1).unsqueeze(0) / 255.0

    return tensor.to(device)


def tensor_to_frame(tensor):
    rgb = (
        tensor.detach()
        .clamp(0, 1)
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    rgb = (rgb * 255).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def dream_frame(model, image, args, device):
    """Apply multi-scale Deep Dream to a single video frame."""
    original_h, original_w = image.shape[-2:]
    detail = torch.zeros_like(image)

    sizes = []
    for octave in range(args.octaves - 1, -1, -1):
        scale = args.octave_scale ** octave
        size = (
            max(32, int(original_h / scale)),
            max(32, int(original_w / scale)),
        )
        if size not in sizes:
            sizes.append(size)

    for h, w in sizes:
        base = F.interpolate(
            image,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        if detail.shape[-2:] != (h, w):
            detail = F.interpolate(
                detail,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

        base = (base + detail * args.octave_blend).clamp(0, 1)

        # Reuse the existing model's gradient-ascent implementation.
        dreamed = dream_once(
            model,
            base,
            args.layer,
            args.intensity,
            args.steps,
            args.step_size,
            device,
        )

        detail = dreamed - base

    return F.interpolate(
        dreamed,
        size=(original_h, original_w),
        mode="bilinear",
        align_corners=False,
    ).clamp(0, 1)


def stabilize_frame(current, previous_gray, previous_dream, blend):
    """
    Warp the previous dreamed frame into the current frame's coordinates
    using optical flow, then blend a small amount to reduce flicker.
    """
    if previous_gray is None or previous_dream is None or blend == 0:
        return current

    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    # Backward flow: for each current pixel, locate its previous-frame pixel.
    flow = cv2.calcOpticalFlowFarneback(
        current_gray,
        previous_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )

    h, w = current_gray.shape
    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    map_x = x + flow[..., 0]
    map_y = y + flow[..., 1]

    warped = cv2.remap(
        previous_dream,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    # Avoid temporal blending at pixels that map outside the previous frame.
    valid = (
        (map_x >= 0)
        & (map_x < w - 1)
        & (map_y >= 0)
        & (map_y < h - 1)
    )

    result = current.copy()
    result[valid] = cv2.addWeighted(
        current[valid], 1.0 - blend,
        warped[valid], blend,
        0,
    )

    return result


def main():
    args = parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise FileNotFoundError(f"Video not found: {input_path}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print(f"Kealzyl Somnium Video Deep Dream | device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = load_model(Path(args.model), device)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError("Could not determine the video's frame rate.")

    if total <= 0:
        cap.release()
        raise RuntimeError("The video contains no readable frames.")

    # Read one frame to determine the processed output dimensions.
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read the first video frame.")

    first = resize_frame(first, args.max_side)
    height, width = first.shape[:2]

    # OpenCV needs dimensions compatible with the chosen video codec.
    width -= width % 2
    height -= height % 2
    if width < 32 or height < 32:
        cap.release()
        raise RuntimeError("Video dimensions are too small to process.")

    silent_path = output_path.with_name(
        output_path.stem + "_silent.mp4"
    )

    writer = cv2.VideoWriter(
        str(silent_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {silent_path}")

    previous_gray = None
    previous_dream = None
    started = time.monotonic()
    processed = 0

    try:
        with torch.inference_mode(False):
            while True:
                if processed == 0:
                    frame = first
                    ok = True
                else:
                    ok, frame = cap.read()

                if not ok:
                    break

                frame = resize_frame(frame, args.max_side)
                frame = cv2.resize(
                    frame, (width, height),
                    interpolation=cv2.INTER_AREA,
                )

                # Keep the temporal history in the same coordinate system.
                frame = stabilize_frame(
                    frame,
                    previous_gray,
                    previous_dream,
                    args.temporal_blend,
                )

                image = frame_to_tensor(frame, device)

                # The model is frozen, but the input needs gradients.
                dreamed = dream_frame(model, image, args, device)
                result = tensor_to_frame(dreamed)

                writer.write(result)

                previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                previous_dream = result
                processed += 1

                if processed % 5 == 0 or processed == total:
                    elapsed = time.monotonic() - started
                    rate = processed / max(elapsed, 1e-6)
                    remaining = max(0, total - processed) / max(rate, 1e-6)

                    print(
                        f"\rFrames: {processed}/{total} "
                        f"({100 * processed / total:.1f}%) | "
                        f"{rate:.2f} FPS | ETA {remaining:.0f}s",
                        end="",
                        flush=True,
                    )

    finally:
        cap.release()
        writer.release()

    print()

    if processed == 0:
        silent_path.unlink(missing_ok=True)
        raise RuntimeError("No video frames were processed.")

    # Preserve the original audio when FFmpeg is installed.
    ffmpeg = shutil.which("ffmpeg")
    muxed = False

    if not args.no_audio and ffmpeg:
        temp_output = output_path.with_name(
            output_path.stem + "_muxed.mp4"
        )

        command = [
            ffmpeg, "-y",
            "-i", str(silent_path),
            "-i", str(input_path),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(temp_output),
        ]

        print("Muxing output video and original audio...")
        result = subprocess.run(command, check=False)

        if result.returncode == 0 and temp_output.is_file():
            temp_output.replace(output_path)
            muxed = True
        else:
            temp_output.unlink(missing_ok=True)
            print("Warning: FFmpeg audio muxing failed.")

    if not muxed:
        if not args.no_audio and not ffmpeg:
            print("Warning: FFmpeg not found; saving a silent video.")
        silent_path.replace(output_path)

    print(f"Processed frames: {processed}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
