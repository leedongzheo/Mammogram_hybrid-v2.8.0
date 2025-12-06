#!/usr/bin/env python3
"""Compute dataset mean/std compatible with the pipeline's raw-intensity normalization.

The project's ToTensor transform does **not** rescale pixel intensities to [0, 1];
it simply casts the numpy array to float. Therefore, mean/std should typically be
computed on the original 8-bit values (0-255). Use the optional flag to rescale
if you intentionally want [0, 1] statistics.

Pass --label_dir and --crop_size to sample patches with the same BalanceCrop
strategy used during training; this better matches the paper's reported stats.
"""
import argparse
import os
import sys
from typing import Iterable, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import random


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(img_dir: str) -> Iterable[str]:
    for name in sorted(os.listdir(img_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMG_EXTS:
            yield os.path.join(img_dir, name)


def load_image(path: str, scale01: bool) -> np.ndarray:
    arr = imageio.imread(path)
    if arr.ndim == 2:  # grayscale HxW -> 1xHxW
        arr = arr[None, ...]
    elif arr.ndim == 3:  # HxWxC -> CxHxW
        arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported image dims {arr.shape} for {path}")

    arr = arr.astype(np.float64, copy=False)
    if scale01:
        arr = arr / 255.0
    return arr


def crop_bounds(start: int, end: int, max_size: int):
    if end - start > max_size:
        raise ValueError(f"Crop size {end-start} exceeds dimension {max_size}")
    if start < 0:
        end -= start
        start = 0
    if end > max_size:
        start -= end - max_size
        end = max_size
    return start, end


def crop_centroid(img: np.ndarray, center: Tuple[int, int], size: Tuple[int, int]):
    """Crop a 2D patch centered at ``center`` with the given ``size``.

    Args:
        img: CxHxW image.
        center: (y, x) center.
        size: (h, w) size.
    """

    h, w = img.shape[-2:]
    th, tw = size
    cy, cx = center
    y1, y2 = crop_bounds(cy - th // 2, cy + (th - th // 2), h)
    x1, x2 = crop_bounds(cx - tw // 2, cx + (tw - tw // 2), w)
    return img[..., y1:y2, x1:x2]


def compute_stats(
    img_dir: str,
    scale01: bool,
    *,
    label_dir: Optional[str] = None,
    crop_size: Optional[Tuple[int, int]] = None,
    balance_prob: float = 0.5,
    samples_per_image: int = 1,
):
    sums = None
    sumsq = None
    count = 0

    for path in list_images(img_dir):
        img = load_image(path, scale01)
        c, h, w = img.shape

        if label_dir is not None:
            label_path = os.path.join(label_dir, os.path.basename(path))
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Missing label for {path}: {label_path}")
            label = imageio.imread(label_path)
            if label.ndim != 2:
                raise ValueError(f"Expected 2D label, got shape {label.shape} for {label_path}")

            pos = np.argwhere(label > 0)
            neg = np.argwhere(label == 0)
            if crop_size is None:
                raise ValueError("crop_size must be set when using label_dir")
            for _ in range(samples_per_image):
                if len(neg) == 0 and len(pos) == 0:
                    raise RuntimeError(f"Label for {label_path} has no pixels to sample")
                elif len(neg) == 0:
                    center = tuple(pos[random.randrange(len(pos))])
                elif len(pos) == 0:
                    center = tuple(neg[random.randrange(len(neg))])
                else:
                    if random.random() <= balance_prob:
                        center = tuple(pos[random.randrange(len(pos))])
                    else:
                        center = tuple(neg[random.randrange(len(neg))])

                patch = crop_centroid(img, center, crop_size)
                flat = patch.reshape(c, -1)
                pixels = flat.shape[1]
                if sums is None:
                    sums = np.zeros(c, dtype=np.float64)
                    sumsq = np.zeros(c, dtype=np.float64)
                sums += flat.sum(axis=1)
                sumsq += (flat ** 2).sum(axis=1)
                count += pixels
        else:
            if sums is None:
                sums = np.zeros(c, dtype=np.float64)
                sumsq = np.zeros(c, dtype=np.float64)
            pixels = h * w
            flat = img.reshape(c, pixels)
            sums += flat.sum(axis=1)
            sumsq += (flat ** 2).sum(axis=1)
            count += pixels

    if count == 0:
        raise RuntimeError(f"No images found in {img_dir}")

    mean = sums / count
    std = np.sqrt(sumsq / count - mean ** 2)
    return mean, std


def write_stats(mean: np.ndarray, std: np.ndarray, output: str):
    with open(output, "w", encoding="utf-8") as f:
        f.write("mean " + " ".join(f"{m:.4f}" for m in mean) + "\n")
        f.write("std " + " ".join(f"{s:.4f}" for s in std) + "\n")



def main():
    parser = argparse.ArgumentParser(description="Compute mean/std for mammogram images")
    parser.add_argument("--img_dir", required=True, help="Directory containing images (e.g., data/img)")
    parser.add_argument(
        "--output", default="data/meanstd.txt", help="Where to save the mean/std file"
    )
    parser.add_argument(
        "--scale01",
        action="store_true",
        help="Divide pixel values by 255.0 before computing statistics (not used in default pipeline)",
    )
    parser.add_argument(
        "--label_dir",
        help="Optional directory with label masks (e.g., data/label) to mimic BalanceCrop sampling",
    )
    parser.add_argument(
        "--crop_size",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        default=None,
        help="Crop size H W (e.g., 512 384) when using label_dir",
    )
    parser.add_argument(
        "--balance_prob",
        type=float,
        default=0.5,
        help="Probability of choosing a positive center when label_dir is set (default: 0.5)",
    )
    parser.add_argument(
        "--samples_per_image",
        type=int,
        default=4,
        help="Number of random BalanceCrop samples per image when label_dir is set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducible sampling",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    mean, std = compute_stats(
        args.img_dir,
        args.scale01,
        label_dir=args.label_dir,
        crop_size=tuple(args.crop_size) if args.crop_size else None,
        balance_prob=args.balance_prob,
        samples_per_image=args.samples_per_image,
    )
    write_stats(mean, std, args.output)

    print(f"Processed {args.img_dir}")
    print("mean:", " ".join(f"{m:.4f}" for m in mean))
    print("std:", " ".join(f"{s:.4f}" for s in std))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
