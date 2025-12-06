# Mammogram_hybrid-v2.8.0
Implementation of the paper "A Unified Mammogram Analysis Method via Hybrid Deep Supervision".

This implementation now targets PyTorch v2.8.0.

Suggested core dependencies:

* `torch==2.8.0`
* `torchvision==0.19.0`
* `imageio`

Label expectations:

* Place segmentation masks under `data/label` with the same base filename as the image.
* Masks are treated as binary: background pixels are `0` and lesion pixels are any value > 0.
* For a normal (negative) image with no mass, the mask should therefore be entirely zeros.

Before running the code by conducting `python train.py`, please store your images and pixel-wise labels in data/img and data/label respectively, and modify the values in data/meanstd.txt according to your data.

Mean/std calculation:

* The built-in `ToTensor` transform **does not** rescale pixel intensities to [0, 1]; it casts the raw values (typically 0–255) to float. Therefore, `data/meanstd.txt` should normally contain statistics computed on the original 8-bit range (e.g., mean≈87, std≈68 for many mammogram sets).
* A helper script is provided to compute matching statistics. Example:

  ```bash
  python tools/compute_meanstd.py --img_dir data/img --output data/meanstd.txt
  ```

  Use `--scale01` only if you deliberately normalize images to [0, 1] elsewhere; otherwise leave it off to match the training pipeline.

* To reproduce the paper-style numbers (≈87/≈68) that follow the training pipeline’s 512×384 BalanceCrop, provide masks and crop size so statistics are collected on training-like patches instead of entire images:

  ```bash
  python tools/compute_meanstd.py \
    --img_dir data/img --label_dir data/label \
    --crop_size 512 384 --samples_per_image 8 \
    --output data/meanstd.txt
  ```
