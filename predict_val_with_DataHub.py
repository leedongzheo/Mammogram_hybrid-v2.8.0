import argparse
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from pathlib import Path
from tqdm import tqdm

import factory as fm
import model as M
import misc   # chứa DataHub
from validate import dice


# -------------------------------------------------------------
# Load model
# -------------------------------------------------------------
def load_model(device):
    mil_downfactor = 128
    drop = (.2,) * 3 + (.5,) * 5 + (.2,) * 3

    model = M.uresnet_16x11x2(
        1, 2, drop, mil_downfactor,
        upsampler=fm.BilinearUp2d()
    )
    model.to(device)
    return model


def load_checkpoint(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


# -------------------------------------------------------------
# Predict single sample
# -------------------------------------------------------------
def predict_one(model, img, mask, device, threshold):
    with torch.no_grad():
        img = img[None].to(device)  # (1,1,H,W)

        logits_stack, _ = model(img)   # multi-head
        logits = logits_stack[-1, 0]   # head cuối

        prob = F.softmax(logits, dim=0)[1]   # class 1
        pred = (prob > threshold).long().cpu()

    dsc = float(dice(pred, mask).cpu())
    return pred.numpy().astype("uint8"), dsc


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="../data")
    parser.add_argument("--val_split", default="split/val.txt")
    parser.add_argument("--datapath", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="pred_val")
    parser.add_argument(
        "--crop_type_test",
        default=None,
        choices=(None, "center", "random"),
        help=(
            "Test-time crop strategy. Use None to keep the original resolution. "
            "If the validation masks are fully black, make sure this matches the "
            "training setup so lesions are not cropped out."
        ),
    )
    parser.add_argument(
        "--crop_size_img_test",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Height/width for test-time image crops when crop_type_test is set.",
    )
    parser.add_argument(
        "--crop_size_label_test",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Height/width for test-time label crops when crop_type_test is set.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---------------------------------------------------------
    # Load DataHub – KHỚP HOÀN TOÀN training/test transform
    # ---------------------------------------------------------
    dh_kwargs = {
        "root": args.root,
        "train_split": None,
        "val_split": args.val_split,
        "test_split": None,
        "datapath": args.datapath,
        "train_batchsize": 1,
        "test_batchsize": 1,
        "modalities": ("label", "img"),
        "rand_flip": (1, 1),              # <<<<< SỬA CHỖ NÀY, KHÔNG ĐỂ None
        "crop_type": None,
        "crop_type_test": args.crop_type_test,
        "crop_size_img_test": args.crop_size_img_test,
        "crop_size_label_test": args.crop_size_label_test,
        "DataSet": misc.datasets.Dataset_SEGCLS_png,
        "num_workers": 0,
    }

    data_cube = misc.DataHub(**dh_kwargs)
    loader = data_cube.valloader()
    sn_list = data_cube.val_sn()

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------
    model = load_model(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    # ---------------------------------------------------------
    # Prepare output dir
    # ---------------------------------------------------------
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = []

    print("\nRunning inference on validation set...\n")

    # ---------------------------------------------------------
    # Predict all validation samples
    # ---------------------------------------------------------
    for i, (images, masks, cls_label) in enumerate(tqdm(loader)):
        img = images[0]
        mask = masks[0]
        sn = sn_list[i]

        pred_mask, dsc = predict_one(model, img, mask, device, args.threshold)

        # Save
        imageio.imwrite(outdir / f"{sn}_gt.png", (mask.numpy().astype("uint8") * 255))
        imageio.imwrite(outdir / f"{sn}_pred.png", (pred_mask * 255))

        metrics.append((sn, dsc))

    # ---------------------------------------------------------
    # Save metrics.txt
    # ---------------------------------------------------------
    with open(outdir / "metrics.txt", "w") as f:
        for sn, dsc in metrics:
            f.write(f"{sn}\t{dsc:.4f}\n")

    mean_dsc = sum(d for _, d in metrics) / len(metrics)
    print(f"\nDONE. Mean DSC = {mean_dsc:.4f}")
    print(f"Saved results to: {outdir}")


if __name__ == "__main__":
    main()
