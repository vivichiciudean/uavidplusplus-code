import glob
import os
import numpy as np
import cv2
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import random

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# Default parameters DINOv2
# default_tile_h = 1092   # height of tile
# default_tile_w = 1092   # width of tile

# # Default parameters DINOv3
default_tile_h = 1088   # height of tile
default_tile_w = 1088   # width of tile

ignore_value = 255     # padding (255,255,255) for masks, and (0,0,0) for images

#TODO - se aplica pe TRAIN/VAL
##TODO (by you)
### APPLY THIS SCRIPT ON flat_uavid_train and flat_uavid_val (don`t forget to update the output dir)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="../data/flat_uavid_val")
    parser.add_argument("--output-img-dir", default="../data/val/Images")
    parser.add_argument("--output-mask-dir", default="../data/val/Labels")
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--split-size-h", type=int, default=default_tile_h, help="Tile height")
    parser.add_argument("--split-size-w", type=int, default=default_tile_w, help="Tile width")
    parser.add_argument("--stride-h", type=int, default=default_tile_h, help="Vertical stride (use < tile height for overlap)")
    parser.add_argument("--stride-w", type=int, default=default_tile_w, help="Horizontal stride (use < tile width for overlap)")
    return parser.parse_args()

def pad_to_tile(img, mask, split_size, ignore_value=255):
    h, w = img.shape[:2]
    target_h = ((h + split_size[0] - 1) // split_size[0]) * split_size[0]
    target_w = ((w + split_size[1] - 1) // split_size[1]) * split_size[1]

    pad_h = target_h - h
    pad_w = target_w - w

    img_padded = cv2.copyMakeBorder(
        img, 0, pad_h, 0, pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    mask_padded = cv2.copyMakeBorder(
        mask, 0, pad_h, 0, pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=(ignore_value, ignore_value, ignore_value)
    )

    return img_padded, mask_padded

def pad_to_tile_center(img, mask, split_size, ignore_value=255):
    h, w = img.shape[:2]
    target_h = ((h + split_size[0] - 1) // split_size[0]) * split_size[0]
    target_w = ((w + split_size[1] - 1) // split_size[1]) * split_size[1]

    pad_h_total = target_h - h
    pad_w_total = target_w - w

    # distribute padding equally on both sides
    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left

    img_padded = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    mask_padded = cv2.copyMakeBorder(
        mask, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(ignore_value, ignore_value, ignore_value)
    )

    return img_padded, mask_padded

def patch_format(inp):
    (input_dir, imgs_output_dir, masks_output_dir, mode, split_size, stride) = inp

    img_paths = sorted(glob.glob(os.path.join(input_dir, "Images", "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(input_dir, "Labels", "*.png")))

    assert len(img_paths) == len(mask_paths), "Mismatch between images and masks count"

    for img_path, mask_path in zip(img_paths, mask_paths):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)

        id = os.path.splitext(os.path.basename(img_path))[0]

        # pad
        img, mask = pad_to_tile_center(img.copy(), mask.copy(), split_size)

        # tiling
        k = 0
        for y in range(0, img.shape[0], stride[0]):
            for x in range(0, img.shape[1], stride[1]):
                img_tile = img[y:y + split_size[0], x:x + split_size[1]]
                mask_tile = mask[y:y + split_size[0], x:x + split_size[1]]

                if img_tile.shape[:2] == split_size and mask_tile.shape[:2] == split_size:
                    out_img_path = os.path.join(imgs_output_dir, f"{id}_{k}.png")
                    out_mask_path = os.path.join(masks_output_dir, f"{id}_{k}.png")
                    print(out_mask_path)
                    cv2.imwrite(out_img_path, img_tile)
                    cv2.imwrite(out_mask_path, mask_tile)
                else:
                    print(img_tile.shape)
                k += 1


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    input_dir = args.input_dir
    imgs_output_dir = args.output_img_dir
    masks_output_dir = args.output_mask_dir
    mode = args.mode
    split_size = (args.split_size_h, args.split_size_w)
    stride = (args.stride_h, args.stride_w)

    os.makedirs(imgs_output_dir, exist_ok=True)
    os.makedirs(masks_output_dir, exist_ok=True)

    inp = [(input_dir, imgs_output_dir, masks_output_dir, mode, split_size, stride)]

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(patch_format, inp)
    t1 = time.time()
    print(f"Images splitting took: {t1 - t0:.2f} s")
