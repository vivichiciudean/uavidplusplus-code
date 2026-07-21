import torch
import torch.nn.functional as F

def inference_tiled(model, image, tile_size=(1088, 1088), overlap=0, device='cpu', num_labels=11):
    _, H, W = image.shape
    th, tw = tile_size
    stride_h, stride_w = th - overlap, tw - overlap

    # Canvas for accumulating logits
    logits_full = torch.zeros((1, num_labels, H, W), device=device)
    count_map = torch.zeros((1, 1, H, W), device=device)

    # Slide over the image and compute logits for each tile
    for y in range(0, H, stride_h):
        for x in range(0, W, stride_w):
            y1, y2 = y, min(y + th, H)
            x1, x2 = x, min(x + tw, W)

            tile = torch.zeros((3, th, tw), device=device)
            tile[:, :y2-y1, :x2-x1] = image[:, y1:y2, x1:x2]
            tile = tile.unsqueeze(0)

            with torch.no_grad():
                out = model(tile)
                logits_tile = F.interpolate(out.logits, size=(th, tw), mode="bilinear", align_corners=False)

            logits_tile = logits_tile[:, :, :y2-y1, :x2-x1]

            logits_full[:, :, y1:y2, x1:x2] += logits_tile
            count_map[:, :, y1:y2, x1:x2] += 1

    # Normalize overlapping regions
    logits_full = logits_full / count_map
    return logits_full