import itertools
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn

from models.common import Conv
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.datasets import LoadImages
from yolov7.utils.plots import plot_one_box

colors_hex = [
    "#c71585",
    "#0000cd",
    "#00ff00",
    "#ffff00",
    "#ff4500",
    "#2f4f4f",
    "#228b22",
    "#00ffff",
    "#1e90ff",
    "#ffdead"
]
colors_rgb = []
for c in colors_hex:
    c = c.lstrip("#")
    colors_rgb.append(tuple(int(c[i:i+2], 16) for i in (0, 2, 4)))


def load_model(model_path):
    # Load model
    ckpt = torch.load(model_path)  # load

    model = ckpt["model"].float().eval()

    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    # Warmup
    example = torch.rand(1, 3, 384, 640).cuda()
    _ = model(example)[0]

    return model


def set_output_metadate(output_path, vid_cap):
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    return fps, w, h, vid_writer


def normalize(arr):
    normalized = torch.tensor(arr, dtype=torch.float32, device="cuda")
    normalized = normalized.float()  # uint8 to fp16/32
    normalized /= 255.0  # 0 - 255 to 0.0 - 1.0
    if normalized.ndimension() == 3:
        normalized = normalized.unsqueeze(0)

    return normalized


def cards_extraction(predictions, input_img, sift):
    # extraction des cartes
    kps = []
    dess = []
    xyxys = []
    for *xyxy, conf, cls in reversed(predictions):
        xyxy = [elem.int() for elem in xyxy]
        card = input_img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :]
        xyxys.append(xyxy)

        # calcul des descripteurs sift
        kp1, des1 = sift.detectAndCompute(card, None)
        kps.append(kp1)
        dess.append(des1)

    return kps, dess, xyxys


def make_card_pairs(predictions, kps, dess, flann):
    # faire les paires de cartes
    pairs_saved = []
    for pair_id in itertools.combinations(range(len(predictions)), r=2):
        kp1 = kps[pair_id[0]]
        kp2 = kps[pair_id[1]]

        if dess[pair_id[0]] is None or dess[pair_id[1]] is None:
            print("None")
            continue

        if len(kp1) < 2 or len(kp2) < 2:
            print("size")
            continue

        matches = flann.knnMatch(dess[pair_id[0]], dess[pair_id[1]], k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.4 * n.distance:
                good.append(m)

        # sauvegarder les bonnes paires
        if len(good) > 5:

            src_pts = np.float32(
                [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                pairs_saved.append(pair_id)

    return pairs_saved


def plot_pairs(pairs_saved, xyxys, input_img):
    for c, (p1, p2) in enumerate(pairs_saved):
        for p in p1, p2:
            try:
                plot_one_box(xyxys[p], input_img, label="", color=colors_rgb[c], line_thickness=5)
            except IndexError:
                pass

    return input_img


def match_cards(predictions, tensor_shape, input_img, sift, flann):
    # Rescale boxes from img_size to im0 size
    predictions[:, :4] = scale_coords(tensor_shape, predictions[:, :4], input_img.shape).round()
    kps, dess, xyxys = cards_extraction(predictions, input_img, sift)
    pairs_saved = make_card_pairs(predictions, kps, dess, flann)
    print(pairs_saved)

    return plot_pairs(pairs_saved, xyxys, input_img)


def twibot():
    source = Path(r"/home/adlane/projets/twinit-dataset/DSC_5508-small.mp4")
    weight = Path(
        r"/home/adlane/projets/twibot-weights/card-detection/yolov7.pt"
    )
    output = Path(r"/home/adlane/projets/twibot/debug/output.mp4")

    model = load_model(weight)

    # Load sift
    sift = cv2.SIFT_create()

    # Load matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    #Load video
    dataset_video = LoadImages(source, img_size=640)

    # Get frames
    metadata_flag = False
    for path, img, im0s, vid_cap in dataset_video:
        start = time.time()

        if not metadata_flag:
            fps, w, h, vid_writer = set_output_metadate(output, vid_cap)
            metadata_flag = True

        img = normalize(img)

        # call model
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=[0])[0]

        if len(pred):
            res_img = match_cards(pred, img.shape[2:], im0s, sift, flann)

        end = time.time()
        print(f"Processing time = {end - start}")

        vid_writer.write(res_img)

    vid_writer.release()
    vid_cap.release()


if __name__ == '__main__':
    # test_main()
    twibot()
