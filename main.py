import io
import itertools
from pathlib import Path
import time

import cv2
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch import nn
from yolov7.utils import torch_utils

from game import Game
from image import TwinitImage, compute_card_diff
from models.common import Conv
from models.experimental import Ensemble
from io_image import read_image, show_image
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


def test_main():
    results_path = Path(r"E:\twibot\twinit-dataset\benchmarks")
    base_path = Path(r"E:\twibot\twinit-dataset\video-6")

    game = Game(
        Path(r"E:\twibot\twinit-dataset\twinit-ref.png"),
        Path(r"E:\twibot\yolov7\runs\train\yolov7-tiny-with-arm13\weights\best.pt")
    )

    start_time = time.time()
    # for image_path in base_path.glob("*.jpg"):
    image_path = base_path / "1-0.jpg"
    image = TwinitImage(image_path, game, results_path)
    image.segment_cards()

    print("Twinit image = ", time.time() - start_time)
    time00 = time.time()
    all_cards = []
    for i, card in enumerate(image.cards):
        time0 = time.time()
        card.crop_card()
        print("Crop card = ", time.time() - time0)

        time1 = time.time()
        card.straighten(i)
        print("Straighten card = ", time.time() - time1)

        time2 = time.time()
        card.recolor()
        print("Recolor = ", time.time() - time2)
        # cv2.imwrite(str(
        #         card.image_belonged.results_path /
        #         f"{card.image_belonged.image_path.stem}-{i}.png"
        #     ), cv2.cvtColor(card.recolored_card, cv2.COLOR_RGB2BGR)
        # )
        card.compute_rotations()
        all_cards.append(card)

    print("Récupération des cartes = ", time.time() - time00)

    indexes_to_keep = []
    for i in range(len(all_cards)):
        for j in range(i+1, len(all_cards)):
            d = compute_card_diff(all_cards[i], all_cards[j])
            if any(np.array(d) > 0.64):
                indexes_to_keep.append(i)
                indexes_to_keep.append(j)
            print(f"({i}, {j}) = {compute_card_diff(all_cards[i], all_cards[j])}")

    print("Récupération des similarités = ", time.time() - start_time)

    image_detection = image.draw_card_countours(list(set(indexes_to_keep)))

    cv2.imwrite(str(
            results_path /
            f"{image_path.stem}-res.png"
        ), cv2.cvtColor(image_detection, cv2.COLOR_RGB2BGR)
    )
    print(time.time() - start_time)
    # whatever

        # card.get_hist()

        # box = keep_contours[0]
        #
        # maxl = np.max(box[:, 1])
        # minl = np.min(box[:, 1])
        # maxc = np.max(box[:, 0])
        # minc = np.min(box[:, 0])
        #
        # print(box)
        # print(maxl)
        # print(minl)
        # print(maxc)
        # print(minc)
        #
        # bbox = image[minl:maxl, minc:maxc, :]
        #
        # cv2.imwrite(str(results_path / "bbox.png"), bbox)

        # return keep_contours


def save_tensor(device, my_tensor):
    print("[python] my_tensor: ", my_tensor)
    f = io.BytesIO()
    torch.save(my_tensor, f, _use_new_zipfile_serialization=True)
    with open('twinit-dataset/debug/py_tensor.pt', "wb") as out_f:
        # Copy the BytesIO stream to the output file
        out_f.write(f.getbuffer())


def twibot():
    source = Path(r"/home/adlane/projets/twinit-dataset/dsc_3686.mp4")
    weight = Path(
        r"/home/adlane/projets/twibot-weights/card-detection/yolov7.pt"
    )

    # Load model
    ckpt = torch.load(weight)  # load

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
    frames_gray = list()
    new_frames = []
    metadata_flag = False
    for path, img, im0s, vid_cap in dataset_video:
        start = time.time()

        img = torch.from_numpy(img).to("cuda")
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if not metadata_flag:
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vid_writer = cv2.VideoWriter(fr"/home/adlane/projets/twibot/debug/output.mp4",
                                         cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            metadata_flag = True

        # call model
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=[0])[0]

        if len(pred):
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0s.shape).round()

            # extraction des cartes
            cards = []
            kps = []
            dess = []
            xyxys = []
            for *xyxy, conf, cls in reversed(pred):
                xyxy = [elem.int() for elem in xyxy]
                card = im0s[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :]
                cards.append(card)
                xyxys.append(xyxy)

                # calcul des descripteurs sift
                kp1, des1 = sift.detectAndCompute(card, None)
                kps.append(kp1)
                dess.append(des1)

            # faire les paires de cartes
            pairs_saved = []
            for pair_id in itertools.combinations(range(len(pred)), r=2):
                kp1 = kps[pair_id[0]]
                kp2 = kps[pair_id[1]]

                if dess[pair_id[0]] is None or dess[pair_id[1]] is None:
                    continue

                if len(kp1) < 2 or len(kp2) < 2:
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

            print(pairs_saved)

            for c, (p1, p2) in enumerate(pairs_saved):
                for p in p1, p2:
                    try:
                        plot_one_box(xyxys[p], im0s, label="", color=colors_rgb[c], line_thickness=5)
                    except IndexError:
                        pass
                    

        end = time.time()
        print(f"Processing time = {end - start}")

        vid_writer.write(im0s)

    vid_writer.release()
    vid_cap.release()

if __name__ == '__main__':
    # test_main()
    twibot()
