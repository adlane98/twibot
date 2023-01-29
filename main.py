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
from utils import read_image, show_image
from yolov7.utils.general import non_max_suppression


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
    # source = Path(r"E:\twibot\twinit-dataset\DSC_3686.MOV")
    source = Path(r"E:\twibot\twinit-dataset\video-9")
    weight = Path(
        r"E:\twibot\yolov7\runs\train\yolov7-tiny-with-arm13\weights\best.pt"
    )

    # Load model
    model = Ensemble()
    ckpt = torch.load(weight, map_location="cpu")  # load

    model = ckpt["model"].float().eval()
    # model.append(
    #     ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())

    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    # Warmup
    example = torch.rand(1, 3, 384, 640)
    out = model(example)[0]
    out = non_max_suppression(out, 0.25, 0.45, classes=[0])
    # traced_script_module = torch_utils.TracedModel(model, device="cpu", img_size=(384,640))

    # traced_script_module = torch.jit.trace(model, example, strict=False)
    # traced_script_module.model.save(r"E:\twibot\yolov7\runs\train\yolov7-tiny-with-arm13\weights\traced_yolov7_model.pt")
    # out = traced_script_module(example)

    # Load sift
    sift = cv2.SIFT_create()

    # Load matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Loading video
    # video = cv2.VideoCapture(str(source))

    # Get frames
    frames_gray = list()
    success = True
    nb_frames = 0
    new_frames = []
    # while success:
    for image_path in source.glob("*.jpg"):
        image_path = Path(r"E:\\twibot\\twinit-dataset\\video-8\\9.jpg")
        start = time.time()

        # success, frame = video.read()
        frame = cv2.imread(str(image_path))
        # frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        nb_frames += 1
        print(f"{nb_frames} read")

        # augmented frames
        frame = cv2.resize(frame, (360, 640))
        # frames_gray = np.array(frames_gray)

        # transform frames to tensor
        frame_chan_first = np.transpose(frame[None, ...], (0, 3, 2, 1))
        frame_t = torch.Tensor(frame_chan_first)

        # mettre a l'echelle les coords en fonction de l'augmentation
        # frame_aug_t = T.Resize((360, 640))(frame_t)
        frame_aug_t = F.pad(frame_t, (0, 0, 12, 12), mode="constant", value=114)

        print(frame_aug_t.numpy().shape)

        frame_aug_t /= 255
        x_df = pd.DataFrame(frame_aug_t.numpy()[0, 1, ...])
        x_df.to_csv(
            'E:\\twibot\\twinit-dataset\\debug\\python1.txt',
            sep=";",
            header=False,
            index=False,
            float_format='%.6f'
        )

        print(frame_aug_t[0, 0, 150, 458])
        save_tensor("cpu", frame_aug_t)

        # call model
        pred = model.forward(frame_aug_t)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=[0])
        pred = pred[0][:, :4]

        pred -= torch.Tensor([0, 12, 0, 12]).cuda().int()
        pred = torch.clip(pred.int(), 0, None)
        pred *= 2

        # extraction des cartes
        cards = []
        kps = []
        dess = []
        i = 0
        for coord in pred:
            card = frame[coord[1]:coord[3], coord[0]:coord[2], :]
            cv2.imwrite(str(source / "cards" / f"{image_path.stem}-{i}.jpg"), card)

            # card = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

            i += 1
            # card = cv2.cvtColor(card, cv2.COLOR_RGB2GRAY)
            # card = cv2.resize(card, (128, 128))

            # cards.append(card)

            # resize des cartes

            # calcul des descripteurs sift
            # kp1, des1 = sift.detectAndCompute(card, None)
            # kps.append(kp1)
            # dess.append(des1)

        # faire les paires de cartes
    #     pairs_saved = []
    #     for pair_id in itertools.combinations(range(len(pred)), r=2):
    #         kp1 = kps[pair_id[0]]
    #         kp2 = kps[pair_id[1]]
    #
    #         matches = flann.knnMatch(dess[pair_id[0]], dess[pair_id[1]], k=2)
    #
    #         good = []
    #         for m, n in matches:
    #             if m.distance < 0.7 * n.distance:
    #                 good.append(m)
    #
    #         # sauvegarder les bonnes paires
    #         if len(good) > 50:
    #
    #             src_pts = np.float32(
    #                 [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    #             dst_pts = np.float32(
    #                 [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    #             M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    #
    #             if M is not None:
    #                 pairs_saved.append(pair_id)
    #
    #     # print(pairs_saved)
    #
    #     new_frame = frame
    #     for (p1, p2) in pairs_saved:
    #         for p in p1, p2:
    #             new_frame = cv2.rectangle(
    #                 new_frame,
    #                 (pred[p, 0], pred[p, 1]),
    #                 (pred[p, 2], pred[p, 3]),
    #                 color=(0, 255, 0),
    #                 thickness=2
    #             )
    #
    #     new_frames.append(new_frame)
    #     end = time.time()
    #     print(f"Processing time = {end - start}")
    #
    # out = cv2.VideoWriter(fr"E:\twibot\twinit-dataset\debug\output.avi", cv2.VideoWriter_fourcc(*'DIVX'), 25.0, (1280, 720))
    #
    # for frame in new_frames:
    #     out.write(frame)
    #     # When everything done, release the video capture and video write objects
    # out.release()

if __name__ == '__main__':
    # test_main()
    twibot()