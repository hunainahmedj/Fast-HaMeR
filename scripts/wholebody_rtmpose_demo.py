# Optional: RTMPose hand-only tracker (no HaMeR). Run from repo root.
import torch
import time
import cv2
import numpy as np

from rtmlib import Hand, PoseTracker, draw_skeleton

device = "cuda"
backend = "onnxruntime"
openpose_skeleton = False

cap = cv2.VideoCapture(0)
hand = PoseTracker(
    Hand,
    det_frequency=7,
    to_openpose=openpose_skeleton,
    mode="lightweight",
    backend=backend,
    device=device,
)

frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1
    if not success:
        break
    s = time.time()
    keypoints, scores = hand(frame)
    det_time = time.time() - s
    print("det: ", det_time)
    img_show = frame.copy()
    img_show = np.zeros(img_show.shape, dtype=np.uint8)
    img_show = draw_skeleton(
        img_show,
        keypoints,
        scores,
        openpose_skeleton=openpose_skeleton,
        kpt_thr=0.3,
        line_width=5,
    )
    img_show = cv2.resize(img_show, (960, 640))
    cv2.imshow("img", img_show)
    cv2.waitKey(10)
