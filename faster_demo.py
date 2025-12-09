import time

import cv2
import numpy as np

import torch
from rtmlib import PoseTracker, Hand, Wholebody, draw_skeleton, draw_bbox


def keypoints_to_bbox(keypoints, scores):

    valid_mask = scores > 0.5
    valid_kpts = keypoints[valid_mask]

    if len(valid_kpts) == 0:
        return None, 0

    # Compute bounding box (min/max x, y)
    min_x, min_y = np.min(valid_kpts, axis=0)
    max_x, max_y = np.max(valid_kpts, axis=0)

    # x_min, y_min, x_max, y_max = box
    width = max_x - min_x
    height = max_y - min_y

    max_side = max(width, height)

    # Adjust bbox to be square and centered
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    min_x = center_x - max_side / 2
    min_y = center_y - max_side / 2
    max_x = center_x + max_side / 2
    max_y = center_y + max_side / 2

    return [min_x, min_y, max_x, max_y], scores[valid_mask].mean()


def run():
    pass


def create_tracker(mode="", det_frequency: int = 10):
    pose_tracker = PoseTracker(
        Wholebody,
        det_frequency=det_frequency,
        to_openpose=openpose_skeleton,
        mode="lightweight",
        backend=backend,
        device=device,
    )


device = "cuda"  # cpu, cuda, mps
backend = "onnxruntime"  # opencv, onnxruntime, openvino
openpose_skeleton = False  # True for openpose-style, False for mmpose-style


left_hand_indices = list(range(91, 112))
right_hand_indices = list(range(112, 133))

# img = cv2.imread("assets/demo.jpg")

# keypoints, scores = pose_tracker(img)
# left_hand_indices = list(range(91, 112))
# right_hand_indices = list(range(112, 133))


# bboxes = []
# is_right = []

# for idx, hand_indices in enumerate([left_hand_indices, right_hand_indices]):
#     box, score = keypoints_to_bbox(
#         keypoints[0, hand_indices], scores=scores[0, hand_indices]
#     )
#     if box is not None:
#         bboxes.append(box)
#         is_right.append(1 if idx == 1 else 0)

# print(bboxes)

# img_show = draw_bbox(img, bboxes=bboxes)

# cv2.imshow("img", img_show)
# cv2.waitKey()


capture = cv2.VideoCapture(0)

frame_idx = 0
try:
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame_idx += 1

        img_show = cv2.flip(frame, 1).copy()

        start_time = time.time()
        keypoints, scores = pose_tracker(img_show)
        det_time = time.time() - start_time

        bboxes = []
        is_right = []

        for idx, hand_indices in enumerate([left_hand_indices, right_hand_indices]):
            box, score = keypoints_to_bbox(
                keypoints[0, hand_indices], scores=scores[0, hand_indices]
            )
            if box is not None:
                bboxes.append(box)
                is_right.append(1 if idx == 1 else 0)

        img_show = draw_bbox(img_show, bboxes)
        cv2.imshow("Camera", img_show)

        if cv2.waitKey(1) == ord("q"):
            break
finally:
    capture.release()
    cv2.destroyAllWindows()
