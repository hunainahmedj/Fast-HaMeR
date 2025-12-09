import os
import sys
import time

import cv2
import numpy as np
from rtmlib import PoseTracker, Wholebody, draw_bbox, draw_skeleton

from pose_tracker import create_pose_tracker


def main():

    pose_tracker = create_pose_tracker()
    capture = cv2.VideoCapture("http://192.168.0.102:8000/camera/mjpeg")

    while True:
        ret, frame = capture.read()
        if not ret:
            print("No frame is being captured from the camera.")
            break

        frame = cv2.flip(frame, 1).copy()

        tracker_start = time.time()
        keypoints, scores = pose_tracker(frame)
        tracker_end = time.time()

        frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)

        cv2.putText(
            frame,
            f"FPS: {(1/(tracker_end-tracker_start)):.0f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    main()
