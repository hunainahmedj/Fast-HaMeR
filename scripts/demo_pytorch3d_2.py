# Optional demo: PyTorch3D mesh rendering (variant 2). Run from repo root.
import os
import sys
import time
import argparse

import cv2
import numpy as np

import torch
import torch_tensorrt
from pytorch3d_renderer import MeshPyTorch3DRenderer

from rtmlib import PoseTracker, Wholebody, draw_bbox

from hamer.utils import recursive_to
from hamer.models import load_hamer, load_efficient_hamer, DEFAULT_CHECKPOINT

from hamer.utils.renderer import cam_crop_to_full
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD


LEFT_HAND_INDICES = list(range(91, 112))
RIGHT_HAND_INDICES = list(range(112, 133))
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def keypoints_to_bbox(keypoints, scores):
    """Create a bounding box around the keypoints.

    Returns:
        tuple(list[float], float): Return bounding box around the keypoints and the mean score.
    """

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


def create_pose_tracker(
    mode: str = "lightweight",
    backend: str = "onnxruntime",
    device: str = "cuda",
    det_frequency: int = 10,
):
    return PoseTracker(
        Wholebody,
        mode=mode,
        device=device,
        backend=backend,
        to_openpose=False,
        det_frequency=det_frequency,
    )


def main():
    parser = argparse.ArgumentParser(description="Fast-HaMeR demo with PyTorch3D (variant 2).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--efficient_hamer",
        action="store_true",
        default=False,
        help="Use efficient hamer",
    )
    parser.add_argument(
        "--rescale_factor", type=float, default=2.0, help="Factor for padding the bbox"
    )
    parser.add_argument(
        "--stream_camera",
        action="store_true",
        default=False,
        help="Stream camera from a video stream.",
    )
    parser.add_argument(
        "--compile_network",
        action="store_true",
        default=False,
        help="Use tensorRT to compile the network.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference/fitting"
    )

    args = parser.parse_args()
    checkpoint = args.checkpoint or DEFAULT_CHECKPOINT

    if args.efficient_hamer:
        model, model_cfg = load_efficient_hamer(checkpoint)
    else:
        model, model_cfg = load_hamer(checkpoint, init_renderer=False)

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = model.to(device)
    model.eval()

    width_dim = 192 if model_cfg.MODEL.BACKBONE.TYPE == "vit" else 256
    if args.compile_network:
        backbone_input = [
            torch_tensorrt.Input(
                min_shape=(1, 3, 256, width_dim),
                opt_shape=(2, 3, 256, width_dim),
                max_shape=(8, 3, 256, width_dim),
            )
        ]
        trt_backbone = torch_tensorrt.compile(
            model.backbone,
            inputs=backbone_input,
            enabled_precisions={
                torch.float16
            },  # Use FP32 precision (use FP16 if supported)
            # device=device,
        )
        model.backbone = trt_backbone
        trt_backbone = None

        trt_transformer = torch_tensorrt.compile(
            model.mano_head.transformer,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, 1, 1), opt_shape=(2, 1, 1), max_shape=(8, 1, 1)
                ),
                torch_tensorrt.Input(
                    min_shape=(1, width_dim, 1280),
                    opt_shape=(2, width_dim, 1280),
                    max_shape=(8, width_dim, 1280),
                ),
            ],
            enabled_precisions={
                torch.float16
            },  # Use FP32 precision (use FP16 if supported)
            device=device,
        )
        model.mano_head.transformer = trt_transformer
        trt_transformer = None

    # load pose tracker
    pose_tracker = create_pose_tracker()

    # start capturing the camera
    camera_source = (
        0 if not args.stream_camera else "http://192.168.0.106:8000/camera/mjpeg"
    )
    capture = cv2.VideoCapture(camera_source)

    try:
        while True:
            start_loop = time.time()
            ret, frame = capture.read()
            if not ret:
                print("No frame is being captured from the camera.")
                break

            # flig the image horizontally
            frame = cv2.flip(frame, 1).copy()

            tracker_start = time.time()
            keypoints, scores = pose_tracker(frame)
            tracker_end = time.time()

            bboxes = []
            is_right = []

            # iterate over left and right hand keypoints cluster to get bounding boxes around the hands
            for idx, hand_indices in enumerate([LEFT_HAND_INDICES, RIGHT_HAND_INDICES]):
                # get the box and score for each hand
                box, score = keypoints_to_bbox(
                    keypoints=keypoints[0, hand_indices], scores=scores[0, hand_indices]
                )
                if box is not None:
                    bboxes.append(box)
                    is_right.append(1 if idx == 1 else 0)

            # if not bounding box found skip the inference
            if len(bboxes) != 0:

                boxes = np.stack(bboxes)
                right = np.stack(is_right)

                dataset = ViTDetDataset(
                    model_cfg, frame, boxes, right, rescale_factor=args.rescale_factor
                )
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
                )

                all_verts = []
                all_cam_t = []
                all_right = []

                # start inference with HaMeR
                for batch in dataloader:
                    batch = recursive_to(batch, device)
                    with torch.no_grad():
                        start_time = time.time()
                        out = model(batch)
                        fps = 1 / (time.time() - start_time)

                    multiplier = 2 * batch["right"] - 1
                    pred_cam = out["pred_cam"]
                    pred_cam[:, 1] = multiplier * pred_cam[:, 1]

                    img_size = batch["img_size"].float()
                    box_size = batch["box_size"].float()
                    box_center = batch["box_center"].float()

                    scaled_focal_length = (
                        model_cfg.EXTRA.FOCAL_LENGTH
                        / model_cfg.MODEL.IMAGE_SIZE
                        * img_size.max()
                    )
                    pred_cam_t_full = (
                        cam_crop_to_full(
                            pred_cam,
                            box_center,
                            box_size,
                            img_size,
                            scaled_focal_length,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    batch_size = batch["img"].shape[0]
                    for n in range(batch_size):
                        # Get filename from path img_path
                        person_id = int(batch["personid"][n])
                        input_patch = batch["img"][n].cpu() * (
                            DEFAULT_STD[:, None, None] / 255
                        ) + (DEFAULT_MEAN[:, None, None] / 255)
                        input_patch = input_patch.permute(1, 2, 0).numpy()

                        # Add all verts and cams to list
                        verts = out["pred_vertices"][n].detach().cpu().numpy()
                        is_right = batch["right"][n].cpu().numpy()
                        verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                        cam_t = pred_cam_t_full[n]
                        all_verts.append(verts)
                        all_cam_t.append(cam_t)
                        all_right.append(is_right)

                # Render the result
                renderer = MeshPyTorch3DRenderer(
                    model_cfg,
                    model.mano.faces,
                    device,
                    render_res=img_size[0],
                    focal_length=scaled_focal_length,
                )

                # Render front view
                if len(all_verts) > 0:
                    render_start = time.time()
                    cam_view = renderer.fast_render_rgb_frame_pytorch3d(
                        all_verts, cam_t=all_cam_t, is_right=all_right
                    )

                    # Overlay image
                    input_img = frame.astype(np.float32)[:, :, ::-1] / 255.0
                    input_img = np.concatenate(
                        [input_img, np.ones_like(input_img[:, :, :1])], axis=2
                    )  # Add alpha channel
                    input_img_overlay = (
                        input_img[:, :, :3] * (1 - cam_view[:, :, 3:])
                        + cam_view[:, :, :3] * cam_view[:, :, 3:]
                    )
                    output_img = 255 * input_img_overlay[:, :, ::-1]
                    render_stop = time.time()

                    cv2.putText(
                        output_img,
                        f"Render FPS: {(1/(render_stop-render_start)):.0f}",
                        (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        output_img,
                        f"HaMeR FPS: {fps:.0f}",
                        (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    output_img = output_img.astype(np.uint8)
                else:
                    output_img = frame
            else:
                output_img = frame

            # img_show = draw_bbox(img_cv2, bboxes)
            cv2.putText(
                output_img,
                f"Tracker FPS: {(1/(tracker_end-tracker_start)):.0f}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                output_img,
                f"FPS: {(1/(time.time()-start_loop)):.0f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Camera", output_img)

            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
