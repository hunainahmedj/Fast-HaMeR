import os
import sys
import time
import argparse
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Master's thesis demo code.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="_DATA/hamer_ckpts/checkpoints/hamer.ckpt",
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--img_folder",
        type=str,
        default="example_data",
        help="Folder with input images",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="out_demo",
        help="Output folder to save rendered results",
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
        "--file_type",
        nargs="+",
        default=["*.jpg", "*.png"],
        help="List of file extensions to consider",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference/fitting"
    )

    args = parser.parse_args()

    model, model_cfg = load_hamer(args.checkpoint, init_renderer=False)

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = model.to(device)
    model.eval()

    # load pose tracker
    pose_tracker = create_pose_tracker()

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    img_paths = [
        img for end in args.file_type for img in Path(args.img_folder).glob(end)
    ]

    for index, img_path in enumerate(img_paths):

        frame = cv2.imread(str(img_path))
        keypoints, scores = pose_tracker(frame)

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
                    out = model(batch)

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

            if len(all_verts) > 0:
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
                output_img = output_img.astype(np.uint8)
            else:
                output_img = frame

            cv2.imwrite(os.path.join(args.out_folder, f"{index}_all.jpg"), output_img)
        else:
            cv2.imwrite(os.path.join(args.out_folder, f"{index}_all.jpg"), frame)


if __name__ == "__main__":
    main()
