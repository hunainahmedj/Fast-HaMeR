from rtmlib import PoseTracker, Wholebody, draw_skeleton, draw_bbox


def create_pose_tracker(
    device: str = "cuda",
    det_frequency: int = 10,
    mode: str = "lightweight",
    backend: str = "onnxruntime",
):
    """Create a pose tracker for tracking body pose and return keypoints

    Returns:
        PoseTracker: Return RTMLib pose tracker.
    """

    return PoseTracker(
        Wholebody,
        mode=mode,
        device=device,
        backend=backend,
        to_openpose=False,
        det_frequency=det_frequency,
    )
