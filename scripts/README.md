# Scripts

- **pose_tracker.py**, **pose_tracker_demo.py** – Pose tracker utilities and demo (used by main demos).
- **demo_pytorch3d.py**, **demo_pytorch3d_2.py** – Optional demos: whole-body pose + HaMeR with PyTorch3D mesh rendering. Run from repo root: `python scripts/demo_pytorch3d.py [--checkpoint PATH] [--efficient_hamer]`.
- **faster_demo.py** – RTMPose whole-body tracker + bbox drawing (no HaMeR).
- **hamer_tensorrt.py** – Example: compile HaMeR backbone with TensorRT.
- **wholebody_rtmpose_demo.py** – RTMPose hand-only tracker (no HaMeR).
