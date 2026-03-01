# Optional: Compile HaMeR backbone with TensorRT. Run from repo root.
import torch
import torch_tensorrt

from hamer.models import load_hamer, DEFAULT_CHECKPOINT

model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
model.eval()

backbone_input = [
    torch_tensorrt.Input(
        min_shape=(1, 3, 256, 192),
        opt_shape=(2, 3, 256, 192),
        max_shape=(8, 3, 256, 192),
        dtype=torch.float,
    )
]

tensorrt_backbone = torch_tensorrt.compile(model.backbone, inputs=backbone_input)

model.backbone = tensorrt_backbone
