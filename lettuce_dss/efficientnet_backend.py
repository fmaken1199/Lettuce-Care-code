"""Load and run the selected EfficientNet-B0 checkpoint."""
from __future__ import annotations

from lettuce_dss.core import CLASS_NAMES

VARIANT = "noerase_crt_balanced"
CHECKPOINT_FORMAT = "lettuce_b0_master_weights_v1"
SELECTED_SEED = 42
RUN_FINGERPRINT = "1ea62839a18989435b9bfd416f53c4824a2329524c0aa967cec47c35d9e0c308"
DATASET_FINGERPRINT = "eff46ba60a6063a258dd431efde513fc4977e028267a4e69a80dd6c1bc7d2e75"
PREPROCESSING = {
    "decode": "PIL_RGB_no_exif_transpose",
    "resize_short_side": 224,
    "center_crop": 224,
    "interpolation": "bilinear",
    "antialias": True,
    "tensor_range": [0, 1],
    "normalization_location": "inside_model",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def validate_checkpoint_metadata(checkpoint, manifest, entry):
    """Check the checkpoint identity and class order."""
    from lettuce_dss.agents.diagnosis import ModelConfigurationError
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ModelConfigurationError("Expected the final B0 master's best_weights.pt state-dictionary export.")
    meta = checkpoint.get("metadata", {})
    config = checkpoint.get("config", {})
    if not all(isinstance(value, dict) for value in (meta, config, manifest, entry)):
        raise ModelConfigurationError("The final classifier metadata and deployment configuration must be dictionaries.")
    exp = meta.get("experiment", {})
    expected = {"key": VARIANT, "backbone": "efficientnet_b0", "classification_loss": "ce",
                "supcon": False, "memory": False, "weighted_anchors": False}
    if (checkpoint.get("class_names") != CLASS_NAMES or exp != expected
            or meta.get("experiment_key") != VARIANT
            or meta.get("kind") != "head" or meta.get("family") != 101
            or meta.get("seed") != SELECTED_SEED or entry.get("seed") != SELECTED_SEED
            or meta.get("fingerprint") != RUN_FINGERPRINT
            or entry.get("run_fingerprint") != RUN_FINGERPRINT
            or meta.get("dataset_fingerprint") != DATASET_FINGERPRINT
            or manifest.get("dataset_fingerprint") != DATASET_FINGERPRINT
            or meta.get("prior_correction_tau") != 0.0
            or config.get("image_size") != 224
            or config.get("balanced_sampling") is not True
            or config.get("erase_probability") != 0.0
            or not isinstance(checkpoint.get("model"), dict)):
        raise ModelConfigurationError("The checkpoint does not match the selected balanced-cRT experiment, seed, preprocessing or class order.")


def make_classifier():
    """Build the exported classifier without downloading weights."""
    import torch
    from torch import nn
    from torchvision import models

    class StudyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = models.efficientnet_b0(weights=None)
            self.encoder.classifier[-1] = nn.Linear(1280, len(CLASS_NAMES))
            self.register_buffer("input_mean", torch.tensor(PREPROCESSING["mean"]).view(1, 3, 1, 1))
            self.register_buffer("input_std", torch.tensor(PREPROCESSING["std"]).view(1, 3, 1, 1))

        def forward(self, image):
            return self.encoder((image - self.input_mean) / self.input_std)

    return StudyClassifier()


def load_classifier(path, manifest, entry, device):
    import torch
    from lettuce_dss.agents.diagnosis import ModelConfigurationError
    # SHA256 is checked before this call.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint_metadata(checkpoint, manifest, entry)
    model = make_classifier()
    state = checkpoint["model"]
    for name, expected in (("input_mean", PREPROCESSING["mean"]), ("input_std", PREPROCESSING["std"])):
        value = state.get(name)
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (1, 3, 1, 1)
                or not torch.allclose(value.float(), torch.tensor(expected).view(1, 3, 1, 1), atol=1e-7, rtol=0)):
            raise ModelConfigurationError("The EfficientNet normalization buffers do not match training.")
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=torch.float32).eval()


def image_tensor(image_path):
    from PIL import Image
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
    transform = T.Compose([
        T.Resize(224, interpolation=InterpolationMode.BILINEAR, antialias=True),
        T.CenterCrop(224), T.ToTensor(),
    ])
    # Preserve the configured decoder.
    with Image.open(image_path) as image:
        return transform(image.convert("RGB")).unsqueeze(0)


def predict_rows(models, image_path, device):
    import torch
    image = image_tensor(image_path).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        return [torch.softmax(model(image), dim=1)[0].detach().cpu().numpy() for model in models]
