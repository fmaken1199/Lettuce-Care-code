"""Conservative image acceptance, with no disease-colour alterations."""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError

from lettuce_dss.core import BaseAgent, decision


class ImageQualityAgent(BaseAgent):
    name = "image_quality"
    goal = "Accept a decodable leaf photograph or request an image that can be processed."

    def __init__(self, max_bytes=15*1024*1024, min_side=64, max_pixels=30_000_000):
        super().__init__()
        self.max_bytes, self.min_side, self.max_pixels = max_bytes, min_side, max_pixels

    def _reject(self, message):
        return self.result({"usable": False, "image_path": None, "request_retake": True},
                           "blocked", [decision("request_retake", message)], [message])

    def run(self, image_bytes: bytes, output_path: Path):
        self.begin()
        if not isinstance(image_bytes, bytes) or not image_bytes:
            return self._reject("Upload a JPG, PNG or WebP photograph of one lettuce leaf.")
        if len(image_bytes) > self.max_bytes:
            return self._reject("The image is too large. Upload a file smaller than 15 MB.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(image_bytes)) as image:
                    width, height = image.size
                    if image.format not in {"JPEG", "PNG", "WEBP", "BMP"}:
                        return self._reject("Use a JPG, PNG, WebP or BMP photograph.")
                    if getattr(image, "n_frames", 1) != 1:
                        return self._reject("Upload a single photograph rather than an animated image.")
                    if width*height > self.max_pixels:
                        return self._reject("The image dimensions are too large. Resize it below 30 megapixels.")
                    if min(width, height) < self.min_side:
                        return self._reject("The image is too small. Upload a clearer photograph at least 64 pixels on each side.")
                    image.load()
                    rgb = image.convert("RGB")
                    rgb.thumbnail((512, 512))
                    gray = np.asarray(rgb.convert("L"), dtype=np.float64)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
                Image.DecompressionBombWarning):
            return self._reject("The image could not be read. Try another JPG, PNG or WebP photograph.")
        laplacian = -4*gray[1:-1, 1:-1]+gray[:-2, 1:-1]+gray[2:, 1:-1]+gray[1:-1, :-2]+gray[1:-1, 2:]
        blur_score = float(laplacian.var())
        dark_fraction, light_fraction = float((gray < 20).mean()), float((gray > 235).mean())
        issues = []
        if blur_score < 35:
            issues.append("The image may lack detail. A sharper, close photograph may help.")
        if dark_fraction > .7:
            issues.append("Most of the image is dark. Consider a photograph in even daylight.")
        if light_fraction > .7:
            issues.append("Most of the image is very bright. Avoid glare on the leaf.")
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Keep original bytes for inference.
            output_path.write_bytes(image_bytes)
        except OSError:
            return self.result({"usable": False, "image_path": None, "request_retake": False}, "error",
                               [decision("withhold_image", "The application could not store the uploaded image.")],
                               ["The image could not be stored. Check the application's output folder."])
        decisions = [decision("accept_image", "The photograph can be decoded and has sufficient dimensions.",
                              pixels_unchanged=True)]
        if issues:
            decisions.append(decision("warn_image_quality", "Simple image-quality checks suggest possible limitations.",
                                      checks=issues, thresholds="heuristic"))
        return self.result({"usable": True, "image_path": str(output_path), "width": width, "height": height,
                            "request_retake": False, "warnings": issues,
                            "quality": {"laplacian_variance": round(blur_score, 3),
                                        "dark_fraction": round(dark_fraction, 4),
                                        "bright_fraction": round(light_fraction, 4),
                                        "assessment": "heuristic", "leaf_presence_verified": False},
                            "pixels_unchanged": True},
                           "partial" if issues else "ok", decisions, issues)
