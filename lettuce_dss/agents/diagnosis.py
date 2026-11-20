"""Single-checkpoint EfficientNet inference and confidence checks."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import threading
from typing import Callable

import numpy as np

from lettuce_dss.core import BaseAgent, CLASS_NAMES, decision

BROAD_LABELS = {"Bacterial", "Viral", "Wilt_and_Leaf_Blight"}


class ModelConfigurationError(ValueError):
    pass


def validated_probabilities(values):
    """Validate and normalize class probabilities."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if (values.ndim != 2 or values.shape[1] != len(CLASS_NAMES) or not values.shape[0]
            or not np.isfinite(values).all() or (values < 0).any()
            or (values > 1 + 1e-6).any()
            or not np.allclose(values.sum(axis=1), 1, atol=1e-3, rtol=0)):
        raise ModelConfigurationError("The model returned invalid class probabilities.")
    return values / values.sum(axis=1, keepdims=True)


def apply_temperature(probabilities, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ModelConfigurationError("Calibration temperature must be positive.")
    values = validated_probabilities(probabilities)
    logits = np.log(np.clip(values, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def normalize_names(names):
    if isinstance(names, dict):
        try:
            return [names[i] if i in names else names[str(i)] for i in range(len(names))]
        except (KeyError, TypeError):
            raise ModelConfigurationError("The model class mapping is incomplete.") from None
    return list(names)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


class DiagnosisAgent(BaseAgent):
    name = "diagnosis"
    goal = "Classify the leaf and choose how specific the recommendation may be."

    def __init__(self, project_root: Path, mode="single", device="cpu", predictor: Callable | None = None):
        super().__init__()
        if mode != "single":
            raise ValueError("Only single-model mode is supported.")
        if device not in {"cpu", "auto"}:
            raise ValueError("device must be cpu or auto")
        self.project_root = Path(project_root).resolve()
        self.mode, self.device, self.predictor = mode, device, predictor
        self._models = None
        self._lock = threading.RLock()
        self.manifest = None
        self.calibration = None

    def _project_file(self, relative_path):
        path = (self.project_root / relative_path).resolve()
        if not path.is_relative_to(self.project_root):
            raise ModelConfigurationError("A model configuration path is outside this application.")
        return path

    def _configuration(self):
        if self.manifest is not None:
            return
        manifest = json.loads((self.project_root / "config/model.json").read_text())
        if manifest["class_names"] != CLASS_NAMES or manifest["imgsz"] != 224:
            raise ModelConfigurationError("Class order or image size does not match the selected model.")
        if manifest.get("backend") != "efficientnet_b0":
            raise ModelConfigurationError("Only the selected EfficientNet classifier is supported.")
        from lettuce_dss.efficientnet_backend import PREPROCESSING, VARIANT, CHECKPOINT_FORMAT, SELECTED_SEED
        if (manifest.get("variant") != VARIANT or manifest.get("preprocessing") != PREPROCESSING
                or manifest.get("checkpoint_format") != CHECKPOINT_FORMAT
                or manifest.get("single_seed") != SELECTED_SEED
                or manifest.get("available_modes") != ["single"]):
            raise ModelConfigurationError("EfficientNet preprocessing or checkpoint contract differs from the selected export.")
        calibration = json.loads(self._project_file(manifest["calibration_path"]).read_text())
        if (calibration.get("model_variant") != VARIANT or not calibration.get("status")):
            raise ModelConfigurationError("EfficientNet requires a matching calibration identity and status.")
        entry = calibration["modes"][f"single_seed_{SELECTED_SEED}"]
        if "checkpoint_sha256" not in entry or entry.get("class_names") != CLASS_NAMES:
            raise ModelConfigurationError("Probability settings must bind the checkpoint hash and class order.")
        weights = manifest.get("weights", [])
        if len(weights) != 1 or weights[0].get("seed") != SELECTED_SEED:
            raise ModelConfigurationError("The deployment manifest must contain the selected seed-42 checkpoint.")
        expected_hashes = {str(SELECTED_SEED): weights[0]["sha256"]}
        if entry["checkpoint_sha256"] != expected_hashes:
            raise ModelConfigurationError("Calibration does not match the configured checkpoint.")
        temperature = float(entry["temperature"])
        if not np.isfinite(temperature) or temperature <= 0:
            raise ModelConfigurationError("Calibration temperature is invalid.")
        calibration_status = calibration["status"]
        if calibration_status not in {
                "not_fitted", "development_fitted_not_independently_validated"}:
            raise ModelConfigurationError("The EfficientNet probability calibration status is not recognized.")
        if calibration_status == "not_fitted" and temperature != 1.0:
            raise ModelConfigurationError("Unfitted probabilities must use identity temperature 1.0.")
        policy = manifest.get("decision_policy", {})
        for name, default in (("confidence_threshold", .8), ("margin_threshold", .15)):
            value = float(policy.get(name, default))
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ModelConfigurationError("Diagnosis decision thresholds must be between zero and one.")
        self.manifest, self.calibration = manifest, entry
        self.backend, self.calibration_status = "efficientnet_b0", calibration_status

    def _weight_entries(self):
        entries = self.manifest["weights"]
        # Check the file before loading it.
        for entry in entries:
            path = self._project_file(entry["path"])
            if not path.is_file():
                raise ModelConfigurationError(
                    f"EfficientNet seed {entry['seed']} weights are not installed. "
                    "Restore the included best_weights.pt model file.")
            if sha256_file(path) != entry.get("sha256"):
                raise ModelConfigurationError("A model checkpoint failed its integrity check. Use the selected study checkpoint.")
        return entries

    def _load_models(self):
        if self._models is not None:
            return
        entries = self._weight_entries()
        import torch
        from lettuce_dss.efficientnet_backend import load_classifier
        self._runtime_device = "cuda:0" if self.device == "auto" and torch.cuda.is_available() else "cpu"
        self._models = [load_classifier(self._project_file(entry["path"]), self.manifest,
                                       entry, self._runtime_device) for entry in entries]

    def _predict(self, image_path):
        if self.predictor is not None:
            output = self.predictor(Path(image_path), self.mode)
            if normalize_names(output["class_names"]) != CLASS_NAMES:
                raise ModelConfigurationError("Prediction class order does not match the application.")
            return validated_probabilities(output["probabilities"])
        self._load_models()
        from lettuce_dss.efficientnet_backend import predict_rows
        return validated_probabilities(predict_rows(self._models, image_path, self._runtime_device))

    def run(self, image_path: Path):
        with self._lock:
            self.begin()
            decisions = []
            try:
                self._configuration()
                if not Path(image_path).is_file():
                    raise FileNotFoundError
                rows = self._predict(image_path)
                if len(rows) != 1:
                    raise ModelConfigurationError("The prediction count does not match the selected model mode.")
                raw = validated_probabilities(rows[0])[0]
                temperature = float(self.calibration["temperature"])
                calibrated = (raw.copy() if self.calibration_status == "not_fitted"
                              else apply_temperature(raw, temperature)[0])
                order = np.argsort(-calibrated, kind="stable")
                label = CLASS_NAMES[int(order[0])]
                confidence, margin = float(calibrated[order[0]]), float(calibrated[order[0]]-calibrated[order[1]])
                policy = self.manifest.get("decision_policy", {})
                confidence_threshold = float(policy.get("confidence_threshold", .8))
                margin_threshold = float(policy.get("margin_threshold", .15))
                uncertain = confidence < confidence_threshold or margin < margin_threshold
                broad = label in BROAD_LABELS
                scope = "general_only" if uncertain or broad else "disease_specific"
                decisions.append(decision("single_prediction",
                                          "Use the frozen deployment mode and its declared probability treatment.",
                                          mode=self.mode, models=1, temperature=temperature))
                if self.calibration_status == "not_fitted":
                    decisions.append(decision("report_uncalibrated_score",
                        "These are raw model probabilities; no EfficientNet temperature has been fitted."))
                decisions.append(decision("request_clearer_image" if uncertain else "report_leading_class",
                                          "The confidence or class separation is below the configured policy."
                                          if uncertain else "The leading class passes the configured confidence and separation policy.",
                                          confidence=confidence, margin=margin, threshold_status="provisional"))
                if broad:
                    decisions.append(decision("limit_specificity", "This inherited dataset label covers more than one possible condition."))
                top3 = [{"class_name": CLASS_NAMES[int(i)], "probability": float(calibrated[i]),
                         "raw_probability": float(raw[i])} for i in order[:3]]
                messages = ["The image is uncertain. Try a clearer photograph of one affected leaf."] if uncertain else []
                data = {"prediction": label, "confidence": confidence, "raw_confidence": float(raw[order[0]]),
                        "margin": margin, "top3": top3, "class_names": CLASS_NAMES,
                        "raw_probabilities": raw.tolist(), "probabilities": calibrated.tolist(),
                        "calibrated_probabilities": (None if self.calibration_status == "not_fitted" else calibrated.tolist()),
                        "uncertainty": bool(uncertain), "source_label_broad": broad,
                        "request_retake": bool(uncertain), "recommendation_scope": scope,
                        "model_mode": self.mode, "model_id": self.manifest["model_id"],
                        "backend": self.backend, "display_name": self.manifest.get("display_name", self.manifest["model_id"]),
                        "probability_kind": "raw" if self.calibration_status == "not_fitted" else "temperature_scaled",
                        "temperature": temperature, "calibration_status": self.calibration_status,
                        "threshold_status": "provisional", "severity_assessed": False,
                        "unknown_disease_detection": False}
                return self.result(data, "partial" if uncertain else "ok", decisions, messages)
            except ImportError:
                message = "The diagnosis dependencies are not ready. Install the requirements listed in README."
            except FileNotFoundError:
                message = "A required image, model or configuration file is missing. Restore the application package and try again."
            except ModelConfigurationError as exc:
                message = str(exc)
            except Exception:
                message = "The diagnosis model could not complete this image. Check the installation and try again."
            decisions.append(decision("withhold_diagnosis", message))
            return self.result({"prediction": None, "uncertainty": True, "request_retake": False,
                                "recommendation_scope": "general_only", "model_mode": self.mode},
                               "error", decisions, [message])
