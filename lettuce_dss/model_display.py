"""Lightweight deployment information for the UI; no inference imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def model_readiness(project_root: Path, mode: str) -> dict:
    'Check configured files, leaving weight integrity/runtime checks to inference.'
    root = Path(project_root).resolve()
    revision = hashlib.sha256()
    info = {"ready": False, "display_name": "Diagnosis model", "missing_seeds": [],
            "calibration_status": "unavailable", "messages": []}

    def project_path(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("A configured model path is outside the project.")
        return path

    def read_json(relative):
        raw = project_path(relative).read_bytes()
        revision.update(str(relative).encode())
        revision.update(raw)
        return json.loads(raw)

    try:
        manifest = read_json("config/model.json")
        read_json("config/app.json")
        info["display_name"] = str(manifest.get("display_name") or manifest.get("backend") or manifest.get("model_id") or "Diagnosis model")
        info["model_id"] = manifest.get("model_id")
        info["deployment_status"] = manifest.get("deployment_status", "unknown")
        info["available_modes"] = manifest.get("available_modes", ["single"])
        if mode != "single" or info["available_modes"] != ["single"]:
            raise ValueError("The selected classifier supports single-model mode only.")
        single_seed = int(manifest["single_seed"])
        expected = [single_seed]
        entries = manifest.get("weights", [])
        if sorted(int(item["seed"]) for item in entries) != sorted(expected):
            raise ValueError("The selected checkpoint set is incomplete in the model configuration.")
        for item in entries:
            path = project_path(item["path"])
            if path.is_file():
                stat = path.stat()
                revision.update(f"{item['path']}:{stat.st_size}:{stat.st_mtime_ns}".encode())
            else:
                info["missing_seeds"].append(int(item["seed"]))
            if not re.fullmatch(r"[0-9a-fA-F]{64}", str(item.get("sha256", ""))):
                raise ValueError("A selected checkpoint has no registered integrity hash.")
        calibration = read_json(manifest["calibration_path"])
        key = f"single_seed_{single_seed}"
        entry = calibration["modes"][key]
        info["calibration_status"] = entry.get("status", calibration.get("status", "unavailable"))
        info["ready"] = not info["missing_seeds"]
        if info["missing_seeds"]:
            seeds = ", ".join(str(seed) for seed in sorted(info["missing_seeds"]))
            info["messages"].append(f"{info['display_name']} checkpoint files are missing for seed(s) {seeds}. Image diagnosis is unavailable until they are installed.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        info["messages"].append("The diagnosis model configuration needs attention: " + str(exc))
    info["revision"] = revision.hexdigest()
    return info


def model_confidence_label(diagnosis: dict) -> str:
    return "Model score (uncalibrated)" if diagnosis.get("calibration_status") == "not_fitted" else "Model confidence"
