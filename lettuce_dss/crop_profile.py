"""Crop form defaults and transparent presentation of their provenance."""
from __future__ import annotations

CROP_DEFAULTS = {
    "environment": "outdoor", "root_system": "soil",
    "growth_stage": "vegetative", "irrigation": "manual",
}
CROP_LABELS = {
    "environment": "Where it is grown", "root_system": "Root environment",
    "growth_stage": "Growth stage", "irrigation": "Watering method",
}
VALUE_LABELS = {
    "unknown": "Not supplied", "vegetative": "Vegetative / leaf growth",
    "manual": "Manual watering", "near_harvest": "Near harvest",
    "hydroponic": "Hydroponic solution", "substrate": "Growing substrate",
}


def crop_value_label(value):
    return VALUE_LABELS.get(value, str(value or "Not supplied").replace("_", " ").title())


def crop_profile_rows(farm):
    """Never infer user confirmation from a value that matches a default."""
    farm = farm if isinstance(farm, dict) else {}
    provenance = farm.get("input_provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    origin_labels = {"ui_default": "Default assumption", "user_selected": "User selected"}
    return [{"Field": label, "Setting": crop_value_label(farm.get(field, "unknown")),
             "Origin": origin_labels.get(provenance.get(field), "Origin not recorded")}
            for field, label in CROP_LABELS.items()]
