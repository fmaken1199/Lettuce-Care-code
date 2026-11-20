"""Environmental context agent: select, validate and describe evidence sources."""
from __future__ import annotations

from datetime import date, datetime, time, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import math
from pathlib import Path

from lettuce_dss.core import BaseAgent, decision
from lettuce_dss.providers import WeatherProvider, empty_weather, parse_utc

PAKISTAN_BOUNDS = {"latitude": [23.5, 37.5], "longitude": [60.5, 77.5]}
PAKISTAN_TZ = timezone(timedelta(hours=5))
SETUP_CHOICES = {
    "environment": {"unknown", "outdoor", "greenhouse", "tunnel", "indoor"},
    "root_system": {"unknown", "soil", "substrate", "hydroponic"},
    "growth_stage": {"unknown", "seedling", "vegetative", "near_harvest"},
    "irrigation": {"unknown", "drip", "overhead", "manual", "recirculating"},
}
SOIL_REASON_CODES = {"urban_cell", "no_coverage", "missing_location", "invalid_location",
                     "not_applicable", "source_unavailable", "disabled"}
MEASUREMENT_RULES = {
    # Bounds reject invalid numeric data, not agronomic suitability thresholds.
    "ph": {"unit": "pH", "range": [0, 14], "fresh_hours": 180 * 24},
    "ec": {"unit": "dS/m", "range": [0, 100], "fresh_hours": 7 * 24},
    "moisture": {"unit": "% VWC", "range": [0, 100], "fresh_hours": 24},
    "root_temperature": {"unit": "C", "range": [-50, 80], "fresh_hours": 24},
    "air_temperature": {"unit": "C", "range": [-70, 65], "fresh_hours": 6},
    "relative_humidity": {"unit": "%", "range": [0, 100], "fresh_hours": 6},
}


def _numeric(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Enter a finite numeric value")
    return float(value)


def _normalise_units(key, value, unit):
    unit = str(unit or "").strip()
    if key == "ph" and unit in {"pH", "ph"}:
        return value
    if key == "ec":
        if unit in {"dS/m", "mS/cm"}:
            return value
        if unit in {"uS/cm", "µS/cm", "μS/cm"}:
            return value / 1000
    if key in {"air_temperature", "root_temperature"}:
        if unit in {"C", "°C", "celsius"}:
            return value
        if unit in {"F", "°F", "fahrenheit"}:
            return (value - 32) * 5 / 9
    if key == "moisture":
        if unit in {"% VWC", "% vwc"}:
            return value
        if unit in {"m3/m3", "m³/m³"}:
            return value * 100
    if key == "relative_humidity" and unit == "%":
        return value
    raise ValueError("The measurement unit is missing or unsupported")


def _measurement_date(raw, now):
    if raw is None or raw == "":
        return {"measured_at": None, "age_hours": None, "date_precision": "unknown"}
    # Naive farmer times are local Pakistan time, unlike the provider's GMT times.
    if isinstance(raw, datetime):
        stamp, precision = raw, "datetime"
    elif isinstance(raw, date):
        stamp, precision = datetime.combine(raw, time(0)), "date"
    elif isinstance(raw, str):
        try:
            if len(raw.strip()) == 10:
                stamp, precision = datetime.combine(date.fromisoformat(raw), time(0)), "date"
            else:
                stamp, precision = datetime.fromisoformat(raw.strip().replace("Z", "+00:00")), "datetime"
        except ValueError as exc:
            raise ValueError("The measurement date is invalid") from exc
    else:
        raise ValueError("The measurement date is invalid")
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=PAKISTAN_TZ)
    stamp = stamp.astimezone(timezone.utc)
    if stamp > now + timedelta(minutes=5):
        raise ValueError("The measurement date is in the future")
    return {"measured_at": stamp.isoformat(), "age_hours": round(max(0, (now - stamp).total_seconds() / 3600), 3),
            "date_precision": precision}


def _missing(key, reason="No measurement was supplied", *, applicable=True):
    return {"value": None, "unit": MEASUREMENT_RULES[key]["unit"], "source": None,
            "measured_at": None, "kind": "missing", "freshness": "unknown", "age_hours": None,
            "applicable": applicable, "reason": reason}


class EnvironmentalAgent(BaseAgent):
    name = "environment"
    goal = "Choose suitable environmental evidence for this farm and preserve its source and limitations."

    def __init__(self, cache_dir: Path, timeout: float = 5.0, fetch_json=None, *, soil_provider=None, project_root=None):
        super().__init__()
        self.provider = WeatherProvider(cache_dir, timeout=timeout, fetcher=fetch_json)
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        self.soil_provider = soil_provider

    def _soil_context(self, location, coordinate_error, root_system, farm, cache_only):
        if coordinate_error:
            return {"status": coordinate_error, "reason_code": "missing_location" if coordinate_error == "no_location" else "invalid_location",
                    "measurements": {}, "notes": ["A valid farm location is needed for regional soil estimates."]}
        if farm.get("use_soil_map", True) is False:
            return {"status": "disabled", "reason_code": "disabled", "measurements": {}, "notes": ["Soil-map lookup was switched off."]}
        if root_system != "soil":
            return {"status": "not_applicable", "reason_code": "not_applicable", "measurements": {}, "notes": ["Geographic soil maps do not describe a growing substrate or nutrient solution."]}
        try:
            if self.soil_provider is None:
                from lettuce_dss.soil_maps import SoilMapProvider
                self.soil_provider = SoilMapProvider(self.project_root)
            return self.soil_provider.get(location["latitude"], location["longitude"],
                    properties=("ph", "sand", "silt", "clay", "ec"),
                    cache_only=cache_only or bool(farm.get("offline")))
        except Exception as exc:
            return {"status": "unavailable", "reason_code": "source_unavailable", "measurements": {},
                    "notes": [f"Regional soil information could not be read ({type(exc).__name__}); farmer inputs remain available."]}

    def run(self, farm: dict, request: dict | None = None):
        self.begin()
        now = datetime.now(timezone.utc)
        decisions, notes = [], []
        farm = farm if isinstance(farm, dict) else {}
        environment = farm.get("environment")
        root_system = farm.get("root_system")
        if not isinstance(environment, str) or environment not in {"outdoor", "greenhouse", "tunnel", "indoor"}:
            environment = "unknown"
            decisions.append(decision("withhold_outdoor_substitution", "The growing environment was missing or not recognised."))
        if not isinstance(root_system, str) or root_system not in {"soil", "substrate", "hydroponic"}:
            root_system = "unknown"
            decisions.append(decision("withhold_root_estimates", "The growing medium was missing or not recognised."))
        # Keep default origins; do not fill missing inputs.
        setup = {"environment": environment, "root_system": root_system}
        for key in ("growth_stage", "irrigation"):
            value = farm.get(key)
            setup[key] = value if isinstance(value, str) and value in SETUP_CHOICES[key] else "unknown"
        supplied_provenance = farm.get("input_provenance")
        supplied_provenance = supplied_provenance if isinstance(supplied_provenance, dict) else {}
        input_provenance = {
            key: origin for key, origin in supplied_provenance.items()
            if key in SETUP_CHOICES and isinstance(origin, str) and origin in {"ui_default", "user_selected"}
            and isinstance(farm.get(key), str) and farm[key] in SETUP_CHOICES[key]
        }
        location = dict(farm.get("location") or {}) if isinstance(farm.get("location"), dict) else {}
        location = {key: location.get(key) for key in ("latitude", "longitude", "label", "source", "accuracy_m")}
        accuracy = location["accuracy_m"]
        location["accuracy_m"] = None
        location["accuracy_status"] = "unknown"
        location["accuracy_note"] = "Horizontal location accuracy was not supplied; coordinate decimal places do not establish accuracy."
        if accuracy is not None:
            try:
                accuracy = _numeric(accuracy)
                if accuracy < 0:
                    raise ValueError("Horizontal accuracy cannot be negative")
                location["accuracy_m"] = accuracy
                location["accuracy_status"] = "coarse" if accuracy > 1000 else "reported"
                location["accuracy_note"] = f"Reported horizontal accuracy: {accuracy:g} m; this is not confirmation of the cultivated plot."
                if accuracy > 1000:
                    location["accuracy_note"] += " The reported radius exceeds 1 km; check that the coordinates identify the actual growing plot."
                    notes.append(location["accuracy_note"])
                    decisions.append(decision("flag_coarse_location", location["accuracy_note"], accuracy_m=accuracy,
                                              threshold_type="operational_location_attention_only"))
            except ValueError:
                location["accuracy_note"] = "The supplied horizontal accuracy was invalid and was not used; coordinate precision was not inferred."
                notes.append(location["accuracy_note"])
        coordinate_error = None
        if location["latitude"] is None or location["longitude"] is None:
            coordinate_error = "no_location"
            for field in ("latitude", "longitude"):
                if location[field] is not None:
                    try:
                        location[field] = _numeric(location[field])
                    except ValueError:
                        location[field] = None
                        coordinate_error = "invalid_location"
        else:
            try:
                lat, lon = _numeric(location["latitude"]), _numeric(location["longitude"])
                if not PAKISTAN_BOUNDS["latitude"][0] <= lat <= PAKISTAN_BOUNDS["latitude"][1] or not \
                        PAKISTAN_BOUNDS["longitude"][0] <= lon <= PAKISTAN_BOUNDS["longitude"][1]:
                    raise ValueError("Coordinates lie outside the approximate Pakistan operating region")
                location.update(latitude=lat, longitude=lon)
            except ValueError:
                coordinate_error = "invalid_location"
                # Invalid user numbers must not leak NaN/Infinity into JSON reports.
                for field in ("latitude", "longitude"):
                    try:
                        location[field] = _numeric(location[field])
                    except ValueError:
                        location[field] = None
        location["status"] = coordinate_error or "provided"
        location["scope_note"] = "Approximate Pakistan operating bounds; this is not a legal-border determination."
        measurements = {key: _missing(key) for key in MEASUREMENT_RULES}
        input_errors = {}
        provided = farm.get("measurements") or {}
        provided = provided if isinstance(provided, dict) else {}
        sample_type = {"soil": "soil", "substrate": "substrate", "hydroponic": "nutrient_solution"}.get(root_system)
        for key, rule in MEASUREMENT_RULES.items():
            if key == "moisture" and root_system == "hydroponic":
                measurements[key] = _missing(key, "Soil volumetric moisture does not describe a hydroponic nutrient solution.", applicable=False)
                if key in provided and provided[key] is not None:
                    decisions.append(decision("exclude_incompatible_measurement", measurements[key]["reason"], measurement=key))
                continue
            item = provided.get(key)
            if item is None or isinstance(item, dict) and item.get("value") is None:
                continue
            try:
                if not isinstance(item, dict):
                    raise ValueError("A measurement needs a value, unit and optional date")
                value = _normalise_units(key, _numeric(item.get("value")), item.get("unit"))
                if not rule["range"][0] <= value <= rule["range"][1]:
                    raise ValueError("The value is outside the accepted input range")
                if key not in {"air_temperature", "relative_humidity"} and item.get("sample_type") and item["sample_type"] != sample_type:
                    raise ValueError("The measurement sample does not match the selected growing medium")
                date_info = _measurement_date(item.get("measured_at"), now)
                freshness = "unknown" if date_info["age_hours"] is None else (
                    "fresh" if date_info["age_hours"] <= rule["fresh_hours"] else "stale")
                measurements[key] = {"value": round(value, 6), "unit": rule["unit"], "source": "farmer",
                                     "kind": "measured", "freshness": freshness, "applicable": True,
                                     "original_value": item["value"], "original_unit": item.get("unit"),
                                     "sample_type": "growing_area_air" if key in {"air_temperature", "relative_humidity"} else sample_type,
                                     "method": str(item.get("method") or "not specified"), **date_info}
                decisions.append(decision("prefer_farmer_measurement", "A valid farm measurement takes priority over a model estimate.",
                                          measurement=key, freshness=freshness))
                if freshness != "fresh":
                    notes.append(f"{key.replace('_', ' ').capitalize()}: the farmer measurement is {'older than the operational freshness window' if freshness == 'stale' else 'undated'}.")
            except (ValueError, TypeError) as exc:
                measurements[key] = _missing(key, str(exc))
                input_errors[key] = str(exc)
                decisions.append(decision("reject_invalid_measurement", str(exc), measurement=key))

        def explain_missing(key, reason, reason_code):
            item = measurements[key]
            if item["kind"] == "missing" and item["applicable"]:
                item["reason"] = (f"Farmer input was rejected: {input_errors[key]}. " if key in input_errors else "") + reason
                item["reason_code"] = reason_code

        cache_only = bool(request and request.get("action") == "summarize_missing")
        # Fetch weather and soil independently.
        with ThreadPoolExecutor(max_workers=2) as pool:
            soil_future = pool.submit(self._soil_context, location, coordinate_error, root_system, farm, cache_only)
            if farm.get("use_weather", True) is False or farm.get("offline"):
                weather = empty_weather("disabled", "Weather lookup was switched off by the farmer.")
            elif coordinate_error:
                weather = empty_weather(coordinate_error, "Add valid farm coordinates in the Pakistan operating region for outdoor weather context.")
            else:
                weather = self.provider.get(location["latitude"], location["longitude"], now=now, cache_only=cache_only)
            soil_map = soil_future.result()
        soil_map["scope"] = "regional_ground_soil_background"
        for key, label in {"ph": "pH", "ec": "EC"}.items():
            mapped = soil_map.get("measurements", {}).get(key, {})
            try:
                value = _numeric(mapped.get("value"))
                valid = 0 <= value <= (14 if key == "ph" else 1000)
                valid = valid and mapped.get("unit") == MEASUREMENT_RULES[key]["unit"]
                valid = valid and bool(mapped.get("source")) and bool(mapped.get("source_url"))
            except ValueError:
                valid = False
            if measurements[key]["kind"] == "missing" and valid and environment == "outdoor" and root_system == "soil":
                measurements[key] = {**deepcopy(mapped), "value": value, "kind": "modelled", "freshness": "background",
                    "sample_type": "mapped_ground_soil", "scope": "regional_ground_soil_background", "applicable": True,
                    "measured_at": None, "valid_at": None, "age_hours": None}
                decisions.append(decision("use_regional_soil_background", "No valid farmer reading was supplied; retain a source-labelled map estimate as background only.",
                                          measurement=key, source=mapped["source"], dataset_version=mapped.get("dataset_version")))
            elif measurements[key]["kind"] == "missing":
                if root_system != "soil" or environment != "outdoor":
                    reason = f"A {label} reading from the actual growing medium is needed; geographic ground-soil estimates are background only."
                    reason_code = "not_applicable"
                else:
                    reason = mapped.get("reason") or " ".join(soil_map.get("notes", [])) or f"No usable regional {label} estimate is available for this location."
                    reason_code = mapped.get("reason_code") or soil_map.get("reason_code") or {
                        "no_location": "missing_location", "invalid_location": "invalid_location",
                        "not_applicable": "not_applicable", "disabled": "disabled",
                    }.get(soil_map.get("status"), "source_unavailable")
                    if not isinstance(reason_code, str) or reason_code not in SOIL_REASON_CODES:
                        reason_code = "source_unavailable"
                explain_missing(key, reason, reason_code)
        decisions.append(decision("select_soil_map_source", "Prefer farmer readings; retain map source, depth and resolution for automatic background estimates.",
                                  soil_map_status=soil_map.get("status")))
        decisions.append(decision("select_weather_source", "Use coordinate-specific weather context when available; preserve farmer inputs during service failure.",
                                  weather_status=weather["status"], feedback_cache_only=cache_only))
        if environment == "outdoor":
            for key, provider_key in {"air_temperature": "temperature", "relative_humidity": "humidity"}.items():
                if measurements[key]["kind"] == "missing" and weather["current"].get(provider_key) is not None:
                    measurements[key] = self._estimate(key, weather["current"][provider_key], weather, weather.get("valid_at"), now=now, scope="outdoor_air")
                elif measurements[key]["kind"] == "missing":
                    reason = (" ".join(weather.get("notes", [])) if weather["status"] not in {"live", "cached", "stale"}
                              else f"No usable recent outdoor {key.replace('_', ' ')} value was returned by the weather provider.")
                    explain_missing(key, reason, weather["status"] if weather["status"] not in {"live", "cached", "stale"} else "provider_value_unavailable")
            if root_system == "soil":
                for key in ("moisture", "root_temperature"):
                    estimate = weather.get("root_estimates", {}).get(key, {})
                    if measurements[key]["kind"] == "missing" and estimate.get("value") is not None:
                        measurements[key] = self._estimate(key, estimate["value"], weather, estimate.get("valid_at"), now=now,
                                                           scope="outdoor_soil", depth_cm=estimate.get("depth_cm"))
                    elif measurements[key]["kind"] == "missing":
                        explain_missing(key, estimate.get("reason") or "No usable current-hour outdoor soil estimate was returned.",
                                        estimate.get("reason_code") or "provider_value_unavailable")
                        measurements[key].update(provider_variable=estimate.get("provider_variable"),
                                                 depth_cm=estimate.get("depth_cm"), valid_at=estimate.get("valid_at"),
                                                 retrieved_at=weather.get("retrieved_at"))
            else:
                decisions.append(decision("withhold_root_estimates", "Outdoor soil estimates are not measurements of substrate or a hydroponic solution."))
                for key in ("moisture", "root_temperature"):
                    explain_missing(key, "Outdoor soil estimates are not used for the selected growing medium; add a reading from the actual root environment.",
                                    "root_medium_excluded")
        else:
            notes.append("Outdoor weather is background context only; indoor growing-area air and root measurements must come from the farmer.")
            decisions.append(decision("withhold_protected_environment_substitution", "Outdoor air and soil estimates do not represent measured greenhouse, tunnel or indoor conditions."))
            for key in ("air_temperature", "relative_humidity", "moisture", "root_temperature"):
                explain_missing(key, "Outdoor estimates are not used as growing-area measurements for the selected protected or unknown environment; add a reading from the crop area.",
                                "growing_environment_excluded")
        for key, item in measurements.items():
            if key in input_errors:
                item["input_error"] = input_errors[key]
                if item["kind"] == "modelled":
                    item["reason"] = f"Farmer input was rejected: {input_errors[key]}. A labelled model estimate is shown instead."
            if item["kind"] == "modelled":
                decisions.append(decision("use_labelled_model_estimate", "No valid farmer measurement was supplied and this source matches the growing setup.",
                                          measurement=key, source=item["source"], freshness=item["freshness"]))
        observations = farm.get("observations") or {}
        observations = observations if isinstance(observations, dict) else {}
        root_zone = observations.get("root_zone")
        observations = {"root_zone": root_zone if isinstance(root_zone, str) and root_zone in {"dry", "moist", "wet"} else "unknown",
                        "notes": str(observations.get("notes") or "")[:2000]}
        missing = [key for key, value in measurements.items() if value["kind"] == "missing" and value["applicable"]]
        if environment == "unknown":
            missing.append("environment")
        if root_system == "unknown":
            missing.append("root_system")
        if coordinate_error:
            missing.append("farm_location")
        if weather["status"] not in {"live", "cached", "stale"} and farm.get("use_weather", True):
            missing.append("outdoor_weather")
        notes.extend(weather["notes"])
        notes.extend(soil_map.get("notes", []))
        notes.append("Mapped soil chemistry describes regional background; it is not a current farm test or a nutrient-solution reading.")
        data = {"location": location, "environment": environment, "root_system": root_system,
                "growth_stage": setup["growth_stage"], "irrigation": setup["irrigation"],
                "input_provenance": input_provenance,
                "weather": weather, "soil_map": soil_map, "measurements": measurements, "observations": observations,
                "missing": missing, "notes": notes}
        if cache_only:
            data["clarification"] = {"action": "summarize_missing", "missing": missing,
                                     "summary": "Available inputs were retained; missing values were not invented.", "network_requested": False}
        self.state["last_location"] = {key: location[key] for key in ("latitude", "longitude")}
        self.state["last_missing"] = missing
        self.state["weather_status"] = weather["status"]
        status = "partial" if missing or weather["status"] == "stale" or any(
            item["kind"] == "measured" and item["freshness"] != "fresh" for item in measurements.values()) else "ok"
        return self.result(data=data, status=status, decisions=decisions,
                           messages=[{"to": "recommendation", "type": "environment_context", "missing": missing,
                                      "weather_status": weather["status"]}])

    @staticmethod
    def _estimate(key, value, weather, valid_at, *, now=None, **details):
        # Check observation age, not download age.
        now = now or datetime.now(timezone.utc)
        try:
            age_hours = round(max(0, (now - parse_utc(valid_at)).total_seconds() / 3600), 3)
        except (ValueError, TypeError):
            age_hours = None
        return {"value": value, "unit": MEASUREMENT_RULES[key]["unit"], "source": weather["source"],
                "source_url": weather["source_url"], "measured_at": None, "valid_at": valid_at,
                "retrieved_at": weather["retrieved_at"], "kind": "modelled", "applicable": True,
                "freshness": "unknown" if age_hours is None else "stale" if weather["status"] == "stale" else "fresh",
                "age_hours": age_hours, "retrieval_age_hours": weather["age_hours"],
                **details}
