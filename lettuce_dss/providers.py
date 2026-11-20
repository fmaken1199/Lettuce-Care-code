'Bounded Open-Meteo adapter with coordinate-specific, timestamped disk cache.'
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.parse import urlencode
from urllib.request import Request, urlopen

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_DOCS = "https://open-meteo.com/en/docs"
SOURCE = "Open-Meteo weather models"
CURRENT_FIELDS = {"temperature": "temperature_2m", "humidity": "relative_humidity_2m",
                  "rain": "precipitation", "wind": "wind_speed_10m"}
SOIL_FIELDS = {"root_temperature": "soil_temperature_6cm", "moisture": "soil_moisture_3_to_9cm"}
SOIL_DETAILS = {"root_temperature": {"label": "soil temperature", "unit": "C", "depth_cm": 6},
                "moisture": {"label": "soil moisture", "unit": "% VWC", "depth_cm": [3, 9]}}
UNITS = {"temperature": "C", "humidity": "%", "rain": "mm", "wind": "km/h"}
_EXPECTED_UNITS = {"temperature_2m": {"°C", "C"}, "relative_humidity_2m": {"%"},
                   "precipitation": {"mm"}, "wind_speed_10m": {"km/h"},
                   "soil_temperature_6cm": {"°C", "C"},
                   "soil_moisture_3_to_9cm": {"m³/m³", "m3/m3"}}
_RANGES = {"temperature_2m": (-90, 65), "relative_humidity_2m": (0, 100),
           "precipitation": (0, 2000), "wind_speed_10m": (0, 500),
           "soil_temperature_6cm": (-70, 90), "soil_moisture_3_to_9cm": (0, 1)}


def parse_utc(value):
    """Parse a provider UTC time; naive provider timestamps explicitly mean GMT."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A timestamp is required")
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def fetch_json(url, timeout=5.0):
    req = Request(url, headers={"User-Agent": "PakistanLettuceDSS/1.0 (+research)",
                               "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        # This response is normally well below 100 KB. Bound unexpected replies.
        payload = response.read(2_000_001)
        if len(payload) > 2_000_000:
            raise ValueError("Weather response exceeds the size limit")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Weather response must be a JSON object")
    return value


def atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def empty_weather(status, reason):
    return {"status": status, "source": SOURCE, "source_url": OPEN_METEO_DOCS,
            "retrieved_at": None, "valid_at": None, "kind": "modelled", "scope": "outdoor",
            "current": dict.fromkeys(CURRENT_FIELDS), "units": dict(UNITS), "forecast": [],
            "root_estimates": {key: _missing_soil(key, reason, status) for key in SOIL_FIELDS},
            "notes": [reason], "age_hours": None,
            "current_age_hours": None, "rain_description": "total precipitation, including rain, showers and snow"}


def _value(value, variable, units):
    if not isinstance(units.get(variable), str) or units.get(variable) not in _EXPECTED_UNITS[variable]:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    lower, upper = _RANGES[variable]
    return float(value) if lower <= value <= upper else None


def _missing_soil(key, reason, reason_code, valid_at=None):
    details = SOIL_DETAILS[key]
    return {"value": None, "unit": details["unit"], "depth_cm": details["depth_cm"],
            "valid_at": valid_at, "provider_variable": SOIL_FIELDS[key],
            "reason": reason, "reason_code": reason_code}


def _soil_at_hour(hourly, units, times, start, indices):
    """Keep a separate failure reason for each requested soil layer."""
    estimates = {}
    index = indices.get(start)
    valid_at = start.isoformat() if index is not None else None
    for key, variable in SOIL_FIELDS.items():
        label = SOIL_DETAILS[key]["label"]
        values = hourly.get(variable)
        if variable not in hourly:
            code, reason = "missing_variable", f"Open-Meteo did not return {label} values."
        elif not isinstance(values, list):
            code, reason = "invalid_array", f"Open-Meteo returned an invalid {label} array."
        elif len(values) != len(times):
            code, reason = "misaligned_array", f"Open-Meteo {label} values did not match the hourly timestamps."
        elif variable not in units:
            code, reason = "missing_unit", f"Open-Meteo did not provide a unit for {label}."
        elif not isinstance(units[variable], str) or units[variable] not in _EXPECTED_UNITS[variable]:
            code, reason = "unsupported_unit", f"Open-Meteo returned an unsupported unit for {label}."
        elif index is None:
            code, reason = "current_hour_unavailable", f"Open-Meteo has no {label} record for the current UTC hour."
        elif values[index] is None:
            code, reason = "null_value", f"Open-Meteo returned no {label} value for the current UTC hour."
        elif isinstance(values[index], bool) or not isinstance(values[index], (int, float)) or not math.isfinite(values[index]):
            code, reason = "invalid_value", f"Open-Meteo returned a non-numeric or non-finite {label} value."
        elif _value(values[index], variable, units) is None:
            code, reason = "out_of_range", f"Open-Meteo returned a {label} value outside the accepted input range."
        else:
            value = float(values[index])
            estimates[key] = {"value": round(value * 100, 6) if key == "moisture" else value,
                              "unit": SOIL_DETAILS[key]["unit"], "depth_cm": SOIL_DETAILS[key]["depth_cm"],
                              "valid_at": valid_at, "provider_variable": variable}
            continue
        estimates[key] = _missing_soil(key, reason, code, valid_at)
    return estimates


def _normalise(payload: dict, *, now: datetime, retrieved_at: str, status: str) -> dict:
    """Keep only values with known units and correctly aligned timestamps."""
    if not isinstance(payload, dict) or payload.get("error"):
        raise ValueError("The weather provider returned an error")
    if payload.get("utc_offset_seconds", 0) != 0:
        raise ValueError("Expected a GMT weather response")
    current = payload.get("current") or {}
    current_units = payload.get("current_units") or {}
    hourly = payload.get("hourly") or {}
    hourly_units = payload.get("hourly_units") or {}
    if not all(isinstance(part, dict) for part in (current, current_units, hourly, hourly_units)):
        raise ValueError("Weather response sections must be objects")
    out = empty_weather(status, "Outdoor model estimates; farm conditions may differ.")
    retrieved = parse_utc(retrieved_at)
    out["retrieved_at"] = retrieved.isoformat()
    out["age_hours"] = round(max(0, (now - retrieved).total_seconds() / 3600), 3)
    try:
        current_time = parse_utc(current.get("time"))
        current_age = (now - current_time).total_seconds() / 3600
    except (ValueError, TypeError):
        current_time, current_age = None, None
    # Check observation age before using weather.
    usable_current = current_time is not None and -0.5 <= current_age <= 6
    if usable_current:
        out["valid_at"] = current_time.isoformat()
        out["current_age_hours"] = round(max(0, current_age), 3)
        out["current"] = {key: _value(current.get(variable), variable, current_units)
                          for key, variable in CURRENT_FIELDS.items()}
        interval = current.get("interval")
        out["current_interval_seconds"] = interval if isinstance(interval, int) and 0 < interval <= 3600 else None
        if current_age > 1:
            out["status"] = "stale"
            out["notes"].append("The current-weather timestamp is more than one hour old.")
    else:
        out["notes"].append("No recent, correctly timed current-weather values were available.")
    times = hourly.get("time", [])
    if not isinstance(times, list):
        raise ValueError("Hourly times must be a list")
    if len(times) > 400:
        raise ValueError("Unexpectedly large hourly weather response")
    all_variables = {**CURRENT_FIELDS, **SOIL_FIELDS}
    # Match each data array to its timestamps.
    arrays = {var: hourly.get(var) for var in all_variables.values()}
    arrays = {var: values for var, values in arrays.items()
              if isinstance(values, list) and len(values) == len(times)}
    rows = []
    seen = set()
    indices = {}
    for index, raw_time in enumerate(times):
        try:
            timestamp = parse_utc(raw_time)
        except (ValueError, TypeError):
            continue
        if timestamp in seen:
            raise ValueError("Duplicate hourly weather timestamps")
        seen.add(timestamp)
        indices[timestamp] = index
        row = {key: _value(arrays[variable][index], variable, hourly_units) if variable in arrays else None
               for key, variable in all_variables.items()}
        rows.append((timestamp, row))
    rows.sort(key=lambda item: item[0])
    start = now.replace(minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=48)
    out["forecast"] = [{"time": timestamp.isoformat(), **{k: row[k] for k in CURRENT_FIELDS}}
                       for timestamp, row in rows if start <= timestamp < end]
    out["forecast_interval"] = "hourly; precipitation is the preceding-hour sum"
    # Select root estimates for the current hour.
    out["root_estimates"] = _soil_at_hour(hourly, hourly_units, times, start, indices)
    if not any(v is not None for v in out["current"].values()) and not any(
            any(row.get(k) is not None for k in CURRENT_FIELDS) for row in out["forecast"]) and not any(
            item["value"] is not None for item in out["root_estimates"].values()):
        raise ValueError("No usable current, forecast or soil weather-model values were returned")
    if len(out["forecast"]) < 48:
        out["notes"].append(f"Only {len(out['forecast'])} of 48 requested future hourly records are available.")
    out["grid_location"] = {key: float(payload[key]) if isinstance(payload.get(key), (int, float))
                            and not isinstance(payload.get(key), bool) and math.isfinite(payload[key]) else None
                            for key in ("latitude", "longitude", "elevation")}
    return out


class WeatherProvider:
    def __init__(self, cache_dir, timeout=5.0, fetcher=None):
        self.cache_dir = Path(cache_dir)
        self.timeout = float(timeout)
        if not 0 < self.timeout <= 30:
            raise ValueError("Weather timeout must be between zero and 30 seconds")
        self.fetcher = fetcher or fetch_json
        self.fresh_seconds = 30 * 60
        self.stale_seconds = 6 * 60 * 60

    def _request(self, latitude, longitude):
        # Keep full coordinates in cache keys.
        params = {"latitude": float(latitude), "longitude": float(longitude),
                  "current": ",".join(CURRENT_FIELDS.values()),
                  "hourly": ",".join([*CURRENT_FIELDS.values(), *SOIL_FIELDS.values()]),
                  "forecast_hours": 48, "temperature_unit": "celsius", "wind_speed_unit": "kmh",
                  "precipitation_unit": "mm", "timezone": "GMT", "timeformat": "iso8601"}
        url = OPEN_METEO_URL + "?" + urlencode(params)
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return url, self.cache_dir / f"weather_{key}.json"

    def get(self, latitude, longitude, *, now=None, cache_only=False):
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                   for value in (latitude, longitude)) or not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            return empty_weather("invalid_location", "Valid latitude and longitude are required for weather lookup.")
        now = now or datetime.now(timezone.utc)
        url, path = self._request(latitude, longitude)
        cached, age = None, None
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(cached, dict):
                raise ValueError("Weather cache must be an object")
            if cached.get("url") != url:
                raise ValueError("Wrong cache coordinates")
            stamp = parse_utc(cached["retrieved_at"])
            age = (now - stamp).total_seconds()
            if not 0 <= age <= self.stale_seconds:
                cached = None
        except (OSError, ValueError, TypeError, KeyError):
            cached = None
        if cached and (age <= self.fresh_seconds or cache_only):
            try:
                return _normalise(cached["payload"], now=now, retrieved_at=cached["retrieved_at"],
                                  status="cached" if age <= self.fresh_seconds else "stale")
            except (ValueError, TypeError, KeyError):
                cached = None
        if cache_only:
            return empty_weather("unavailable", "No usable weather cache was available; feedback does not start another download.")
        try:
            payload = self.fetcher(url, self.timeout)
            result = _normalise(payload, now=now, retrieved_at=now.isoformat(), status="live")
            try:
                atomic_json(path, {"url": url, "retrieved_at": now.isoformat(), "payload": payload})
            except (OSError, ValueError, TypeError):
                result["notes"].append("Weather is available, but the local cache could not be written.")
            return result
        except Exception as exc:
            # Continue when a provider fails.
            error_type = type(exc).__name__
            if cached:
                try:
                    result = _normalise(cached["payload"], now=now, retrieved_at=cached["retrieved_at"], status="stale")
                    result["notes"].append(f"Weather refresh failed ({error_type}); the timestamped cache is being used.")
                    return result
                except (ValueError, TypeError, KeyError):
                    pass
            return empty_weather("unavailable", f"Weather could not be retrieved ({error_type}); farmer inputs remain available.")
