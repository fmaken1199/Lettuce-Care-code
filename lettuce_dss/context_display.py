'One plain-language environmental view for the app and both report exports.'
from __future__ import annotations

import html
import math

LABELS = {
    "ph": "pH", "ec": "Electrical conductivity (EC)",
    "moisture": "Volumetric moisture", "root_temperature": "Root-zone temperature",
    "sand": "Sand", "silt": "Silt", "clay": "Clay",
    "air_temperature": "Air temperature at the crop", "relative_humidity": "Relative humidity at the crop",
    "farm_location": "Farm location", "outdoor_weather": "Outdoor weather",
    "farmer_root_zone_observation": "Farmer root-zone observation (dry/moist/wet)",
    # Preserve readability when opening a record from version 1.0.0.
    "observed root-zone moisture": "Farmer root-zone observation (dry/moist/wet)",
}
HEADERS = ("Measurement", "Reading", "Evidence", "Details")


def context_label(key):
    return LABELS.get(str(key), str(key).replace("_", " "))


def _reading(value, unit=""):
    if value is None:
        return "Unavailable"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            return "Unavailable"
        value = f"{value:.6g}"
    return f"{value} {unit}".strip()


def _evidence(item):
    if item.get("applicable") is False:
        return "Not applicable"
    kind = {"measured": "Farmer measurement", "modelled": "Model estimate", "missing": "Missing"}.get(
        item.get("kind"), "Unknown")
    source = item.get("source")
    parts = [kind]
    if source and source != "farmer":
        parts.append(str(source))
    if item.get("kind") in {"measured", "modelled"}:
        freshness = item.get("freshness", "unknown")
        parts.append({"fresh": "fresh", "stale": "older reading", "unknown": "age unknown", "background": "background soil map"}.get(freshness, str(freshness)))
    return "; ".join(parts)


def _details(item):
    parts = []
    depth = item.get("depth_cm")
    if isinstance(depth, (tuple, list)) and len(depth) == 2:
        parts.append(f"Soil depth: {depth[0]}–{depth[1]} cm")
    elif depth is not None:
        parts.append(f"Soil depth: {depth} cm")
    for key, label in (("measured_at", "Measured"), ("valid_at", "Estimate time"), ("retrieved_at", "Retrieved")):
        if item.get(key):
            parts.append(f"{label}: {item[key]}")
    for key, label in (("version", "Dataset version"), ("dataset_version", "Dataset version"), ("resolution_m", "Map resolution (m)")):
        if item.get(key) is not None:
            parts.append(f"{label}: {item[key]}")
    if item.get("kind") == "measured" and not item.get("measured_at"):
        parts.append("Measurement date not supplied")
    if item.get("date_precision") == "date":
        parts.append("Date only; time was not supplied")
    if item.get("reason"):
        parts.append(str(item["reason"]))
    if item.get("reason_code"):
        parts.append("Availability code: " + str(item["reason_code"]))
    if item.get("input_error") and item["input_error"] != item.get("reason"):
        parts.append("Farmer input: " + str(item["input_error"]))
    return "\n".join(parts)


def environment_view(env: dict, farm_setup=None, location=None):
    env = env or {}
    setup = farm_setup or {k: env.get(k) for k in ("environment", "root_system")}
    loc = env.get("location") or location or {}
    location_parts = [str(loc["label"])] if loc.get("label") else []
    if loc.get("latitude") is not None and loc.get("longitude") is not None:
        location_parts.append(f"{loc['latitude']}, {loc['longitude']}")
    if loc.get("source") and location_parts:
        location_parts.append("Source: " + str(loc["source"]))
    if loc.get("accuracy_note"):
        location_parts.append(str(loc["accuracy_note"]))
    elif isinstance(loc.get("accuracy_m"), (int, float)) and not isinstance(loc["accuracy_m"], bool) and math.isfinite(loc["accuracy_m"]) and loc["accuracy_m"] >= 0:
        location_parts.append(f"Reported location accuracy: ±{loc['accuracy_m']:.0f} m")
    rows = []
    medium = {"soil": "Soil", "substrate": "Substrate", "hydroponic": "Nutrient solution"}.get(setup.get("root_system"), "Root zone")
    for key, item in env.get("measurements", {}).items():
        label = context_label(key)
        if key in {"ph", "ec", "moisture"}:
            label = medium + " " + label
        rows.append(dict(zip(HEADERS, [label,
            "Not applicable" if item.get("applicable") is False else _reading(item.get("value"), item.get("unit", "")),
            _evidence(item), _details(item)])))
    weather = env.get("weather", {})
    weather_rows = []
    for key, label, unit in (("temperature", "Outdoor air temperature", "C"), ("humidity", "Outdoor relative humidity", "%"),
                             ("rain", "Precipitation", "mm"), ("wind", "Wind speed", "km/h")):
        if key in weather.get("current", {}):
            detail = ""
            if key == "rain":
                interval = weather.get("current_interval_seconds")
                detail = f"Preceding {interval} seconds" if interval else "Current provider interval; not a daily total"
            weather_rows.append(dict(zip(HEADERS, [label, _reading(weather["current"][key], unit),
                                                  "Outdoor model estimate" if weather["current"][key] is not None else "Missing", detail])))
    # Sum the same forecast window as recommendations.
    forecast = weather.get("forecast", [])[:24]
    rain_rows = [r for r in forecast if isinstance(r.get("rain"), (int, float))
                 and not isinstance(r["rain"], bool) and math.isfinite(r["rain"])]
    count = len(rain_rows)
    detail = f"Sum of {count}/24 available hourly precipitation values; each covers the preceding hour."
    if forecast:
        detail += f"\nInterval end times: {forecast[0].get('time', 'unknown')} to {forecast[-1].get('time', 'unknown')}"
    weather_rows.append(dict(zip(HEADERS, ["Hourly precipitation outlook",
        _reading(sum(r["rain"] for r in rain_rows) if rain_rows else None, "mm"),
        "Model estimate; " + ("complete 24-record coverage" if count == 24 else "partial coverage" if count else "unavailable"), detail])))
    weather_info = ["Status: " + str(weather.get("status", "unavailable"))]
    for key, label in (("source", "Source"), ("valid_at", "Weather time"), ("retrieved_at", "Retrieved")):
        if weather.get(key):
            weather_info.append(f"{label}: {weather[key]}")
    observation = env.get("observations", {}).get("root_zone", "unknown")
    observation = "Not applicable to a nutrient solution" if setup.get("root_system") == "hydroponic" else (
        "Not supplied" if observation == "unknown" else str(observation).capitalize())
    soil_map = env.get("soil_map", {}) or {}
    map_rows = [dict(zip(HEADERS, ["Background soil " + context_label(key),
        _reading(item.get("value"), item.get("unit", "")), _evidence(item), _details(item)]))
        for key, item in soil_map.get("measurements", {}).items() if isinstance(item, dict)]
    map_info = ["Status: " + str(soil_map.get("status", "unavailable"))]
    if soil_map.get("source"):
        map_info.append("Source: " + str(soil_map["source"]))
    return {"location": "; ".join(location_parts) or "Not supplied",
            "setup": " / ".join(str(setup.get(k) or "unknown").replace("_", " ") for k in ("environment", "root_system")),
            "measurements": rows, "weather": weather_rows, "weather_info": weather_info,
            "soil_map": map_rows, "soil_map_info": map_info,
            "observation": observation,
            "notes": list(dict.fromkeys(str(n) for n in env.get("notes", []) + weather.get("notes", []) + soil_map.get("notes", [])))}


def environment_summary(env: dict, farm_setup=None):
    """Short readings, without turning background maps into crop measurements."""
    env = env or {}
    setup = farm_setup or env
    root = setup.get("root_system", "soil")
    parts = []
    labels = {"ph": "pH", "ec": "EC", "moisture": "moisture", "root_temperature": "root temperature"}
    readings = env.get("measurements", {})
    root_parts = []
    for key, label in labels.items():
        item = readings.get(key, {})
        if item.get("applicable") is False or item.get("value") is None or item.get("kind") == "missing":
            continue
        value = _reading(item["value"], "" if key == "ph" else item.get("unit", ""))
        if value == "Unavailable":
            continue
        estimate = item.get("kind") == "modelled"
        # A mapped soil value is not a solution or imported-substrate reading.
        background = item.get("freshness") == "background"
        if background and root != "soil":
            continue
        suffix = " (Estimated, background soil)" if background else " (Estimated)" if estimate else " (Farmer reading)"
        root_parts.append(label + " " + value + suffix)
    medium = {"soil": "Soil", "hydroponic": "Nutrient solution", "substrate": "Substrate"}.get(root, "Root zone")
    parts.append(medium + ": " + ("; ".join(root_parts) + "." if root_parts else "no readings available."))
    if root == "soil":
        map_values = env.get("soil_map", {}).get("measurements", {})
        texture = [context_label(key).lower() + " " + _reading(map_values[key].get("value"), map_values[key].get("unit", "%"))
                   for key in ("sand", "silt", "clay") if map_values.get(key, {}).get("value") is not None]
        if texture:
            parts.append("Background soil texture (Estimated): " + "; ".join(texture) + ".")
    weather = env.get("weather", {})
    current = weather.get("current", {})
    weather_values = []
    for key, label, unit in (("temperature", "air", "C"), ("humidity", "humidity", "%"), ("rain", "current rain", "mm")):
        if current.get(key) is not None:
            value = _reading(current[key], unit)
            if value != "Unavailable":
                weather_values.append(label + " " + value)
    if weather_values:
        qualifier = "Estimated, older data" if weather.get("status") == "stale" else "Estimated"
        parts.append("Outdoor weather (" + qualifier + "): " + "; ".join(weather_values) + ".")
    else:
        parts.append("Outdoor weather: no readings available.")
    unavailable = []
    for key in ("ph", "ec"):
        item = readings.get(key, {})
        if item.get("value") is None or item.get("kind") == "missing":
            unavailable.append(key)
            label = "pH" if key == "ph" else "EC"
            reasons = {
                "urban_cell": "the soil map classifies this location as urban land and has no usable soil value",
                "no_coverage": "the soil map has no usable value at these coordinates",
                "missing_location": "farm coordinates were not supplied",
                "invalid_location": "valid farm coordinates are required",
                "not_applicable": "outdoor soil-map values do not describe this growing setup; add a reading from the crop's actual root environment",
                "source_unavailable": "the soil-data source could not be reached or read",
                "disabled": "soil-map access is disabled",
            }
            reason = reasons.get(item.get("reason_code")) or str(item.get("reason") or "no reading or applicable estimate was supplied").rstrip(".")
            parts.append(label + " unavailable: " + reason + ".")
    if unavailable:
        parts.append("Recommendations use the available evidence; missing pH or EC values are not assumed. You can add a measured reading.")
    return parts


def markdown_text(value):
    # Raw HTML and table delimiters in farmer labels must remain plain text.
    text = html.escape(str(value), quote=False).replace("\\", "\\\\")
    for ch in ("|", "*", "_", "[", "]", "`", "#"):
        text = text.replace(ch, "\\" + ch)
    return text.replace("\r", " ").replace("\n", " / ")


def environment_markdown(view):
    lines = ["### Conditions used in this report", "", "Growing setup: " + markdown_text(view["setup"]), "",
             "Farm location: " + markdown_text(view["location"]), "", "Times below retain their recorded UTC offsets.", ""]
    sections = [("Soil and crop measurements", view["measurements"]), ("Outdoor weather", view["weather"])]
    if view.get("soil_map"):
        sections.append(("Background soil map", view["soil_map"]))
    for heading, rows in sections:
        lines += ["### " + heading, ""]
        if heading == "Outdoor weather":
            lines += [markdown_text("; ".join(view["weather_info"])), ""]
        if heading == "Background soil map":
            lines += [markdown_text("; ".join(view["soil_map_info"])), "", "Geographic soil background; this does not describe a nutrient solution or imported substrate.", ""]
        if rows:
            lines += ["| " + " | ".join(HEADERS) + " |", "| " + " | ".join("---" for _ in HEADERS) + " |"]
            lines += ["| " + " | ".join(markdown_text(row[h]) for h in HEADERS) + " |" for row in rows]
            lines.append("")
        else:
            lines += ["No readings are available in this record.", ""]
    lines += [context_label("farmer_root_zone_observation") + ": " + markdown_text(view["observation"]), ""]
    lines += ["- " + markdown_text(note) for note in view["notes"]]
    return "\n".join(lines) + "\n"


def environment_html(view):
    def e(value):
        return html.escape(str(value)).replace("\n", "<br>")
    parts = ['<div class="environment-details"><h3>Conditions used in this report</h3><p>Growing setup: ' + e(view["setup"]) +
             '<br>Farm location: ' + e(view["location"]) + '</p><small>Times below retain their recorded UTC offsets.</small>']
    sections = [("Soil and crop measurements", view["measurements"]), ("Outdoor weather", view["weather"])]
    if view.get("soil_map"):
        sections.append(("Background soil map", view["soil_map"]))
    for heading, rows in sections:
        parts.append('<h3>' + heading + '</h3>')
        if heading == "Outdoor weather":
            parts.append('<p>' + e("; ".join(view["weather_info"])) + '</p>')
        if heading == "Background soil map":
            parts.append('<p>' + e("; ".join(view["soil_map_info"])) + '</p><p>Geographic soil background; this does not describe a nutrient solution or imported substrate.</p>')
        if rows:
            parts.append('<div class="table-wrap"><table><thead><tr>' + ''.join('<th scope="col">' + h + '</th>' for h in HEADERS) + '</tr></thead><tbody>')
            parts += ['<tr>' + ''.join('<td>' + e(row[h]) + '</td>' for h in HEADERS) + '</tr>' for row in rows]
            parts.append('</tbody></table></div>')
        else:
            parts.append('<p>No readings are available in this record.</p>')
    parts.append('<p>' + context_label("farmer_root_zone_observation") + ': ' + e(view["observation"]) + '</p>')
    if view["notes"]:
        parts.append('<ul>' + ''.join('<li>' + e(note) + '</li>' for note in view["notes"]) + '</ul>')
    parts.append('</div>')
    return ''.join(parts)
