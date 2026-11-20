'SoilGrids background maps, using local subsets, point cache, then bounded WCS.'
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
from zipfile import BadZipFile

from .providers import atomic_json, parse_utc

DOCS = "https://docs.isric.org/globaldata/soilgrids/wcs.html"
LAYERS_DOCS = "https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html"
SOURCE = "ISRIC SoilGrids background soil map"
WCS_URL = "https://maps.isric.org/mapserv"
PROPERTIES = {
    "ph": {"layer": "phh2o", "unit": "pH", "range": [0, 14], "raw_divisor": 10},
    "sand": {"layer": "sand", "unit": "%", "range": [0, 100], "raw_divisor": 10},
    "silt": {"layer": "silt", "unit": "%", "range": [0, 100], "raw_divisor": 10},
    "clay": {"layer": "clay", "unit": "%", "range": [0, 100], "raw_divisor": 10},
    "ec": {"layer": None, "unit": "dS/m", "range": [0, 1000], "raw_divisor": 1,
           "dataset_version": "HWSD2.0", "depth_cm": [0, 20], "statistic": "dominant_soil_unit",
           "source": "FAO/IIASA HWSD2.0 background soil map",
           "source_url": "https://www.fao.org/soils-portal/data-hub/soil-maps-and-databases/harmonized-world-soil-database-v20/en/"},
}
DEFAULTS = {"enabled": True, "online_enabled": True, "map_directory": "data/soilmaps",
            "cache_directory": "runtime/soil_map_cache", "dataset_version": "SoilGrids2.0",
            "depth_cm": [5, 15], "statistic": "mean", "timeout_seconds": 4.0,
            "failure_cache_seconds": 1800, "capabilities_cache_days": 30,
            "point_half_width_degrees": 0.005}


class SoilMapLookupError(ValueError):
    """A known coverage condition, separate from service or decoding failure."""

    def __init__(self, message, reason_code="no_coverage", **details):
        super().__init__(message)
        self.reason_code = reason_code
        self.details = details


def _failure_code(error):
    code = getattr(error, "reason_code", None)
    if code in {"urban_cell", "no_coverage", "invalid_location"}:
        return code
    # Keep coverage errors separate from timeouts.
    message = str(error).lower()
    if "urban/mining" in message:
        return "urban_cell"
    if "nodata" in message or "outside this soil-map subset" in message:
        return "no_coverage"
    return "source_unavailable"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fetch_bytes(url, timeout=4.0, max_bytes=8_000_000):
    """One bounded request, with no hidden retries or unbounded read."""
    request = Request(url, headers={"User-Agent": "PakistanLettuceDSS/1.1 (research)",
                                    "Accept": "image/tiff,application/xml,text/xml"})
    start = time.monotonic()
    with urlopen(request, timeout=timeout) as response:
        blocks, size = [], 0
        while True:
            block = response.read(min(65536, max_bytes + 1 - size))
            if not block:
                break
            blocks.append(block)
            size += len(block)
            if size > max_bytes:
                raise ValueError("Soil map response exceeds the size limit")
            if time.monotonic() - start > timeout * 1.5:
                raise TimeoutError("Soil map response exceeded the time budget")
    return b"".join(blocks)


def _valid_coordinate(lat, lon):
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
               for x in (lat, lon)) and -90 <= lat <= 90 and -180 <= lon <= 180


def _safe_json(path):
    try:
        if path.stat().st_size > 10_000_000:
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _cache_write(path, payload):
    """Caching improves availability but is never required to return a value."""
    try:
        atomic_json(path, payload)
        return True
    except (OSError, ValueError, TypeError):
        return False


def _rasterio():
    try:
        import rasterio
        return rasterio
    except ImportError as exc:
        raise RuntimeError("GeoTIFF support is not installed. Run: python -m pip install -r requirements.txt") from exc


def sample_raster(source, latitude, longitude):
    'Read the pixel containing the coordinate, honoring CRS, bounds and nodata.'
    if not isinstance(source, bytes) and Path(source).suffix.lower() == ".npz":
        return sample_grid(source, latitude, longitude)
    rasterio = _rasterio()
    from contextlib import ExitStack
    from rasterio.warp import transform
    from rasterio.windows import Window
    with ExitStack() as stack:
        if isinstance(source, bytes):
            memory = stack.enter_context(rasterio.io.MemoryFile(source))
            dataset = stack.enter_context(memory.open())
        else:
            dataset = stack.enter_context(rasterio.open(source))
        if dataset.count != 1 or dataset.crs is None:
            raise ValueError("Soil raster needs one band and an explicit coordinate reference system")
        xs, ys = transform("EPSG:4326", dataset.crs, [longitude], [latitude])
        x, y = xs[0], ys[0]
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("Soil raster coordinate transformation failed")
        row, column = dataset.index(x, y)
        if not (0 <= row < dataset.height and 0 <= column < dataset.width):
            raise SoilMapLookupError("The requested location is outside this soil-map subset")
        cell = dataset.read(1, window=Window(column, row, 1, 1), masked=True)
        if bool(cell.mask is True) or bool(getattr(cell, "mask", False).any()):
            raise SoilMapLookupError("The soil map has no land-soil prediction at this location (nodata)")
        value = float(cell[0, 0])
        if not math.isfinite(value):
            raise ValueError("Soil map cell is non-finite")
        return value


def sample_grid(path, latitude, longitude, *, with_details=False):
    'Sample an exported north-up EPSG:4326 grid using only NumPy.'
    import numpy as np
    if Path(path).stat().st_size > 100_000_000:
        raise ValueError("Packaged soil grid exceeds the size limit")
    with Path(path).open("rb") as stream, np.load(stream, allow_pickle=False) as grid:
        values = grid["values"]
        transform = np.asarray(grid["transform"], dtype=float)
        crs = str(grid["crs"].item())
        nodata = grid["nodata"].item()
        if values.ndim != 2 or values.size > 50_000_000 or transform.shape != (6,) or not np.isfinite(transform).all():
            raise ValueError("Invalid soil grid dimensions or affine transform")
        a, b, c, d, e, f = transform
        if crs != "EPSG:4326" or b != 0 or d != 0 or a <= 0 or e >= 0:
            raise ValueError("Packaged soil grids must be north-up EPSG:4326")
        # Correct grid-edge rounding only.
        row_position, column_position = (latitude - f) / e, (longitude - c) / a
        if row_position < 0 and abs(latitude - f) <= 1e-12:
            row_position = 0.0
        if column_position < 0 and abs(longitude - c) <= 1e-12:
            column_position = 0.0
        row, column = math.floor(row_position), math.floor(column_position)
        if not (0 <= row < values.shape[0] and 0 <= column < values.shape[1]):
            raise SoilMapLookupError("The requested location is outside this soil-map subset")
        value = float(values[row, column])
        point_details = {}
        for field in ("mapping_unit_id", "component_share", "component_sequence", "component_count"):
            if field in grid:
                array = grid[field]
                if array.shape != values.shape:
                    raise ValueError("Soil-map provenance grid does not match its value grid")
                scalar = float(array[row, column])
                if math.isfinite(scalar) and scalar >= 0:
                    point_details[field] = int(scalar) if scalar.is_integer() else scalar
        if not math.isfinite(value) or value == nodata:
            if point_details.get("mapping_unit_id") == 7001:
                # HWSD2 unit 7001 is urban and has no EC value.
                raise SoilMapLookupError("HWSD2 maps this location as urban/mining land (mapping unit 7001) and supplies no soil EC estimate (nodata).",
                                         "urban_cell", **point_details)
            if "mapping_unit_id" in point_details:
                raise SoilMapLookupError(f"HWSD2 mapping unit {point_details['mapping_unit_id']} has no usable soil EC estimate (nodata).",
                                         **point_details)
            raise SoilMapLookupError("The soil map has no land-soil prediction at this location (nodata)")
        return {"value": value, **point_details} if with_details else value


def coverage_ids(xml):
    """Discover IDs rather than assuming that a server advertises a layer."""
    if not isinstance(xml, bytes) or len(xml) > 2_000_000:
        raise ValueError("Invalid WCS capabilities response")
    root = ET.fromstring(xml)
    names = []
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "CoverageOfferingBrief":
            names.extend(child.text.strip() for child in element if child.tag.rsplit("}", 1)[-1] == "name" and child.text)
        elif tag == "CoverageId" and element.text:
            names.append(element.text.strip())
    if not names:
        raise ValueError("The WCS service did not advertise any usable coverage IDs")
    return sorted(set(names))


def coverage_url(layer, coverage_id, bounds, resolution_m=250):
    'WCS 1.0 BBOX uses longitude,latitude order for EPSG:4326.'
    west, south, east, north = map(float, bounds)
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Invalid geographic subset bounds")
    middle_lat = (south + north) / 2
    res_y = resolution_m / 111320.0
    res_x = resolution_m / (111320.0 * max(0.1, math.cos(math.radians(middle_lat))))
    params = {"map": f"/map/{layer}.map", "SERVICE": "WCS", "VERSION": "1.0.0",
              "REQUEST": "GetCoverage", "COVERAGE": coverage_id, "CRS": "EPSG:4326",
              "RESPONSE_CRS": "EPSG:4326", "BBOX": ",".join(format(v, ".12g") for v in bounds),
              "RESX": format(res_x, ".12g"), "RESY": format(res_y, ".12g"),
              "FORMAT": "GEOTIFF_INT16"}
    return WCS_URL + "?" + urlencode(params)


class SoilMapProvider:
    """Inject ``fetch(url, timeout, max_bytes)`` / ``sampler(source, lat, lon)`` for tests."""

    def __init__(self, project_root=None, config_path=None, *, fetch=None, sampler=None):
        self.root = Path(project_root or Path(__file__).resolve().parents[1])
        self.config = {**DEFAULTS, **_safe_json(Path(config_path) if config_path else self.root / "config/soil_maps.json")}
        self.map_dir = self.root / self.config["map_directory"]
        self.cache_dir = self.root / self.config["cache_directory"]
        self.fetch = fetch or fetch_bytes
        self.sampler = sampler or sample_raster
        self.timeout = min(10.0, max(0.2, float(self.config["timeout_seconds"])))

    def _metadata(self, key, metadata=None):
        metadata = metadata or {}
        prop = PROPERTIES[key]
        details = {"unit": prop["unit"], "source": metadata.get("source", prop.get("source", SOURCE)),
                "source_url": metadata.get("source_url", prop.get("source_url", DOCS)), "dataset_version": metadata.get("dataset_version", prop.get("dataset_version", self.config["dataset_version"])),
                "depth_cm": metadata.get("depth_cm", prop.get("depth_cm", self.config["depth_cm"])),
                "statistic": metadata.get("statistic", prop.get("statistic", self.config["statistic"])),
                "resolution_m": metadata.get("resolution_m"), "source_resolution_m": metadata.get("source_resolution_m", metadata.get("resolution_m")),
                "freshness": "background", "retrieved_at": metadata.get("retrieved_at"),
                "valid_at": None, "measured_at": None, "applicable": True,
                "sample_type": "soil", "dataset_note": "Static geographic prediction, not a current farm measurement."}
        for field in ("measurement_method", "aggregation_method", "observation_period", "limitations", "license", "citation", "source_sha256", "source_tiff_sha256", "database_sha256", "raster_sha256", "product_variant", "mapping_unit_id", "component_share", "component_sequence", "component_count"):
            if field in metadata:
                details[field] = metadata[field]
        return details

    def _identity_matches(self, key, metadata):
        prop = PROPERTIES[key]
        return all(metadata.get(field) == prop.get(field, self.config[field])
                   for field in ("dataset_version", "depth_cm", "statistic"))

    def _measurement(self, key, raw, metadata, method):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            raise ValueError("Soil map did not supply a finite number")
        divisor = metadata.get("raw_divisor", PROPERTIES[key]["raw_divisor"])
        if isinstance(divisor, bool) or not isinstance(divisor, (int, float)) or not math.isfinite(divisor) or divisor <= 0:
            raise ValueError("Invalid soil-map scaling metadata")
        value = raw / divisor
        low, high = PROPERTIES[key]["range"]
        if not low <= value <= high:
            raise ValueError("Soil map value is outside the accepted physical range")
        return {**self._metadata(key, metadata), "value": round(value, 3), "kind": "modelled",
                "retrieved_at": metadata.get("retrieved_at") or utc_now(), "access_method": method}

    def _missing(self, key, reason, reason_code=None):
        return {**self._metadata(key), **getattr(reason, "details", {}), "value": None, "kind": "missing",
                "reason": str(reason), "reason_code": reason_code or _failure_code(reason)}

    def _point_path(self, key, lat, lon):
        # No coordinate rounding: a cache from a neighbouring point is never used.
        prop = PROPERTIES[key]
        identity = [float(lat).hex(), float(lon).hex(), key, prop.get("dataset_version", self.config["dataset_version"]),
                    prop.get("depth_cm", self.config["depth_cm"]), prop.get("statistic", self.config["statistic"])]
        digest = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
        return self.cache_dir / (digest + ".json")

    def _local(self, key, lat, lon):
        manifest = _safe_json(self.map_dir / "manifest.json")
        errors = []
        maps = manifest.get("maps", [])
        if not isinstance(maps, list):
            errors.append("The local soil-map manifest must contain a list of maps")
            maps = []
        for metadata in maps:
            if not isinstance(metadata, dict) or metadata.get("property") != key:
                continue
            if not self._identity_matches(key, metadata):
                errors.append(ValueError("The registered local soil map does not match the configured dataset, depth and statistic."))
                continue
            try:
                path = (self.map_dir / metadata["path"]).resolve()
                if not path.is_relative_to(self.map_dir.resolve()):
                    raise ValueError("Soil-map file must be inside the configured map directory")
                if not path.is_file():
                    errors.append(FileNotFoundError("A registered local soil-map file is missing."))
                    continue
                if key == "ec" and path.suffix.lower() == ".npz" and self.sampler is sample_raster:
                    sampled = sample_grid(path, lat, lon, with_details=True)
                    return self._measurement(key, sampled.pop("value"), {**metadata, **sampled}, "local_map"), errors
                return self._measurement(key, self.sampler(path, lat, lon), metadata, "local_map"), errors
            except (ValueError, OSError, RuntimeError, KeyError, TypeError, BadZipFile, EOFError) as exc:
                errors.append(exc)
        return None, errors

    def requested_coverage_id(self, key):
        layer = PROPERTIES[key]["layer"]
        if layer is None:
            raise ValueError("EC is available only from a registered local HWSD2 soil map, not SoilGrids WCS")
        depth = "-".join(str(number) for number in self.config["depth_cm"])
        return f"{layer}_{depth}cm_{self.config['statistic']}"

    def discover(self, key):
        layer = PROPERTIES[key]["layer"]
        if layer is None:
            raise ValueError("EC is available only from a registered local HWSD2 soil map, not SoilGrids WCS")
        path = self.cache_dir / ("capabilities_" + layer + ".json")
        cached = _safe_json(path)
        try:
            age = (datetime.now(timezone.utc) - parse_utc(cached["retrieved_at"])).total_seconds()
            names = cached["coverage_ids"] if 0 <= age < self.config["capabilities_cache_days"] * 86400 else None
        except (KeyError, TypeError, ValueError):
            names = None
        if not names:
            url = WCS_URL + "?" + urlencode({"map": f"/map/{layer}.map", "SERVICE": "WCS", "VERSION": "1.0.0", "REQUEST": "GetCapabilities"})
            names = coverage_ids(self.fetch(url, self.timeout, 2_000_000))
            _cache_write(path, {"coverage_ids": names, "retrieved_at": utc_now(), "source_url": url})
        wanted = self.requested_coverage_id(key)
        if wanted not in names:
            raise ValueError(f"The WCS service does not advertise the requested {wanted} layer")
        return wanted

    def _one(self, key, lat, lon, cache_only, urban_context=False):
        local, errors = self._local(key, lat, lon)
        path = self._point_path(key, lat, lon)
        if local:
            _cache_write(path, {"latitude": lat, "longitude": lon, "measurement": local})
            return local
        cached = _safe_json(path)
        candidate = cached.get("measurement")
        if cached.get("latitude") == lat and cached.get("longitude") == lon and isinstance(candidate, dict):
            known_units_and_source = candidate.get("unit") == PROPERTIES[key]["unit"] and all(
                isinstance(candidate.get(field), str) and candidate[field].strip() for field in ("source", "source_url"))
            if self._identity_matches(key, candidate) and known_units_and_source:
                value = candidate.get("value")
                if not isinstance(value, bool) and isinstance(value, (float, int)) and math.isfinite(value) and PROPERTIES[key]["range"][0] <= value <= PROPERTIES[key]["range"][1]:
                    return {**candidate, "kind": "modelled", "freshness": "background", "valid_at": None, "measured_at": None, "access_method": "point_cache"}
        if PROPERTIES[key]["layer"] is None:
            reason = errors[-1] if errors else "No local HWSD2 background EC estimate covers this location."
            result = self._missing(key, reason)
            result["reason"] += " SoilGrids does not supply EC."
            return result
        if urban_context and errors and _failure_code(errors[-1]) == "no_coverage":
            result = self._missing(key, errors[-1])
            result["reason"] += " HWSD2 also identifies this coordinate as urban/mining land. Automatic online soil-map retries were skipped; use the cultivated plot coordinates or a measured value."
            result["coverage_context"] = {"reason_code": "urban_cell", "mapping_unit_id": 7001, "dataset_version": "HWSD2.0"}
            return result
        if cache_only or not self.config.get("online_enabled"):
            return self._missing(key, errors[-1] if errors else "No local or cached soil-map value covers this location; online map access is disabled.")

        def failed_fallback(reason):
            if errors and _failure_code(errors[-1]) in {"urban_cell", "no_coverage"}:
                result = self._missing(key, errors[-1])
                result["reason"] += " Online fallback: " + str(reason)
                result["fallback_reason_code"] = _failure_code(reason)
                return result
            return self._missing(key, reason)
        # Cache outages per soil property.
        failure_path = self.cache_dir / ("failure_" + PROPERTIES[key]["layer"] + ".json")
        failure = _safe_json(failure_path)
        try:
            age = (datetime.now(timezone.utc) - parse_utc(failure["failed_at"])).total_seconds()
            if 0 <= age < self.config["failure_cache_seconds"]:
                return failed_fallback("Online soil maps are temporarily unavailable; retry after the service cooldown. " + failure.get("reason", ""))
        except (KeyError, ValueError, TypeError):
            pass
        try:
            if self.sampler is sample_raster:
                _rasterio()  # Fail fast without four unnecessary network calls.
            identifier = self.discover(key)
            half = min(0.02, max(0.003, float(self.config["point_half_width_degrees"])))
            bounds = [max(-180, lon - half), max(-90, lat - half), min(180, lon + half), min(90, lat + half)]
            url = coverage_url(PROPERTIES[key]["layer"], identifier, bounds)
            payload = self.fetch(url, self.timeout, 2_000_000)
            metadata = {"source_url": url, "retrieved_at": utc_now(), "resolution_m": 250, "raw_divisor": PROPERTIES[key]["raw_divisor"]}
            result = self._measurement(key, self.sampler(payload, lat, lon), metadata, "wcs")
            _cache_write(path, {"latitude": lat, "longitude": lon, "measurement": result})
            return result
        except Exception as exc:
            # Provider and optional decoding failures must not stop diagnosis.
            reason = str(exc)[:400] or type(exc).__name__
            # Do not treat nodata as a service outage.
            local_problem = isinstance(exc, RuntimeError) or any(word in reason.lower() for word in ("nodata", "outside this", "physical range"))
            if not local_problem:
                _cache_write(failure_path, {"failed_at": utc_now(), "reason": reason})
            return failed_fallback(exc)

    def get(self, lat, lon, properties=("ph", "sand", "silt", "clay"), cache_only=False):
        keys = list(dict.fromkeys(properties))
        if any(key not in PROPERTIES for key in keys):
            raise ValueError("Supported soil-map properties: ph, sand, silt, clay, ec")
        out = {"source": "Geographic background soil maps" if "ec" in keys else SOURCE, "source_url": DOCS, "measurements": {}, "notes": [
            "Mapped pH and texture describe background land soil at 5–15 cm, not a current test or a hydroponic solution.",
            "Where supplied, EC is a separate HWSD2 soil-map estimate at 0–20 cm. No nutrient dose or current soil moisture is inferred from static maps."]}
        if not _valid_coordinate(lat, lon):
            code = "missing_location" if lat is None or lon is None else "invalid_location"
            out.update(status="invalid_location", reason_code=code,
                       measurements={key: self._missing(key, "Valid latitude and longitude are required", code) for key in keys})
            return out
        lat, lon = float(lat), float(lon)
        out["location"] = {"latitude": lat, "longitude": lon}
        if not self.config.get("enabled"):
            out.update(status="disabled", reason_code="disabled",
                       measurements={key: self._missing(key, "Soil-map access is disabled", "disabled") for key in keys})
            return out
        if keys:
            measurements = {}
            # Resolve local EC before online fallbacks.
            if "ec" in keys:
                measurements["ec"] = self._one("ec", lat, lon, cache_only)
            urban_context = measurements.get("ec", {}).get("reason_code") == "urban_cell"
            remaining = [key for key in keys if key != "ec"]
            if remaining:
                with ThreadPoolExecutor(max_workers=min(4, len(remaining))) as pool:
                    values = list(pool.map(lambda key: self._one(key, lat, lon, cache_only, urban_context), remaining))
                measurements.update(zip(remaining, values))
            out["measurements"] = {key: measurements[key] for key in keys}
        supplied = sum(item.get("value") is not None for item in out["measurements"].values())
        out["status"] = "available" if supplied == len(keys) and keys else "partial" if supplied else "unavailable"
        return out
