'Load and validate the single active treatment knowledge base.'
from __future__ import annotations

from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlparse

from lettuce_dss.core import CLASS_NAMES

KNOWLEDGE_FILE = "knowledge/treatment_knowledge_base.json"
ROOT_SYSTEMS = {"soil", "substrate", "hydroponic"}
ENVIRONMENTS = {"outdoor", "greenhouse", "tunnel", "indoor"}
CONTEXT_VALUES = {
    "growth_stage": {"seedling", "vegetative", "near_harvest"},
    "irrigation": {"drip", "overhead", "manual", "recirculating"},
}
BUILTIN_HANDLERS = dict(zip(
    (f"R{i:02d}" for i in range(1, 14)),
    ("withhold_actions", "select_class_actions", "preserve_broad_label",
     "hydroponic_context", "respond_to_root_observation", "prioritise_airflow_check",
     "avoid_outdoor_proxy", "schedule_moisture_check", "use_ph_as_background",
     "apply_report_feedback", "omit_incompatible_action", "rank_actions", "select_medication")))
BROAD_LABELS = {"Bacterial", "Viral", "Wilt_and_Leaf_Blight"}


def _require(condition, message):
    if not condition:
        raise ValueError("Invalid treatment knowledge base: " + message)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _ids(value, available, field, *, empty=True):
    _require(isinstance(value, list) and all(_text(x) for x in value), field + " must be a list of IDs")
    _require(len(value) == len(set(value)), field + " contains duplicate IDs")
    _require((empty or bool(value)) and set(value) <= set(available), field + " has missing or unknown links")


def _score(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 100


def _validate_knowledge(database: dict) -> None:
    """Fail closed on invalid taxonomy, scope, evidence links or rule syntax."""
    _require(isinstance(database, dict), "root must be an object")
    _require(type(database.get("schema_version")) is int and database["schema_version"] == 1,
             "unsupported schema version")
    for key in ("metadata", "policy", "diseases", "treatments", "sources", "rules"):
        _require(isinstance(database.get(key), dict), key + " must be an object")
    metadata, policy = database["metadata"], database["policy"]
    for key in ("version", "scope", "review_date"):
        _require(_text(metadata.get(key)), "metadata." + key + " is required")
    try:
        date.fromisoformat(metadata["review_date"])
    except ValueError:
        raise ValueError("Invalid treatment knowledge base: review_date must be an ISO date") from None
    _require(type(policy.get("maximum_actions")) is int and 1 <= policy["maximum_actions"] <= 20,
             "maximum_actions must be an integer from 1 to 20")
    _require(type(policy.get("maximum_medication_options")) is int and policy["maximum_medication_options"] == 2,
             "maximum_medication_options must be 2")
    _require(_score(policy.get("protected_humidity_attention_percent")), "invalid humidity attention threshold")
    sources, treatments, diseases, rules = (database[key] for key in ("sources", "treatments", "diseases", "rules"))
    _require(bool(sources), "source registry is empty")
    for sid, source in sources.items():
        _require(_text(sid) and isinstance(source, dict), "malformed source")
        _require(source.get("id", sid) == sid, "source key and id differ: " + sid)
        _require(all(_text(source.get(k)) for k in ("title", "publisher", "url")), "incomplete source: " + sid)
        url = urlparse(source["url"])
        _require(url.scheme == "https" and bool(url.netloc), "sources require HTTPS URLs: " + sid)
    care, medication = set(), set()
    for tid, treatment in treatments.items():
        _require(_text(tid) and isinstance(treatment, dict), "malformed treatment")
        kind = treatment.get("treatment_type")
        _require(kind in {"care", "chemical", "biological"}, "unknown treatment_type: " + tid)
        _ids(treatment.get("source_ids"), sources, tid + ".source_ids",
             empty=kind == "care" and tid == "retake_image" and treatment.get("evidence_type") == "application input policy")
        if kind == "care":
            care.add(tid)
            _require(all(_text(treatment.get(k)) for k in ("title", "detail", "category", "reason")), "incomplete care: " + tid)
            _require(treatment["category"] not in {"chemical", "biological"}, "medication cannot be disguised as care: " + tid)
            _require(_score(treatment.get("priority")), "invalid care priority: " + tid)
        else:
            medication.add(tid)
            for key in ("active_ingredient", "purpose", "application_note", "evidence_scope", "jurisdiction", "review_status"):
                _require(_text(treatment.get(key)), "missing " + key + ": " + tid)
            _require(treatment.get("regulatory_status") == "not_assessed_for_user_location",
                     "international references cannot imply local authorization: " + tid)
            _require(all(key in treatment and treatment[key] is None for key in ("product_name", "dose", "phi")),
                     "reference cards cannot specify a product, dose or PHI: " + tid)
        for key, allowed in (("root_systems", ROOT_SYSTEMS), ("environments", ENVIRONMENTS)):
            if key in treatment or kind != "care":
                _ids(treatment.get(key), allowed, tid + "." + key, empty=False)
        if "diseases" in treatment:
            _ids(treatment["diseases"], CLASS_NAMES, tid + ".diseases", empty=False)
    _require(set(diseases) == set(CLASS_NAMES), "exactly the ten classifier classes are required")
    for label, disease in diseases.items():
        _require(isinstance(disease, dict), "malformed disease: " + label)
        _require(all(_text(disease.get(k)) for k in ("display_name", "summary")), "incomplete disease: " + label)
        _require(type(disease.get("broad_label")) is bool and disease["broad_label"] == (label in BROAD_LABELS),
                 "broad-label scope cannot be changed: " + label)
        _ids(disease.get("action_ids"), care, label + ".action_ids", empty=False)
        med = disease.get("medication")
        _require(isinstance(med, dict) and _text(med.get("summary")), "missing medication outcome: " + label)
        _require(med.get("status") in {"options", "not_needed", "not_selected", "no_curative", "unavailable"},
                 "invalid medication status: " + label)
        _ids(med.get("option_ids"), medication, label + ".option_ids")
        _ids(med.get("source_ids"), sources, label + ".medication.source_ids")
        _require((med["status"] == "options") == bool(med["option_ids"]), "medication status/links disagree: " + label)
        _require(label not in BROAD_LABELS | {"Healthy", "Lettuce_Mosaic_Virus"} or not med["option_ids"],
                 "no targeted medication permitted for this label: " + label)
        for tid in med["option_ids"]:
            if "diseases" in treatments[tid]:
                _require(label in treatments[tid]["diseases"], "inconsistent treatment-disease link: " + tid)
    _ids(database.get("uncertain_action_ids"), care, "uncertain_action_ids", empty=False)
    required_actions = {"routine_monitor", "retake_image", "steady_moisture", "hydroponic_flow", "hydroponic_measure",
                        "check_drainage", "check_dry_root", "protected_airflow", "rain_irrigation_check", "soil_test_plan", "dry_foliage"}
    _require(required_actions <= care, "a built-in policy references missing care actions")
    _require(set(BUILTIN_HANDLERS) <= set(rules), "all R01-R13 guard policies are required")
    for rid, rule in rules.items():
        _require(isinstance(rid, str) and re.fullmatch(r"R[0-9]{2,}", rid) and isinstance(rule, dict), "malformed rule")
        _require(rule.get("id") == rid, "rule key and id differ: " + rid)
        _require(all(_text(rule.get(k)) for k in ("name", "description", "handler")), "incomplete rule: " + rid)
        _require(rule.get("kind") in {"application_policy", "mixed", "evidence_based"}, "invalid rule kind: " + rid)
        _ids(rule.get("source_ids"), sources, rid + ".source_ids")
        common_fields = {"id", "name", "kind", "handler", "description", "source_ids"}
        allowed_fields = common_fields if rid in BUILTIN_HANDLERS else common_fields | {"when", "add_action_ids", "priority_overrides"}
        _require(set(rule) <= allowed_fields, "unsupported rule fields: " + rid)
        if rid in BUILTIN_HANDLERS:
            _require(rule["handler"] == BUILTIN_HANDLERS[rid], "cannot replace guard handler: " + rid)
            _require(not any(key in rule for key in ("when", "add_action_ids", "priority_overrides")), "guards cannot carry executable context clauses: " + rid)
            continue
        _require(rule["handler"] == "context_action", "unknown context handler: " + rid)
        clauses = rule.get("when")
        _require(isinstance(clauses, list) and bool(clauses), "context rule requires conditions: " + rid)
        fields = []
        for clause in clauses:
            _require(isinstance(clause, dict) and set(clause) == {"field", "equals"}, "only field/equals clauses are supported: " + rid)
            field, value = clause["field"], clause["equals"]
            _require(isinstance(field, str) and field in CONTEXT_VALUES and isinstance(value, str)
                     and value in CONTEXT_VALUES[field], "invalid context field/value: " + rid)
            fields.append(field)
        _require(len(fields) == len(set(fields)), "duplicate context fields: " + rid)
        _ids(rule.get("add_action_ids"), care, rid + ".add_action_ids")
        priorities = rule.get("priority_overrides", {})
        _require(isinstance(priorities, dict) and set(priorities) <= care and all(_score(v) for v in priorities.values()),
                 "invalid context priorities: " + rid)
        _require(bool(rule["add_action_ids"] or priorities), "context rule has no action: " + rid)


def validate_knowledge(database: dict) -> None:
    """Normalize malformed data failures to an explicit validation error."""
    try:
        _validate_knowledge(database)
    except (TypeError, KeyError, AttributeError) as exc:
        raise ValueError("Invalid treatment knowledge base: malformed field type or missing required field") from exc


def parse_knowledge(raw: bytes) -> dict:
    """Validate serialized data and reject duplicate keys before schema checks."""
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate JSON key: " + str(key))
            result[key] = value
        return result
    database = json.loads(raw, object_pairs_hook=unique_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON value: " + value)))
    validate_knowledge(database)
    database["_knowledge_sha256"] = hashlib.sha256(raw).hexdigest()
    return database


def load_knowledge(project_root: Path) -> dict:
    """Read only the canonical active JSON file, validate, then attach its hash."""
    return parse_knowledge((Path(project_root) / KNOWLEDGE_FILE).read_bytes())


def load_sources(project_root: Path) -> dict:
    return load_knowledge(project_root)["sources"]


def knowledge_identity(database: dict) -> dict:
    return {"version": database["metadata"]["version"],
            "sha256": database.get("_knowledge_sha256") or hashlib.sha256(
                json.dumps(database, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "schema_version": database["schema_version"], "scope": database["metadata"]["scope"],
            "review_date": database["metadata"]["review_date"], "counts": count_knowledge(database)}


def count_knowledge(database: dict) -> dict:
    """Count registry records, not triggered rules or number of research papers."""
    treatments = database["treatments"]
    medications = [value for value in treatments.values() if value["treatment_type"] != "care"]
    return {"classes": len(database["diseases"]),
            "care_actions": sum(value["treatment_type"] == "care" for value in treatments.values()),
            "medication_options": len(medications),
            "unique_active_ingredients": len({value["active_ingredient"].strip().casefold() for value in medications}),
            "sources": len(database["sources"]),
            "unique_source_urls": len({value["url"].rstrip("/") for value in database["sources"].values()}),
            "rules": len(database["rules"])}


def source_records(database: dict, source_ids: list) -> list:
    return [{"id": sid, **{key: database["sources"][sid][key] for key in ("title", "publisher", "url")}}
            for sid in sorted(set(source_ids))]

