'Deterministic recommendation selection with source-linked decision traces.'
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

from lettuce_dss.core import BaseAgent, decision
from lettuce_dss.medication import MedicationSelector
from lettuce_dss.knowledge import (load_knowledge, validate_knowledge, knowledge_identity,
                                  source_records, CONTEXT_VALUES, ROOT_SYSTEMS, ENVIRONMENTS)


def _number(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


class RecommendationAgent(BaseAgent):
    name = "recommendation"
    goal = "Select source-backed lettuce actions and medication references that fit the diagnosis and growing conditions."

    def __init__(self, project_root: Path):
        super().__init__()
        self.root = Path(project_root)
        self.database = load_knowledge(self.root)
        self.identity = knowledge_identity(self.database)
        self.sources = self.database["sources"]
        self.knowledge_sha256 = self.identity["sha256"]
        # A compatibility view, not a second knowledge base or a file fallback.
        self.kb = {"version": self.identity["version"], "policy": self.database["policy"],
                   "diseases": self.database["diseases"],
                   "uncertain_action_ids": self.database["uncertain_action_ids"],
                   "actions": {key: value for key, value in self.database["treatments"].items()
                               if value["treatment_type"] == "care"}}
        self.medications = MedicationSelector(self.database)

    def _validate_knowledge(self) -> None:
        validate_knowledge(self.database)

    def run(self, diagnosis: dict, environment: dict, farm: dict,
            request: dict | None = None) -> dict:
        self.begin()
        self.state["feedback_runs"] = self.state.get("feedback_runs", 0) + bool(request)
        decisions = []
        messages = []
        missing = []
        trace = []
        farm = farm if isinstance(farm, dict) else {}
        diag = diagnosis.get("data") or {}
        env = environment.get("data") or {}
        setup = {}
        allowed_setup = {"environment": ENVIRONMENTS, "root_system": ROOT_SYSTEMS, **CONTEXT_VALUES}
        for field, allowed in allowed_setup.items():
            value = env.get(field, farm.get(field))
            setup[field] = value if isinstance(value, str) and value in allowed else "unknown"
        origins = env.get("input_provenance", farm.get("input_provenance"))
        origins = origins if isinstance(origins, dict) else {}
        provenance = {field: (origins.get(field) if isinstance(origins.get(field), str) and origins.get(field) in {"ui_default", "user_selected"}
                              and value != "unknown" else "unknown_origin") for field, value in setup.items()}
        assumed_context = [field for field, origin in provenance.items() if origin == "ui_default"]
        common = {"knowledge_version": self.identity["version"], "knowledge_sha256": self.knowledge_sha256,
                  "knowledge_base": deepcopy(self.identity), "knowledge_counts": deepcopy(self.identity["counts"]),
                  "farm_setup": setup, "input_provenance": provenance, "assumed_context": assumed_context,
                  "evidence_scope": self.identity["scope"], "growth_stage": setup["growth_stage"],
                  "irrigation": setup["irrigation"]}
        prediction = diag.get("prediction")
        if isinstance(prediction, dict):
            prediction = prediction.get("class_name") or prediction.get("label")
        if diagnosis.get("status") not in {"ok", "partial"} or prediction not in self.kb["diseases"]:
            medication = self.medications.select(None, growth_stage=setup["growth_stage"], input_provenance=provenance)
            return self.result(
                {**common, "actions": [], "disease_label": None,
                 "medication": medication,
                 "summary": "A usable leaf diagnosis is needed before recommendations can be selected.",
                 "missing_context": ["usable diagnosis"], "source_ids": [],
                 "root_system": setup["root_system"], "environment": setup["environment"],
                 "rule_trace": [], "excluded_actions": [], "deferred_actions": [], "deferred_action_ids": [],
                 "triggered_rule_ids": ["R01", "R13"]}, status="blocked",
                decisions=[decision("withhold_actions", "Diagnosis is missing, blocked or outside the model class list.", policy_id="R01"),
                           decision("select_medication", medication["summary"], policy_id="R13", medication_status=medication["status"], option_ids=[])],
                messages=[{"to": "diagnosis", "type": "missing_result", "request": "usable_diagnosis"}])

        entry = self.kb["diseases"][prediction]
        # Keep missing setup unknown.
        root_system = setup["root_system"]
        growing_environment = setup["environment"]
        if root_system not in {"soil", "substrate", "hydroponic"}:
            root_system = "unknown"
            missing.append("growing root system")
        if growing_environment not in {"outdoor", "greenhouse", "tunnel", "indoor"}:
            growing_environment = "unknown"
            missing.append("growing environment")
        uncertainty_known = isinstance(diag.get("uncertainty"), bool)
        uncertain = diag.get("uncertainty") is True
        # Do not expand uncertain or broad diagnoses.
        restricted_scope = diag.get("recommendation_scope") != "disease_specific" or entry["broad_label"]
        limited = uncertain or not uncertainty_known or restricted_scope
        if not uncertainty_known:
            missing.append("diagnosis uncertainty assessment")
        if diag.get("recommendation_scope") not in {"general_only", "disease_specific"}:
            missing.append("diagnosis recommendation scope")
        # Broad labels do not identify a pathogen.
        if uncertain:
            selected = list(self.kb["uncertain_action_ids"])
            selection_action = "general_care"
            selection_reason = "Uncertain diagnosis: use general care and a follow-up image."
        elif not uncertainty_known:
            selected = ["routine_monitor"]
            selection_action = "general_care"
            selection_reason = "The diagnosis confidence assessment is missing; use general monitoring until the handoff is complete."
        elif restricted_scope and entry["broad_label"]:
            selected = list(entry["action_ids"])
            selection_action = "select_category_care"
            selection_reason = "The source class is broad; use curated category-level care without assigning a pathogen."
        elif restricted_scope:
            selected = ["routine_monitor"]
            selection_action = "general_care"
            selection_reason = "The diagnosis policy limits recommendations to general care."
        else:
            selected = list(entry["action_ids"])
            selection_action = "select_class_actions"
            selection_reason = f"Select the curated actions for {prediction}."
        decisions.append(decision(selection_action, selection_reason,
                                  policy_id="R02", disease_label=prediction))
        action_rules = {aid: ["R02"] for aid in selected}
        action_reasons = {aid: [selection_reason] for aid in selected}

        def add_actions(ids, rule_id, reason):
            for aid in ids:
                selected.append(aid)
                action_rules.setdefault(aid, []).append(rule_id)
                action_reasons.setdefault(aid, []).append(reason)

        # Record gated candidates without promoting them.
        for aid in entry["action_ids"]:
            if aid not in selected:
                trace.append({"status": "excluded", "item_id": aid, "item_type": "care",
                              "reason": selection_reason, "rule_ids": ["R02"],
                              "source_ids": list(self.kb["actions"][aid]["source_ids"])})
        if entry["broad_label"]:
            decisions.append(decision("preserve_broad_label", "No pathogen or specific vector is inferred from this inherited dataset label.", policy_id="R03"))
            for aid in selected:
                action_rules[aid].append("R03")
        if root_system == "hydroponic":
            add_actions(["hydroponic_flow", "hydroponic_measure"], "R04", "The supplied root system is hydroponic.")
            decisions.append(decision("hydroponic_context", "Use nutrient-solution care; omit soil-bed actions.", policy_id="R04"))
        elif root_system in {"soil", "substrate"}:
            add_actions(["steady_moisture"], "R02", "General moisture care applies to the supplied soil or substrate setup.")

        observations = env.get("observations") or farm.get("observations") or {}
        wetness = observations.get("root_zone", "unknown")
        if root_system in {"soil", "substrate"} and wetness in {"wet", "dry"}:
            add_actions(["check_drainage" if wetness == "wet" else "check_dry_root"], "R05", f"Farmer reported a {wetness} root zone.")
            decisions.append(decision("respond_to_root_observation", f"Farmer reported the root zone as {wetness}.", policy_id="R05", observation=wetness))
        elif wetness == "unknown" and root_system in {"soil", "substrate"}:
            missing.append("farmer_root_zone_observation")

        measurements = env.get("measurements") or {}
        # Use validated measurements only.
        rh = measurements.get("relative_humidity") or {}
        rh_value = _number(rh)
        fresh_rh = rh.get("freshness") == "fresh" and rh.get("source") in {"farmer", "farmer_input", "sensor"}
        protected = growing_environment in {"greenhouse", "tunnel", "indoor"}
        raised_humidity_priority = False
        if protected:
            add_actions(["protected_airflow"], "R06", "The supplied growing environment is protected; inspect airflow and condensation.")
            if fresh_rh and rh_value is not None and 0 <= rh_value <= 100:
                raised_humidity_priority = rh_value >= self.kb["policy"]["protected_humidity_attention_percent"]
                if raised_humidity_priority:
                    decisions.append(decision("prioritise_airflow_check", f"Recent protected-area humidity measurement is {rh_value:g}%.", policy_id="R06", threshold_type="operational_attention_only"))
            else:
                missing.append("recent humidity measurement inside the growing area")
                decisions.append(decision("avoid_outdoor_proxy", "Outdoor humidity is not treated as a greenhouse or indoor measurement.", policy_id="R07"))
                action_rules["protected_airflow"].append("R07")

        weather = env.get("weather") or {}
        weather_usable = weather.get("status") in {"live", "cached"}
        current = weather.get("current") or {}
        forecast_rows = weather.get("forecast") or []
        forecast = forecast_rows[0] if isinstance(forecast_rows, list) and forecast_rows else forecast_rows
        forecast = forecast if isinstance(forecast, dict) else {}
        rain_values = [current.get("precipitation_mm"), current.get("rain_mm"),
                       current.get("precipitation"), current.get("rain"), forecast.get("precipitation_sum_mm"),
                       forecast.get("rain_24h_mm"), forecast.get("precipitation_24h_mm")]
        if isinstance(forecast_rows, list):
            rain_values.extend(row.get("rain") for row in forecast_rows[:24] if isinstance(row, dict))
        rain_present = weather_usable and any((_number(v) or 0) > 0 for v in rain_values)
        exposed_rain = growing_environment == "outdoor" and rain_present
        if exposed_rain and root_system in {"soil", "substrate"}:
            add_actions(["rain_irrigation_check"], "R08", "Fresh outdoor weather indicates rain; check the actual root zone before irrigation.")
            decisions.append(decision("schedule_moisture_check", "Fresh outdoor weather indicates precipitation; actual bed moisture is still checked separately.", policy_id="R08"))
        if growing_environment == "outdoor" and not weather_usable:
            missing.append("recent weather")

        ph = measurements.get("ph") or {}
        if (root_system == "soil" and ph.get("source") in {"farmer", "farmer_input", "sensor"}
                and ph.get("freshness") == "fresh" and _number(ph) is not None):
            add_actions(["soil_test_plan"], "R09", "A current vetted soil pH measurement is available for nutrient planning.")
            decisions.append(decision("use_ph_as_background", "Keep farmer pH for nutrient planning without deriving a dose or disease cause.", policy_id="R09"))
        if root_system == "hydroponic":
            for field in ("ph", "ec", "root_temperature"):
                value = measurements.get(field) or {}
                if _number(value) is None or value.get("freshness") != "fresh":
                    missing.append(f"recent nutrient-solution {field.replace('_', ' ')}")

        context_priorities = {}
        for rid, rule in sorted(self.database["rules"].items()):
            if rule["handler"] != "context_action":
                continue
            if all(setup[clause["field"]] == clause["equals"] for clause in rule["when"]):
                assumed = [clause["field"] for clause in rule["when"] if provenance[clause["field"]] == "ui_default"]
                reason = rule["description"] + (" This uses a preset scenario assumption: " + ", ".join(assumed) + "." if assumed else "")
                add_actions(rule["add_action_ids"], rid, reason)
                for aid, priority in rule.get("priority_overrides", {}).items():
                    if aid in selected:
                        context_priorities[aid] = priority
                        action_rules.setdefault(aid, []).append(rid)
                        action_reasons.setdefault(aid, []).append(reason)
                decisions.append(decision(rule["name"], reason, policy_id=rid,
                                          inputs={clause["field"]: setup[clause["field"]] for clause in rule["when"]},
                                          input_provenance={clause["field"]: provenance[clause["field"]] for clause in rule["when"]},
                                          source_ids=rule["source_ids"]))

        excluded = set((request or {}).get("exclude_ids") or [])
        if request:
            decisions.append(decision("apply_report_feedback", (request.get("reason") or "Reporting requested removal of incompatible actions.")[:300], policy_id="R10", exclude_ids=sorted(str(x) for x in excluded)))
            messages.append({"to": "reporting", "type": "revision_complete", "excluded_ids": sorted(str(x) for x in excluded)})

        actions = []
        for aid in dict.fromkeys(selected):
            action = deepcopy(self.kb["actions"][aid])
            if aid in excluded:
                trace.append({"status": "excluded", "item_id": aid, "item_type": "care",
                              "reason": "Reporting feedback excluded this action.",
                              "rule_ids": list(dict.fromkeys(action_rules.get(aid, []) + ["R10"])),
                              "source_ids": list(action["source_ids"])})
                continue
            roots = action.get("root_systems")
            setups = action.get("environments")
            if (roots and root_system not in roots) or (setups and growing_environment not in setups):
                decisions.append(decision("omit_incompatible_action", f"{aid} does not match the selected growing setup.", policy_id="R11", action_id=aid))
                trace.append({"status": "excluded", "item_id": aid, "item_type": "care",
                              "reason": "The action does not cover the supplied root system or growing environment.",
                              "rule_ids": list(dict.fromkeys(action_rules.get(aid, []) + ["R11"])),
                              "source_ids": list(action["source_ids"])})
                continue
            action["id"] = aid
            action["base_priority"] = action["priority"]
            action["requires_specific_diagnosis"] = not limited and not entry["broad_label"] and prediction != "Healthy" and aid in entry["action_ids"]
            if aid == "protected_airflow" and raised_humidity_priority:
                action["priority"] = 90
                action["reason"] += f" Your recent inside humidity measurement is {rh_value:g}%."
                action_reasons[aid].append(f"A fresh measured inside humidity of {rh_value:g}% reaches the operational attention threshold.")
            if aid == "dry_foliage" and exposed_rain:
                action["priority"] = 90
                action["reason"] += " Recent outdoor weather indicates rain, so avoiding additional splash takes priority."
                action_rules.setdefault(aid, []).append("R08")
                action_reasons.setdefault(aid, []).append("Fresh outdoor rain increases the attention priority for limiting additional splash.")
            if aid in context_priorities:
                action["priority"] = context_priorities[aid]
            action["constraints"] = {"root_systems": roots or ["soil", "substrate", "hydroponic"],
                                     "environments": setups or ["outdoor", "greenhouse", "tunnel", "indoor"]}
            action["sources"] = source_records(self.database, action["source_ids"])
            action["triggered_rule_ids"] = list(dict.fromkeys(action_rules.get(aid, []) + ["R11", "R12"]))
            action["selection_reason"] = " ".join(dict.fromkeys(action_reasons.get(aid, [])))
            action["knowledge_version"], action["knowledge_sha256"] = self.identity["version"], self.knowledge_sha256
            action["input_provenance"] = deepcopy(provenance)
            action["assumed_context"] = list(assumed_context)
            actions.append(action)
        actions.sort(key=lambda a: (-a["priority"], a["id"]))
        limit = self.kb["policy"]["maximum_actions"]
        deferred_ids = [a["id"] for a in actions[limit:]]
        for action in actions[limit:]:
            trace.append({"status": "deferred", "item_id": action["id"], "item_type": "care",
                          "reason": "Eligible action deferred by the maximum action count after priority ordering; the score is not treatment efficacy.",
                          "rule_ids": action["triggered_rule_ids"], "source_ids": list(action["source_ids"])})
        actions = actions[:limit]
        for rank, action in enumerate(actions, 1):
            action["priority_score"] = action["priority"]
            action["priority"] = rank
            trace.append({"status": "selected", "item_id": action["id"], "item_type": "care",
                          "reason": action["selection_reason"], "rule_ids": action["triggered_rule_ids"],
                          "source_ids": list(action["source_ids"])})
        source_ids = sorted({sid for action in actions for sid in action["source_ids"]})
        medication = self.medications.select(prediction, uncertain=uncertain or not uncertainty_known,
                                            restricted_scope=restricted_scope,
                                            broad_label=entry["broad_label"],
                                            root_system=root_system, environment=growing_environment,
                                            growth_stage=setup["growth_stage"], input_provenance=provenance,
                                            exclude_ids=excluded)
        trace.extend(medication["selection_trace"])
        source_ids = sorted(set(source_ids + medication["source_ids"]))
        decisions.append(decision("select_medication", medication["summary"], policy_id="R13",
                                  medication_status=medication["status"],
                                  option_ids=[item["id"] for item in medication["options"]],
                                  medication_knowledge_version=(self.medications.kb or {}).get("version"),
                                  evidence_scope="International lettuce references; product authorization at the user location is not assessed"))
        decisions.append(decision("rank_actions", "Order by transparent action priorities and context; these are not efficacy scores.", policy_id="R12", action_ids=[a["id"] for a in actions], deferred_ids=deferred_ids))
        if uncertain:
            summary = "Start with general crop care and a clearer image before choosing disease-specific actions."
        elif not uncertainty_known:
            summary = "Continue general crop care; the diagnosis confidence assessment was not supplied."
        elif restricted_scope and not entry["broad_label"]:
            summary = "The current diagnosis policy supports general crop care and continued observation."
        else:
            summary = entry["summary"]
        # Keep policy and agronomic evidence distinct.
        for item in decisions:
            rule = self.database["rules"][item["policy_id"]]
            item.setdefault("source_ids", list(rule["source_ids"]))
            item["rule_kind"] = rule["kind"]
            item["knowledge_version"], item["knowledge_sha256"] = self.identity["version"], self.knowledge_sha256
        triggered = sorted({rid for row in trace for rid in row["rule_ids"]} |
                           {item["policy_id"] for item in decisions})
        # Record each final selection once.
        selected_trace_ids = {(row["item_type"], row["item_id"]) for row in trace if row["status"] in {"selected", "deferred"}}
        trace = [row for row in trace if row["status"] != "excluded" or
                 (row["item_type"], row["item_id"]) not in selected_trace_ids]
        return self.result({**common, "actions": actions, "disease_label": prediction,
                            "medication": medication,
                            "display_name": entry["display_name"], "source_label_broad": entry["broad_label"],
                            "summary": summary, "recommendation_scope": "general_only" if limited else "disease_specific",
                            "missing_context": sorted(set(missing)), "source_ids": source_ids,
                            "root_system": root_system, "environment": growing_environment,
                            "triggered_rule_ids": triggered, "rule_trace": trace,
                            "excluded_actions": [row for row in trace if row["item_type"] == "care" and row["status"] == "excluded"],
                            "deferred_actions": [row for row in trace if row["item_type"] == "care" and row["status"] == "deferred"],
                            "deferred_action_ids": deferred_ids},
                           status="partial" if limited or missing else "ok", decisions=decisions, messages=messages)
