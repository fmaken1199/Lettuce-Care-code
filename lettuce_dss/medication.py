'Select sourced international treatment references from the unified database.'
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from lettuce_dss.core import CLASS_NAMES
from lettuce_dss.knowledge import validate_knowledge, knowledge_identity, source_records, parse_knowledge


def medication_result(status: str, summary: str, options=None, source_ids=None) -> dict:
    return {"status": status, "summary": summary, "options": options or [],
            "source_ids": sorted(set(source_ids or []))}


class MedicationSelector:
    """Finite diagnosis/setup filters; all evidence comes from one validated KB."""

    def __init__(self, database: dict | Path, sources: dict | None = None):
        if isinstance(database, (str, Path)):
            self.raw_bytes = Path(database).read_bytes()
            database = parse_knowledge(self.raw_bytes)
        else:
            validate_knowledge(database)
            self.raw_bytes = json.dumps(database, sort_keys=True, allow_nan=False).encode()
        self.database = deepcopy(database)
        self.sources = self.database["sources"]
        if sources is not None and sources != self.sources:
            raise ValueError("Medication sources must come from the same unified knowledge base")
        self.identity = knowledge_identity(self.database)
        # Compatibility view only. No second file is loaded or maintained.
        self.kb = {"version": self.identity["version"],
                   "maximum_options": self.database["policy"]["maximum_medication_options"],
                   "diseases": {key: value["medication"] for key, value in self.database["diseases"].items()},
                   "options": {key: value for key, value in self.database["treatments"].items()
                               if value["treatment_type"] in {"chemical", "biological"}}}

    def select(self, label: str | None, *, uncertain=True, restricted_scope=True,
               broad_label=False, root_system="unknown", environment="unknown",
               growth_stage="unknown", input_provenance=None, exclude_ids=None) -> dict:
        provenance = deepcopy(input_provenance or {})
        entry = self.kb["diseases"].get(label)
        candidate_ids = list(entry["option_ids"]) if entry else []
        trace = []
        excluded = set(exclude_ids or [])
        rule_ids = ["R13"]
        near_harvest = growth_stage == "near_harvest"

        def record(oid, status, reason, ids):
            trace.append({"status": status, "item_id": oid, "item_type": "medication",
                          "reason": reason, "rule_ids": list(dict.fromkeys(ids)),
                          "source_ids": list(self.kb["options"][oid]["source_ids"])})

        def finish(status, summary, options=None, source_ids=None, rejection=None, guard_ids=None):
            if rejection:
                for oid in candidate_ids:
                    record(oid, "excluded", rejection, guard_ids or ["R13"])
            result = medication_result(status, summary, options, source_ids)
            all_rules = list(dict.fromkeys(rule_ids + (guard_ids or []) +
                                           [rid for row in trace for rid in row["rule_ids"]]))
            result.update(knowledge_version=self.identity["version"], knowledge_sha256=self.identity["sha256"],
                          knowledge_base=deepcopy(self.identity), selection_trace=trace,
                          triggered_rule_ids=all_rules,
                          input_provenance=provenance, growth_stage=growth_stage,
                          harvest_review_required=near_harvest and bool(options),
                          sources=source_records(self.database, result["source_ids"]))
            return result

        if label not in CLASS_NAMES:
            return finish("not_selected", "No medication selected without a usable lettuce diagnosis.", guard_ids=["R01", "R13"])
        if uncertain is not False:
            reason = "No medication selected because the diagnosis is uncertain or its confidence assessment is missing."
            return finish("not_selected", reason, rejection=reason, guard_ids=["R02", "R13"])
        # Keep explicit no-treatment outcomes for viral labels.
        if entry["status"] in {"not_needed", "no_curative"}:
            return finish(entry["status"], entry["summary"], source_ids=entry["source_ids"])
        if broad_label or self.database["diseases"][label]["broad_label"]:
            reason = "The disease label is too broad to select a targeted treatment reference."
            return finish("not_selected", reason, rejection=reason, guard_ids=["R03", "R13"])
        if restricted_scope is not False:
            reason = "The current diagnosis supports general crop care only."
            return finish("not_selected", reason, rejection=reason, guard_ids=["R02", "R13"])
        if entry["status"] != "options":
            return finish(entry["status"], entry["summary"], source_ids=entry["source_ids"])

        options = []
        for oid in candidate_ids:
            option = self.kb["options"][oid]
            if oid in excluded:
                record(oid, "excluded", "Reporting feedback excluded this treatment reference.", ["R10", "R13"])
                continue
            if root_system not in option["root_systems"] or environment not in option["environments"]:
                record(oid, "excluded", "The reference does not cover the supplied root system and growing environment.", ["R11", "R13"])
                continue
            if len(options) >= self.kb["maximum_options"]:
                record(oid, "deferred", "Additional eligible reference deferred by the two-option display limit; this is not an efficacy ranking.", ["R12", "R13"])
                continue
            item = deepcopy(option)
            item["id"] = oid
            item["diseases"] = [name for name, disease in self.kb["diseases"].items() if oid in disease["option_ids"]]
            item["triggered_rule_ids"] = ["R13", "R11", "R12"]
            item["selection_reason"] = "The predicted disease and supplied growing setup match this international reference."
            item["source_urls"] = [self.sources[sid]["url"] for sid in item["source_ids"]]
            item["sources"] = source_records(self.database, item["source_ids"])
            item["knowledge_version"], item["knowledge_sha256"] = self.identity["version"], self.identity["sha256"]
            item["input_provenance"] = deepcopy(provenance)
            item["assumed_context"] = [field for field in ("environment", "root_system", "growth_stage")
                                       if provenance.get(field) == "ui_default"]
            item["harvest_review_required"] = near_harvest
            if near_harvest:
                item["application_note"] += " Near harvest: check the actual product label, crop authorization and preharvest interval before considering use; this reference provides no application interval."
                if "R15" in self.database["rules"]:
                    item["triggered_rule_ids"].append("R15")
                    rule_ids.append("R15")
            options.append(item)
            record(oid, "selected", item["selection_reason"], item["triggered_rule_ids"])
        if not options:
            return finish("not_selected", "No suitable treatment option is listed for this growing setup. Follow the crop-care steps.", source_ids=entry["source_ids"])
        summary = entry["summary"]
        if near_harvest:
            summary += " Harvest is near: check the label's waiting period before applying any product."
        return finish("options", summary, options,
                      entry["source_ids"] + [sid for item in options for sid in item["source_ids"]])
