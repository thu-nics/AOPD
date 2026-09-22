"""Offline, source-grounded customer briefs; never an online action teacher.

The API sees user_scenario only. Drafts are unusable until individually reviewed
and frozen. Frozen content and source hashes are part of self-teacher identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

BRIEF_PROTOCOL = "tau-customer-brief-v1"
CATEGORIES = {"goal", "constraint", "preference", "known", "unknown"}
PROMPT = """Extract a compact factual customer brief from this user_scenario.
Return JSON {"facts":[{"category":"goal|constraint|preference|known|unknown",
"text":"third-person factual English statement","source_field":"instructions.FIELD",
"quote":"exact supporting substring from that field"}]}.
Capture ALL goals, limits, conditions, alternatives, dates, amounts, quantities,
negations, what the customer knows/can provide and does not know. Keep conflicts
as facts; do not reconcile them. Preserve conditional fallbacks and disclosure
restrictions (e.g. customer knows an answer but initially withholds it).
Use third person, not roleplay commands. Exclude persona/emotion, simulator/tool
instructions and demands to coerce the assistant or violate policy. Preserve a
substantive requested outcome even if originally expressed coercively.
Do not recommend actions, decide policy eligibility, invent facts, resolve the
task or include user-side tool/API names. Do not include hidden evaluation data.
Each fact needs a verbatim source quote; use only this scenario as evidence.
Include exact known identifiers in the offline brief: runtime redacts them.
Only output the requested JSON, without commentary."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def official_tasks(source_root):
    root = Path(source_root) / "data/tau2/domains"
    for domain in ("airline", "retail", "telecom"):
        ids = set(json.loads((root / domain / "split_tasks.json").read_text())["train"])
        for task in json.loads((root / domain / "tasks.json").read_text()):
            if task["id"] in ids:
                yield domain, task


def validate_facts(scenario, facts):
    if not isinstance(facts, list) or not facts:
        raise ValueError("empty customer brief")
    for fact in facts:
        if set(fact) != {"category", "text", "source_field", "quote"} or fact["category"] not in CATEGORIES:
            raise ValueError("invalid customer fact schema")
        if not all(isinstance(fact[k], str) and fact[k].strip() for k in ("text", "source_field", "quote")):
            raise ValueError("empty fact text/evidence")
        value = scenario
        for part in fact["source_field"].split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if not isinstance(value, str) or fact["quote"] not in value:
            raise ValueError("fact quote is not an exact source excerpt")


class BriefStore:
    def __init__(self, path):
        if not path:
            raise ValueError("self privileged context requires a reviewed frozen customer brief file")
        self.path = str(Path(path).resolve())
        bundle = json.loads(Path(path).read_text())
        if bundle.get("protocol") != BRIEF_PROTOCOL or bundle.get("status") != "frozen":
            raise ValueError("customer briefs are not reviewed/frozen")
        records = bundle["records"]
        self.records = {(r["domain"], str(r["task_id"])): r for r in records}
        if len(self.records) != len(records):
            raise ValueError("duplicate customer briefs")
        for r in records:
            validate_facts(r["user_scenario"], r["facts"])
            if r.get("review_status") != "approved" or r["scenario_sha256"] != digest(r["user_scenario"]):
                raise ValueError("unreviewed or source-mismatched customer brief")
        self.fingerprint = digest(bundle)

    def get(self, domain, task):
        r = self.records[(domain, str(task["id"]))]
        if r["scenario_sha256"] != digest(task["user_scenario"]):
            raise ValueError(f"customer brief source drift: {domain}:{task['id']}")
        # A copy avoids accidental contamination of the frozen in-memory store.
        return json.loads(json.dumps(r["facts"]))


@lru_cache(maxsize=8)
def load_briefs(path):
    return BriefStore(path)


def main():
    import httpx

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(protocol=BRIEF_PROTOCOL, prompt_sha256=digest(PROMPT), model=args.model, api_base=args.api_base, temperature=0, thinking={"type": "disabled"}, max_tokens=8192)
    manifest = out / "generation_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError("brief generation protocol mismatch")
    manifest.write_text(json.dumps(settings, indent=2) + "\n")
    key = os.environ["DEEPSEEK_API_KEY"]
    client = httpx.Client(timeout=300, trust_env=False)

    def generate(entry):
        domain, task = entry
        path = out / (domain + "-" + digest(task["id"])[:16] + ".json")
        if path.exists():
            old = json.loads(path.read_text())
            if old["scenario_sha256"] != digest(task["user_scenario"]):
                raise ValueError("source drift in draft")
            return
        response = client.post(
            args.api_base.rstrip("/") + "/chat/completions",
            headers={"Authorization": "Bearer " + key},
            json=dict(model=args.model, temperature=0, thinking={"type": "disabled"}, max_tokens=8192, response_format={"type": "json_object"}, messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": json.dumps(task["user_scenario"], ensure_ascii=False)}]),
        )
        response.raise_for_status()
        data = response.json()
        if data["choices"][0]["finish_reason"] != "stop":
            raise ValueError("truncated customer brief")
        (out / (path.stem + ".response.json")).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        facts = json.loads(data["choices"][0]["message"]["content"])["facts"]
        validation_error = None
        try:
            validate_facts(task["user_scenario"], facts)
        except ValueError as exc:
            validation_error = str(exc)
        record = dict(domain=domain, task_id=str(task["id"]), user_scenario=task["user_scenario"], scenario_sha256=digest(task["user_scenario"]), facts=facts, review_status="pending", validation_error=validation_error, usage=data.get("usage"), response_model=data.get("model"))
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        print("DRAFT", domain, task["id"], flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(generate, official_tasks(args.source_root)))


if __name__ == "__main__":
    main()
