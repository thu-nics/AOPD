"""Deterministic teacher-only reference guidance; never execute reference steps."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema

ANSWER_PROTOCOL = "tau-reference-guidance-v1"

# A toggle has no target boolean. Never infer one from the task ID, or direct
# a second toggle after the customer has already fixed the setting.
USER_ACTIONS = {
    "toggle_airplane_mode": "Ask the customer to check airplane mode and adjust it as needed to restore connectivity.",
    "toggle_roaming": "Ask the customer to check data roaming and adjust it as needed for their location and plan.",
    "toggle_data": "Ask the customer to check mobile data and adjust the setting as needed to restore connectivity.",
    "toggle_data_saver_mode": "Ask the customer to check data saver and adjust it if it restricts the required service.",
    "toggle_wifi_calling": "Ask the customer to check Wi-Fi calling and adjust it according to the troubleshooting policy.",
    "disconnect_vpn": "Ask the customer to disconnect their VPN if it is still connected and causing the problem.",
    "reseat_sim_card": "Guide the customer to reseat their SIM card if this repair has not already succeeded.",
    "reset_apn_settings": "Guide the customer to restore the default access point settings if they are still incorrect.",
    "reboot_device": "Ask the customer to restart their phone if this step is still needed.",
    "make_payment": "After a payment request is received and accepted, ask the customer to complete the payment themselves.",
}


def customer_step(name, arguments):
    if name in USER_ACTIONS and not arguments:
        return USER_ACTIONS[name]
    if name == "set_network_mode_preference" and set(arguments) == {"mode"} and arguments["mode"] == "4g_5g_preferred":
        return "Ask the customer to prefer 4G/5G networks if that preference is not already set."
    if name == "grant_app_permission" and set(arguments) == {"app_name", "permission"}:
        if arguments["app_name"] == "messaging" and arguments["permission"] in {"storage", "sms"}:
            permission = "storage" if arguments["permission"] == "storage" else "SMS"
            return f"Ask the customer to grant the messaging app {permission} permission if it is missing."
    raise ValueError(f"unmapped customer reference action: {name} {arguments}")


def target_state(assertion):
    name, args = assertion["func_name"], assertion.get("arguments") or {}
    shapes = {
        "assert_mobile_data_status": ("user", {"expected_status"}),
        "assert_internet_speed": ("user", {"expected_speed", "expected_desc"}),
        "assert_data_refueling_amount": ("assistant", {"customer_id", "line_id", "expected_amount"}),
        "assert_service_status": ("user", {"expected_status"}),
        "assert_no_overdue_bill": ("assistant", {"overdue_bill_id"}),
        "assert_can_send_mms": ("user", {"expected_status"}),
    }
    if name not in shapes or (assertion.get("env_type"), set(args)) != shapes[name] or assertion.get("assert_value") is not True:
        raise ValueError(f"unmapped reference target: {assertion}")
    if name in {"assert_mobile_data_status", "assert_can_send_mms"} and not isinstance(args["expected_status"], bool):
        raise ValueError(f"reference target requires a boolean: {assertion}")
    if name == "assert_mobile_data_status":
        return "Mobile data should work." if args["expected_status"] else "Mobile data is expected to remain unavailable; do not claim it works."
    if name == "assert_can_send_mms":
        return "The customer should be able to send MMS messages." if args["expected_status"] else "MMS sending is expected to remain unavailable; do not claim it works."
    if name == "assert_internet_speed":
        return f"Internet speed should be at least {args['expected_speed']} Mbps with quality described as {args['expected_desc']}."
    if name == "assert_data_refueling_amount":
        return f"Customer {args['customer_id']}, line {args['line_id']}: total refueled data should be {args['expected_amount']} GB; this is a total, not an additional amount on every turn."
    if name == "assert_service_status":
        if args["expected_status"] == "connected":
            return "The customer's phone should have network service."
        if args["expected_status"] == "no_service":
            return "Network service is expected to remain unavailable; do not claim it was restored."
        raise ValueError(f"unmapped reference service status: {args['expected_status']}")
    return f"Bill {args['overdue_bill_id']} should be settled and no longer outstanding."


def reference_guidance(task, tools):
    """Allowlist answer fields; missing answers are explicit, never invented."""
    criteria = task.get("evaluation_criteria") or {}
    schemas = {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools}
    steps = []
    for action in criteria.get("actions") or []:
        role, name, args = action.get("requestor", "assistant"), action["name"], action.get("arguments") or {}
        if role == "assistant":
            if name not in schemas:
                raise ValueError(f"reference assistant tool is unavailable: {name}")
            jsonschema.validate(args, schemas[name])
            steps.append({"actor": "assistant", "tool": name, "arguments": copy.deepcopy(args)})
        elif role == "user":
            steps.append({"actor": "customer", "instruction": customer_step(name, args)})
        else:
            raise ValueError(f"unknown reference actor: {role}")
    targets = [target_state(a) for a in criteria.get("env_assertions") or []]
    targets.extend(criteria.get("nl_assertions") or [])
    communicate = criteria.get("communicate_info") or []
    return {
        "available": bool(steps or targets or communicate),
        "reference_steps": steps,
        "target_outcomes": copy.deepcopy(targets),
        "information_to_communicate": copy.deepcopy(communicate),
    }


def reference_source_fingerprint(source_root):
    """Pin official train answers only; test answers do not affect identity."""
    root = Path(source_root) / "data/tau2/domains"
    records = {}
    for domain in ("airline", "retail", "telecom"):
        ids = set(map(str, json.loads((root / domain / "split_tasks.json").read_text())["train"]))
        rows = json.loads((root / domain / "tasks.json").read_text())
        records[domain] = {str(row["id"]): row.get("evaluation_criteria") for row in rows if str(row["id"]) in ids}
        if set(records[domain]) != ids:
            raise ValueError(f"missing or inconsistent official train reference IDs: {domain}")
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
