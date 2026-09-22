"""Teacher-only customer information, optional live facts or reference guidance."""

from __future__ import annotations

import json

from .customer_briefs import load_briefs
from .self_teacher_context import _identifier_pattern

STATE_PROTOCOL = "tau-customer-live-state-v1"
MODES = ("customer", "customer_and_state", "answer_conditioned")


def plain(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def fields(value, names):
    value = plain(value)
    return {k: value[k] for k in names.split() if k in value and value[k] is not None}


def records(value):
    value = plain(value)
    return list(value.values()) if isinstance(value, dict) else list(value)


def _contains(value, texts):
    return bool(value) and any(_identifier_pattern(str(value)).search(t) for t in texts)


def _public_text(chat):
    return [json.dumps(m.get("content"), ensure_ascii=False) for m in chat if m.get("role") in {"user", "tool"} and m.get("content") is not None]


def _name(user):
    name = user.get("name", user.get("full_name", ""))
    if isinstance(name, dict):
        return " ".join(str(name[k]) for k in ("first_name", "last_name") if name.get(k))
    return name


def bind_customer(domain, scenario, db, user_db=None):
    # No task ID, initial-state assertions or evaluation criteria are available.
    # Exact identifiers in customer instructions take precedence over names.
    raw = [json.dumps(scenario.get("instructions") or {}, ensure_ascii=False)]
    users = records(db["customers" if domain == "telecom" else "users"])
    if domain == "telecom":
        phone = (user_db or {}).get("surroundings", {}).get("phone_number")
        lines = [line for line in db["lines"] if line["phone_number"] == phone]
        if len(lines) != 1:
            raise ValueError("ambiguous current customer device/line binding")
        matches = [u for u in users if lines[0]["line_id"] in u["line_ids"]]
    else:
        matches = [u for u in users if _contains(u.get("user_id"), raw)]
        if not matches:
            matches = [u for u in users if any(_contains(u.get(k), raw) for k in ("email", "phone_number"))]
        if not matches:
            matches = [u for u in users if _contains(_name(u), raw)]
        if not matches:
            collection = db["reservations" if domain == "airline" else "orders"]
            owner_ids = {v["user_id"] for k, v in collection.items() if _contains(k, raw)}
            matches = [u for u in users if u["user_id"] in owner_ids]
    if len(matches) > 1 and domain != "telecom":
        narrowed = [u for u in matches if any(_contains(u.get("address", {}).get(k), raw) for k in ("zip", "zip_code"))]
        if narrowed:
            matches = narrowed
    if len(matches) != 1:
        raise ValueError(f"{domain}: cannot uniquely bind customer from scenario and DB relationships ({len(matches)})")
    return matches[0]


def live_customer_state(domain, scenario, environment):
    """Read allowlisted fields from the actual paused orchestrator, never reset/sync."""
    db = plain(environment.tools.db)
    user_tools = getattr(environment, "user_tools", None)
    user_db = plain(user_tools.db) if user_tools is not None else None
    customer = bind_customer(domain, scenario, db, user_db)
    secrets = set()
    for key in ("user_id", "customer_id", "email", "phone_number"):
        if customer.get(key):
            secrets.add(str(customer[key]))
    if _name(customer):
        secrets.add(_name(customer))
    result = {"customer": {}}
    if domain in {"airline", "retail"}:
        result["customer"] = fields(customer, "membership")
        collection_name = "reservations" if domain == "airline" else "orders"
        owned = {k: v for k, v in db[collection_name].items() if v["user_id"] == customer["user_id"]}

        # If no order/reservation is specified, all this customer's objects can
        # be relevant (e.g. return every non-gaming item); never other customers.
        selected = sorted(owned)
        result[collection_name] = []
        for key in selected:
            obj = owned[key]
            secrets.add(key)
            if domain == "airline":
                row = fields(obj, "reservation_id status cabin insurance created_at origin destination flight_type total_baggages nonfree_baggages")
                row["passenger_count"] = len(obj["passengers"])
                row["customer_is_passenger"] = any(" ".join((p["first_name"], p["last_name"])).casefold() == _name(customer).casefold() for p in obj["passengers"])
                row["flights"] = []
                for f in obj["flights"]:
                    entry = fields(f, "flight_number origin destination date price")
                    flight = db["flights"].get(f["flight_number"], {})
                    status = flight.get("dates", {}).get(f["date"], {})
                    entry["current_status"] = status.get("status")
                    row["flights"].append(entry)
                    secrets.add(f["flight_number"])
            else:
                row = fields(obj, "order_id status exchange_price_difference cancel_reason")
                row["delivery_location"] = fields(obj["address"], "city state country")
                row["items"] = [fields(i, "item_id name price options") for i in obj["items"]]
                secrets.update(i["item_id"] for i in obj["items"])
                for k in ("return_items", "exchange_items", "exchange_new_items"):
                    if obj.get(k) is not None:
                        row[k] = obj[k]
                        secrets.update(obj[k])
            row["payments"] = []
            methods = customer.get("payment_methods", {})
            for payment in obj["payment_history"]:
                method_id = payment.get("payment_id", payment.get("payment_method_id", ""))
                method = methods.get(method_id, {})
                row["payments"].append(dict(fields(payment, "amount transaction_type"), source=method.get("source", method.get("brand", "unknown"))))
            result[collection_name].append(row)
    elif domain == "telecom":
        device = user_db["device"]
        result["device"] = fields(
            device,
            "airplane_mode sim_card_status sim_card_missing network_connection_status network_technology_connected network_signal_strength "
            "data_enabled roaming_enabled network_mode_preference active_apn_settings wifi_enabled wifi_connected wifi_signal_strength "
            "wifi_calling_enabled wifi_calling_mms_over_wifi data_saver_mode vpn_enabled_setting vpn_connected",
        )
        if device.get("vpn_details"):
            result["device"]["vpn_performance"] = device["vpn_details"].get("server_performance")
        result["device"]["app_permissions"] = {k: v["permissions"] for k, v in device["app_statuses"].items()}
        result["surroundings"] = fields(user_db["surroundings"], "is_abroad roaming_allowed mobile_data_usage_exceeded line_active")
        result["lines"] = []
        phone = user_db["surroundings"]["phone_number"]
        for line in db["lines"]:
            if line["phone_number"] != phone:
                continue
            secrets.update([line["line_id"], line["phone_number"], line["plan_id"]])
            row = fields(line, "line_id status data_used_gb data_refueling_gb roaming_enabled contract_end_date")
            plan = next(p for p in db["plans"] if p["plan_id"] == line["plan_id"])
            row["plan"] = fields(plan, "data_limit_gb monthly_price")
            result["lines"].append(row)
        result["bills"] = [fields(b, "status total_due due_date period_start period_end") for b in db["bills"] if b["customer_id"] == customer["customer_id"]]
    else:
        raise ValueError(f"unsupported private state domain: {domain}")
    return result, secrets


def redact_private(value, scenario, public_chat, secrets=()):
    """Retain knowledge availability, but private lookup values are not executable."""
    # Reuse exact, boundary-aware public grounding; unlike the former projection
    # this does not discard whole identity/knowledge sentences.
    flat = json.dumps(value, ensure_ascii=False)
    raw = json.dumps(scenario, ensure_ascii=False)
    public = _public_text(public_chat)
    all_secrets = set(str(s) for s in secrets if s)
    from .self_teacher_context import _CUSTOMER_NAME, _IDENTIFIERS, _LETTER_REFERENCE, _PHONE

    all_secrets.update(m.group(0) for p in (_PHONE, _IDENTIFIERS) for m in p.finditer(raw + "\n" + flat))
    all_secrets.update(_LETTER_REFERENCE.findall(raw))
    known = (scenario.get("instructions") or {}).get("known_info", "")
    all_secrets.update(_CUSTOMER_NAME.findall(known))
    patterns = []
    for secret in sorted(all_secrets, key=lambda v: (-len(v), v)):
        if _contains(secret, public):
            continue
        from hashlib import sha256

        alias = "[private reference " + sha256(secret.encode()).hexdigest()[:8] + "]"
        patterns.append((_identifier_pattern(secret), alias))

    def visit(obj):
        if isinstance(obj, str):
            for pattern, alias in patterns:
                obj = pattern.sub(alias, obj)
            return obj
        if isinstance(obj, list):
            return [visit(v) for v in obj]
        if isinstance(obj, dict):
            return {k: visit(v) for k, v in obj.items()}
        return obj

    return visit(value)


def privileged_context(domain, task, public_chat, *, briefs_path, mode, environment=None, tools=None):
    if mode not in MODES:
        raise ValueError(f"unsupported self privileged context mode: {mode}")
    store = load_briefs(str(briefs_path))
    facts = store.get(domain, task)
    customer = {}
    for fact in facts:
        customer.setdefault(fact["category"], []).append(fact["text"])
    result = {"customer": customer}
    if mode == "answer_conditioned":
        from .self_teacher_answers import reference_guidance

        if tools is None:
            raise ValueError("answer-conditioned teacher requires the actual assistant tools")
        # Full answer parameters are intentional, not public consent/evidence.
        result["reference"] = reference_guidance(task, tools)
        return result
    secrets = ()
    if mode == "customer_and_state":
        if environment is None:
            raise ValueError("live environment is required for customer_and_state")
        result["current_state"], secrets = live_customer_state(domain, task["user_scenario"], environment)
    return redact_private(result, task["user_scenario"], public_chat, secrets)
