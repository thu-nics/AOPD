"""Shared teacher policy wording. Student prompts and real history stay intact.

Reference answers are opt-in through the separate teacher-only projection.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

MANUAL_PATH = Path(__file__).with_name("self_teacher_telecom_manual.md")
# The public policy is pinned too: do not silently replace a changed upstream
# manual with an outdated paraphrase. Other domains pass through unchanged.
NATIVE_MANUAL_SHA256 = "015a3ee49ec8c199c5e1a059c922081937b92cf3a5ea74fedbc4357816f389ba"


@lru_cache(maxsize=1)
def context_projection_fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes() + MANUAL_PATH.read_bytes() + Path(__file__).with_name("self_teacher_privilege.py").read_bytes() + Path(__file__).with_name("customer_briefs.py").read_bytes() + Path(__file__).with_name("self_teacher_answers.py").read_bytes()).hexdigest()


@lru_cache(maxsize=32)
def teacher_public_policy(content: str) -> str:
    """Replace only the known Telecom manual; never rewrite chat/tool results."""
    opening, closing = "<tech_support_policy>", "</tech_support_policy>"
    if opening not in content and closing not in content:
        return content
    if content.count(opening) != 1 or content.count(closing) != 1:
        raise ValueError("unexpected Telecom policy wrapper; review Self-AOPD wording")
    start = content.index(opening) + len(opening)
    end = content.index(closing)
    original = content[start:end].strip()
    if hashlib.sha256(original.encode()).hexdigest() != NATIVE_MANUAL_SHA256:
        raise ValueError("Telecom public manual changed; review Self-AOPD wording")
    projected = content[:start] + "\n" + MANUAL_PATH.read_text().strip() + "\n" + content[end:]
    # The main policy also mentions two customer-side payment APIs. Keep the
    # payment/consent/status-check procedure, but not the simulator call syntax.
    projected = projected.replace("Check their payment requests using the check_payment_request tool.", "Review the payment request they received.")
    projected = projected.replace("If the user accepts the payment request, use the make_payment tool to make the payment.", "If the customer accepts the payment request, ask them to complete the payment.")
    if "check_payment_request" in projected or "make_payment" in projected:
        raise ValueError("Telecom customer payment instructions changed; review Self-AOPD wording")
    return projected


_PHONE = re.compile(r"(?<!\w)(?:\+1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?!\w)")
_IDENTIFIERS = re.compile(
    r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b"  # email
    r"|\b[a-zA-Z]+(?:_[a-zA-Z]+)*_\d+\b"  # Tau user/payment IDs
    r"|\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{5,}\b"  # reservation/order/flight references
    r"|(?<![\w$€.])\d{5,}(?!\w|\.\d| (?:dollars|USD|GB)\b)"  # postal/item/order IDs, not amounts
)
_LETTER_REFERENCE = re.compile(r"(?i:\b(?:reservations?(?:\s+(?:IDs?|number))?|confirmation(?:\s+number)?)\s*(?:(?:is|are)\s*)?[\s:#('\"]*)([A-Z]{6})\b")
_CUSTOMER_NAME = re.compile(
    r"(?:You are |You're |Your name is |You name is |called )"
    r"([A-Z][a-z]+(?: [A-Z][a-z]+)?)(?=[,.(]| with | in | from | and | living | residing )"
)


def _identifier_pattern(value):
    # Boundaries prevent an ID appearing as part of an email/longer ID from
    # counting as independently disclosed. Phone punctuation is not semantic.
    if _PHONE.fullmatch(value):
        digits = re.sub(r"\D", "", value)[-10:]
        body = r"[\s().-]*".join(digits)
        return re.compile(r"(?<!\w)(?:\+?1[\s().-]*)?" + body + r"(?!\w)")
    return re.compile(r"(?<![\w@.+-])" + re.escape(value) + r"(?![\w@+-]|\.[\w])", re.IGNORECASE)
