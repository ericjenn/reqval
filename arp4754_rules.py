"""
ARP4754A Validation Rules — loaded from rules.json
====================================================
This module is the single source of truth for all validation rules.
It reads rules.json at import time and exposes:

  RULESETS        — full parsed JSON structure (list of category dicts)
  get_ruleset(cat)— returns the ruleset dict for a category name
  format_rules_for_prompt(cat) — returns a formatted string ready to
                                  inject into an LLM system prompt,
                                  including checks and failure_severity
  AMBIGUOUS_TERMS — list of vague/subjective terms to flag
  WEAK_MODAL_VERBS— modal verbs weaker than "shall"
  DAL_LEVELS      — DAL A-E definitions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

# ── Load rules.json (co-located with this file) ───────────────────────────
_RULES_PATH = Path(__file__).parent / "rules.json"

if not _RULES_PATH.exists():
    raise FileNotFoundError(
        f"rules.json not found at {_RULES_PATH}. "
        "Place rules.json in the same directory as arp4754_rules.py."
    )

with open(_RULES_PATH, encoding="utf-8") as _fh:
    _RAW = json.load(_fh)

STANDARD  : str  = _RAW.get("standard", "ARP4754A")
PURPOSE   : str  = _RAW.get("purpose", "System Requirements Validation")
VERSION   : str  = _RAW.get("version", "1.0")
RULESETS  : list = _RAW.get("rulesets", [])

# Build lookup by category name for O(1) access
_RULESET_BY_CATEGORY: dict = {rs["category"]: rs for rs in RULESETS}


# ── Accessors ─────────────────────────────────────────────────────────────

def get_ruleset(category: str) -> Optional[dict]:
    """Return the full ruleset dict for a category (e.g. 'completeness')."""
    return _RULESET_BY_CATEGORY.get(category)


def get_rules(category: str) -> List[dict]:
    """Return the list of rule dicts for a category."""
    rs = get_ruleset(category)
    return rs["rules"] if rs else []


def format_rules_for_prompt(category: str) -> str:
    """
    Build a richly formatted block ready for injection into an LLM system
    prompt. Each rule is rendered with:
      - rule_id and title
      - objective
      - numbered check steps (the actual checks to perform)
      - failure_severity badge

    Example output:
      ┌─ REQ-C01 · Unique requirement identifier  [SEVERITY: HIGH]
      │  Objective: Ensure each requirement is uniquely identifiable
      │  Checks:
      │    1. Verify the requirement has an identifier
      │    2. Verify the identifier is unique across the requirement set
      └─────────────────────────────────────────────────────────────

    Returns empty string if category is not found.
    """
    rs = get_ruleset(category)
    if not rs:
        return ""

    section = rs.get("arp4754a_section", "?")
    desc    = rs.get("description", category)
    lines   = [f"ARP4754A §{section} — {desc}\n"]

    for rule in rs.get("rules", []):
        rid      = rule.get("rule_id", "???")
        title    = rule.get("title", "")
        obj      = rule.get("objective", "")
        checks   = rule.get("checks", [])
        severity = rule.get("failure_severity", "MEDIUM")

        lines.append(f"┌─ {rid} · {title}  [SEVERITY: {severity}]")
        lines.append(f"│  Objective : {obj}")
        lines.append( "│  Checks    :")
        for i, chk in enumerate(checks, 1):
            lines.append(f"│    {i}. {chk}")
        lines.append( "└" + "─" * 60)

    return "\n".join(lines)


def format_all_rules_for_prompt() -> str:
    """Return formatted rules for all categories, separated by blank lines."""
    return "\n\n".join(
        format_rules_for_prompt(rs["category"]) for rs in RULESETS
    )


def severity_order(severity: str) -> int:
    """Numeric rank for sorting by severity (lower = more severe)."""
    return {"BLOCKING": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(
        severity.upper(), 99
    )


def rules_by_severity(category: str, severity: str) -> List[dict]:
    """Return only rules of a given severity level within a category."""
    return [
        r for r in get_rules(category)
        if r.get("failure_severity", "").upper() == severity.upper()
    ]


def blocking_rules(category: str) -> List[dict]:
    return rules_by_severity(category, "BLOCKING")


def all_rule_ids() -> List[str]:
    """Flat list of every rule_id across all categories."""
    return [r["rule_id"] for rs in RULESETS for r in rs.get("rules", [])]


# ── Retained constants (used by agents for quick reference) ───────────────

AMBIGUOUS_TERMS = [
    "fast", "slow", "quickly", "soon", "adequate", "sufficient", "appropriate",
    "reliable", "robust", "flexible", "user-friendly", "intuitive", "efficient",
    "as required", "if applicable", "where necessary", "when needed", "approximately",
    "about", "around", "roughly", "good", "bad", "better", "best", "optimal",
    "minimize", "maximize", "normal", "abnormal", "reasonable", "acceptable",
]

WEAK_MODAL_VERBS = ["may", "might", "could", "would", "can", "should be able to"]

DAL_LEVELS = {
    "A": "Catastrophic — Loss of aircraft or multiple fatalities",
    "B": "Hazardous — Large reduction in safety margins, crew distress",
    "C": "Major — Significant reduction in safety margins",
    "D": "Minor — Slight reduction in safety margins",
    "E": "No safety effect",
}
