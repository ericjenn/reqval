"""
ARP4754A Validation Rules — single source of truth: rules.json
===============================================================
All validation rules (ARP4754A analysis + wording + morphology) live in
rules.json v3.0. This module loads that file once and exposes all constants
and helper functions consumed by agents.py.

Public API (unchanged — agents.py requires no modification)
───────────────────────────────────────────────────────────
Constants
  STANDARD, VERSION, PURPOSE          str
  RULESETS                            list[dict]
  MANDATORY_MODAL_VERB                str
  WEAK_MODAL_VERBS                    list[str]   from RULE-R03 checks
  AMBIGUOUS_TERMS                     list[str]   flat, from RULE-C02 checks
  AMBIGUOUS_TERMS_BY_CAT              dict        {cat_name: {severity, note, terms}}
  FORBIDDEN_PATTERNS                  list[dict]  [{pattern, severity, note}]
  SENTENCE_MORPHOLOGY                 dict        {cat: {severity, issues}}
  STRUCTURAL_RULES                    list[dict]  [{rule_id, title, check, severity, …}]
  EARS_PATTERNS                       list[dict]
  DAL_LEVELS                          dict

Functions
  get_ruleset(category)               → dict | None
  get_rules(category)                 → list[dict]
  format_rules_for_prompt(cat)        → str
  format_all_rules_for_prompt()       → str
  format_wording_for_prompt()         → str
  format_ears_for_prompt()            → str
  blocking_rules(category)            → list[dict]
  rules_by_severity(cat, sev)         → list[dict]
  severity_order(sev)                 → int
  all_rule_ids()                      → list[str]
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

# ── Load rules.json ───────────────────────────────────────────────────────────

_RULES_PATH = Path(__file__).parent / "rules.json"
if not _RULES_PATH.exists():
    raise FileNotFoundError(
        f"rules.json not found at {_RULES_PATH}. "
        "Place rules.json in the same directory as arp4754_rules.py."
    )
with open(_RULES_PATH, encoding="utf-8") as _fh:
    _RAW = json.load(_fh)

# ── Top-level metadata ────────────────────────────────────────────────────────

STANDARD : str  = _RAW.get("standard", "ARP4754A")
PURPOSE  : str  = _RAW.get("purpose",  "System Requirements Validation")
VERSION  : str  = _RAW.get("version",  "3.0")
RULESETS : list = _RAW.get("rulesets", [])

_RULESET_BY_CATEGORY: dict = {rs["category"]: rs for rs in RULESETS}

# ── Modal verb ────────────────────────────────────────────────────────────────

MANDATORY_MODAL_VERB: str = _RAW.get("mandatory_modal_verb", {}).get("verb", "shall")

# ─────────────────────────────────────────────────────────────────────────────
# Core accessors — defined BEFORE extraction helpers so get_rules() is usable
# ─────────────────────────────────────────────────────────────────────────────

def get_ruleset(category: str) -> Optional[dict]:
    return _RULESET_BY_CATEGORY.get(category)

def get_rules(category: str) -> List[dict]:
    rs = get_ruleset(category)
    return rs["rules"] if rs else []

def format_rules_for_prompt(category: str) -> str:
    """Return a structured prompt block for the given category's ruleset."""
    rs = get_ruleset(category)
    if not rs:
        return ""
    origin = rs.get("origin", "")
    desc   = rs.get("description", category)
    header = f"{origin} — {desc}" if origin else desc
    lines  = [header + "\n"]
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
        lines.append("└" + "─" * 60)
    return "\n".join(lines)

def format_all_rules_for_prompt() -> str:
    return "\n\n".join(format_rules_for_prompt(rs["category"]) for rs in RULESETS)

def severity_order(severity: str) -> int:
    return {"BLOCKING": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(severity.upper(), 99)

def rules_by_severity(category: str, severity: str) -> List[dict]:
    return [r for r in get_rules(category)
            if r.get("failure_severity", "").upper() == severity.upper()]

def blocking_rules(category: str) -> List[dict]:
    return rules_by_severity(category, "BLOCKING")

def all_rule_ids() -> List[str]:
    return [r["rule_id"] for rs in RULESETS for r in rs.get("rules", [])]


# ─────────────────────────────────────────────────────────────────────────────
# Extraction helpers — run after get_rules() is defined
# ─────────────────────────────────────────────────────────────────────────────

def _extract_weak_modals() -> List[str]:
    """
    Extract weak modal verbs from RULE-R03 check steps.
    Each relevant line is indented and has the form "  verb — note".
    """
    result = []
    for rule in get_rules("correctness"):
        if rule.get("rule_id") == "RULE-R03":
            for chk in rule.get("checks", []):
                # Lines like "  should      — recommendation, not obligation"
                m = re.match(r"^\s{2,6}(\S+(?:\s+\S+)*?)\s+—\s+", chk)
                if m:
                    result.append(m.group(1).strip())
            break
    return result


def _extract_ambiguous_terms() -> tuple[dict, List[str]]:
    """
    Extract ambiguous term categories from RULE-C02 check steps.
    Each relevant line has the form:
      [category_name / SEVERITY] term1, term2, … — note
    """
    by_cat: dict     = {}
    flat: List[str]  = []
    for rule in get_rules("completeness"):
        if rule.get("rule_id") == "RULE-C02":
            for chk in rule.get("checks", []):
                m = re.match(
                    r"^\[([^/\]]+)\s*/\s*(\w+)\]\s+(.+?)\s+—\s+(.+)$", chk
                )
                if not m:
                    continue
                cat_name = m.group(1).strip().replace(" ", "_")
                severity = m.group(2).strip()
                terms    = [t.strip() for t in m.group(3).split(",") if t.strip()]
                note     = m.group(4).strip()
                by_cat[cat_name] = {"severity": severity, "note": note, "terms": terms}
                flat.extend(terms)
            break
    return by_cat, flat


def _extract_forbidden_patterns() -> List[dict]:
    """
    Forbidden patterns come from two sources:
      • RULE-W01/W02: TBD, TBC — extracted from first check step
      • RULE-M07 [open_ended_lists]: etc., etc, including but not limited to,
        and/or — extracted from the tagged check step
    """
    result = []

    # TBD / TBC
    for rule in get_rules("wording"):
        checks = rule.get("checks", [])
        pm = re.search(r"'([^']+)'", checks[0]) if checks else None
        if pm:
            result.append({
                "pattern":  pm.group(1),
                "severity": rule.get("failure_severity", "BLOCKING"),
                "note":     checks[1] if len(checks) > 1 else "",
            })

    # open_ended_lists patterns from RULE-M07
    for rule in get_rules("morphology"):
        if rule.get("rule_id") == "RULE-M07":
            for chk in rule.get("checks", []):
                if "[open_ended_lists]" not in chk:
                    continue
                # Extract all single-quoted tokens
                patterns = re.findall(r"'([^']+)'", chk)
                note_part = chk.split("—")[-1].strip() if "—" in chk else chk
                for pat in patterns:
                    result.append({
                        "pattern":  pat,
                        "severity": rule.get("failure_severity", "HIGH"),
                        "note":     note_part,
                    })
            break

    return result


def _extract_sentence_morphology() -> dict:
    """
    Build {category_name: {severity, issues: {tag: description}}} from
    morphology rules RULE-M01..RULE-M07 (not M08 which is sentence length).
    Each check step starting with [tag] contributes one issue entry.
    """
    result  = {}
    tag_re  = re.compile(r"^\[([^\]]+)\]\s+(.+)$")
    for rule in get_rules("morphology"):
        rid = rule.get("rule_id", "")
        if rid == "RULE-M08":
            continue
        cat_name = rule.get("title", rid).lower().replace(" ", "_")
        severity = rule.get("failure_severity", "MEDIUM")
        issues   = {}
        for chk in rule.get("checks", []):
            m = tag_re.match(chk.strip())
            if m:
                issues[m.group(1)] = m.group(2)
        if issues:
            result[cat_name] = {"severity": severity, "issues": issues}
    return result


def _extract_structural_rules() -> List[dict]:
    """
    Expose RULE-M08 (sentence length) in the legacy STRUCTURAL_RULES shape
    for backward-compatible callers.
    """
    result = []
    for rule in get_rules("morphology"):
        if rule.get("rule_id") == "RULE-M08":
            checks = rule.get("checks", [])
            result.append({
                "rule_id":    rule["rule_id"],
                "title":      rule.get("title", ""),
                "check":      checks[0] if checks else "",
                "severity":   rule.get("failure_severity", "MEDIUM"),
                "note":       rule.get("objective", ""),
                "threshold":  30,
                "comparison": "gt",
            })
    return result


# ── EARS patterns ─────────────────────────────────────────────────────────────

_EARS_SECTION  = _RAW.get("ears_patterns", {})
EARS_PATTERNS: List[dict] = _EARS_SECTION.get("patterns", [])

# ── DAL levels ────────────────────────────────────────────────────────────────

DAL_LEVELS = {
    "A": "Catastrophic — Loss of aircraft or multiple fatalities",
    "B": "Hazardous — Large reduction in safety margins, crew distress",
    "C": "Major — Significant reduction in safety margins",
    "D": "Minor — Slight reduction in safety margins",
    "E": "No safety effect",
}

# ── Derived constants (populated after all helpers are defined) ───────────────

WEAK_MODAL_VERBS:      List[str]  = _extract_weak_modals()
AMBIGUOUS_TERMS_BY_CAT, AMBIGUOUS_TERMS = _extract_ambiguous_terms()
FORBIDDEN_PATTERNS:    List[dict] = _extract_forbidden_patterns()
SENTENCE_MORPHOLOGY:   dict       = _extract_sentence_morphology()
STRUCTURAL_RULES:      List[dict] = _extract_structural_rules()


# ═════════════════════════════════════════════════════════════════════════════
# Prompt formatters
# ═════════════════════════════════════════════════════════════════════════════

def format_wording_for_prompt() -> str:
    """
    Full wording & morphology block for injection into agent prompts.
    Sections: weak modals, ambiguous terms, forbidden patterns,
    morphology rules, sentence length.
    """
    lines = [
        f"WORDING & MORPHOLOGY RULES (rules.json v{VERSION})",
        "═" * 65,
        f'MANDATORY MODAL VERB: "{MANDATORY_MODAL_VERB}" — all binding obligations.',
        "",
    ]

    # ── Weak modals (RULE-R03) ────────────────────────────────────────────
    lines.append("WEAK MODAL VERBS (RULE-R03) — replace with 'shall' for binding obligations:")
    for rule in get_rules("correctness"):
        if rule["rule_id"] == "RULE-R03":
            for chk in rule.get("checks", []):
                if re.match(r"^\s{2,6}\S", chk):
                    lines.append(f"  •{chk}")
            break
    lines.append("")

    # ── Ambiguous terms (RULE-C02) ────────────────────────────────────────
    lines.append("AMBIGUOUS / UNVERIFIABLE TERMS (RULE-C02) — flag with category and remediation:")
    for cat_name, cat in AMBIGUOUS_TERMS_BY_CAT.items():
        terms = ", ".join(cat.get("terms", []))
        lines.append(f"  [{cat_name} / {cat['severity']}]  {terms}")
        lines.append(f"    → {cat['note']}")
    lines.append("")

    # ── Forbidden patterns (RULE-W01/W02, RULE-M07) ───────────────────────
    lines.append("FORBIDDEN PATTERNS (RULE-W01/W02, RULE-M07) — always flag, unconditionally:")
    for p in FORBIDDEN_PATTERNS:
        lines.append(f"  • '{p['pattern']}' [{p['severity']}]  {p.get('note', '')}")
    lines.append("")

    # ── Morphology (RULE-M01..M07) ────────────────────────────────────────
    lines.append("SENTENCE MORPHOLOGY BAD PRACTICES (RULE-M01..M07):")
    for rule in get_rules("morphology"):
        if rule["rule_id"] == "RULE-M08":
            continue
        rid = rule["rule_id"]
        sev = rule.get("failure_severity", "MEDIUM")
        lines.append(f"  [{rid} — {rule['title']}] [{sev}]")
        for chk in rule.get("checks", []):
            lines.append(f"    • {chk}")
    lines.append("")

    # ── Sentence length (RULE-M08) ────────────────────────────────────────
    for rule in get_rules("morphology"):
        if rule["rule_id"] == "RULE-M08":
            sev = rule.get("failure_severity", "MEDIUM")
            lines.append(f"SENTENCE LENGTH (RULE-M08) [{sev}]:")
            for chk in rule.get("checks", []):
                lines.append(f"  • {chk}")
            break

    return "\n".join(lines)


def format_ears_for_prompt() -> str:
    """EARS patterns block for injection into the recommender agent prompt."""
    ref = _EARS_SECTION.get("reference", "")
    lines = [
        "EARS PATTERNS — Easy Approach to Requirements Syntax",
        "═" * 65,
        f"Reference: {ref}",
        "",
        "INSTRUCTION: Choose the MOST APPROPRIATE EARS pattern for every rewrite.",
        "Do NOT default to Ubiquitous when a trigger, state, condition, or",
        "unwanted behaviour is present or implied.",
        "",
    ]
    for p in EARS_PATTERNS:
        kw = p.get("keyword") or "—"
        lines += [
            f"┌─ {p['name']}  (keyword: {kw})",
            f"│  Template : {p['template']}",
            f"│  Use when : {p['use_when']}",
            f"│  Example  : {p['example']}",
            "└" + "─" * 65,
        ]
    return "\n".join(lines)
