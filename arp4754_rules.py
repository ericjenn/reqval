"""
ARP4754A Validation Rules — loaded from rules.json + wording_rules.json
=========================================================================
Single source of truth for all validation and wording rules.

From rules.json
───────────────
  RULESETS, STANDARD, VERSION, PURPOSE
  get_rules(cat), format_rules_for_prompt(cat), blocking_rules(cat), …

From wording_rules.json
───────────────────────
  MANDATORY_MODAL_VERB       str
  WEAK_MODAL_VERBS           list[str]
  AMBIGUOUS_TERMS            list[str]          flat, for quick membership tests
  AMBIGUOUS_TERMS_BY_CAT     dict[str, dict]    categorised, with severity + note
  FORBIDDEN_PATTERNS         list[dict]         {pattern, severity, category, note}
  SENTENCE_MORPHOLOGY        dict               bad-practice taxonomy by category
  STRUCTURAL_RULES           list[dict]         explicit rule_id entries
  EARS_PATTERNS              list[dict]         EARS rewriting templates
  DAL_LEVELS                 dict

  format_wording_for_prompt()  → str   full wording block for analysis agents
  format_ears_for_prompt()     → str   EARS patterns block for recommender agent
"""

from __future__ import annotations

import json
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

STANDARD  : str  = _RAW.get("standard", "ARP4754A")
PURPOSE   : str  = _RAW.get("purpose",  "System Requirements Validation")
VERSION   : str  = _RAW.get("version",  "1.0")
RULESETS  : list = _RAW.get("rulesets", [])
_RULESET_BY_CATEGORY: dict = {rs["category"]: rs for rs in RULESETS}

# ── Load wording_rules.json ───────────────────────────────────────────────────
_WORDING_PATH = Path(__file__).parent / "wording_rules.json"
if not _WORDING_PATH.exists():
    raise FileNotFoundError(
        f"wording_rules.json not found at {_WORDING_PATH}. "
        "Place wording_rules.json in the same directory as arp4754_rules.py."
    )
with open(_WORDING_PATH, encoding="utf-8") as _wfh:
    _WORDING = json.load(_wfh)

# ── Modal verb constants ──────────────────────────────────────────────────────
MANDATORY_MODAL_VERB: str = _WORDING.get("mandatory_modal_verb", {}).get("verb", "shall")

WEAK_MODAL_VERBS: List[str] = [
    e["verb"] for e in _WORDING.get("weak_modal_verbs", {}).get("verbs", [])
]

# ── Ambiguous terms ───────────────────────────────────────────────────────────
# Categorised dict (with severity + note per category) — for rich prompt injection
AMBIGUOUS_TERMS_BY_CAT: dict = _WORDING.get("ambiguous_terms", {}).get("categories", {})

# Flat list — for quick membership tests and backward-compatible agent usage
AMBIGUOUS_TERMS: List[str] = [
    term
    for cat in AMBIGUOUS_TERMS_BY_CAT.values()
    for term in cat.get("terms", [])
]

# ── Forbidden patterns ────────────────────────────────────────────────────────
FORBIDDEN_PATTERNS: List[dict] = _WORDING.get("forbidden_patterns", {}).get("patterns", [])

# ── Sentence morphology bad practices ────────────────────────────────────────
SENTENCE_MORPHOLOGY: dict = _WORDING.get("sentence_morphology_bad_practices", {}).get("categories", {})

# ── Structural rules (rule_id keyed) ─────────────────────────────────────────
STRUCTURAL_RULES: List[dict] = _WORDING.get("structural_rules", {}).get("rules", [])

# ── EARS patterns ─────────────────────────────────────────────────────────────
EARS_PATTERNS: List[dict] = _WORDING.get("ears_patterns", {}).get("patterns", [])

# ── DAL levels ────────────────────────────────────────────────────────────────
DAL_LEVELS = {
    "A": "Catastrophic — Loss of aircraft or multiple fatalities",
    "B": "Hazardous — Large reduction in safety margins, crew distress",
    "C": "Major — Significant reduction in safety margins",
    "D": "Minor — Slight reduction in safety margins",
    "E": "No safety effect",
}


# ── rules.json accessors ──────────────────────────────────────────────────────

def get_ruleset(category: str) -> Optional[dict]:
    return _RULESET_BY_CATEGORY.get(category)

def get_rules(category: str) -> List[dict]:
    rs = get_ruleset(category)
    return rs["rules"] if rs else []

def format_rules_for_prompt(category: str) -> str:
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


# ── Wording prompt formatters ─────────────────────────────────────────────────

def format_wording_for_prompt() -> str:
    """
    Full wording & morphology block for injection into analysis agent prompts
    (completeness, correctness). Covers all sections of wording_rules.json.
    """
    lines = [
        "WORDING & MORPHOLOGY RULES (wording_rules.json v"
        + _WORDING.get("version", "?") + ")",
        "═" * 65,
        f'MANDATORY MODAL VERB: "{MANDATORY_MODAL_VERB}" — all binding obligations.',
        "",
    ]

    # ── Weak modals ──
    lines.append("WEAK MODAL VERBS — flag as REQ-R03 [LOW]:")
    for e in _WORDING.get("weak_modal_verbs", {}).get("verbs", []):
        lines.append(f"  • {e['verb']:<24} {e.get('note', '')}")
    lines.append("")

    # ── Ambiguous terms by category ──
    lines.append("AMBIGUOUS / UNVERIFIABLE TERMS — flag as REQ-C02, REQ-V02, REQ-V03 [HIGH]:")
    for cat_name, cat in AMBIGUOUS_TERMS_BY_CAT.items():
        sev   = cat.get("severity", "HIGH")
        note  = cat.get("note", "")
        terms = ", ".join(cat.get("terms", []))
        lines.append(f"  [{cat_name}] [{sev}]  {terms}")
        lines.append(f"    → {note}")
    lines.append("")

    # ── Forbidden patterns ──
    lines.append("FORBIDDEN PATTERNS — always flag regardless of context:")
    for p in FORBIDDEN_PATTERNS:
        sev = p.get("severity", "HIGH")
        lines.append(f"  • {p['pattern']:<28} [{sev}]  {p.get('note', '')}")
    lines.append("")

    # ── Sentence morphology bad practices ──
    lines.append("SENTENCE MORPHOLOGY BAD PRACTICES:")
    for cat_name, cat in SENTENCE_MORPHOLOGY.items():
        sev = cat.get("severity", "MEDIUM")
        lines.append(f"  [{cat_name}] [{sev}]")
        for issue_id, desc in cat.get("issues", {}).items():
            lines.append(f"    • {issue_id}: {desc}")
    lines.append("")

    # ── Structural rules ──
    lines.append("STRUCTURAL RULES (pre-parser checks):")
    for r in STRUCTURAL_RULES:
        sev = r.get("severity", "MEDIUM")
        lines.append(f"  • {r['rule_id']} [{sev}]  {r['title']}: {r['check']}")

    return "\n".join(lines)


def format_ears_for_prompt() -> str:
    """
    EARS patterns block for injection into the recommender agent prompt.
    Provides the pattern name, keyword, template, when to use it, and an example.
    """
    meta = _WORDING.get("ears_patterns", {})
    ref  = meta.get("reference", "")
    lines = [
        "EARS PATTERNS — Easy Approach to Requirements Syntax",
        "═" * 65,
        f"Reference: {ref}",
        "",
        "INSTRUCTION: For every corrected rewrite, choose the MOST APPROPRIATE",
        "EARS pattern based on the nature of the requirement, and apply its",
        "template exactly. Do NOT default to Ubiquitous when a trigger, state,",
        "condition, or unwanted behaviour is present or implied.",
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
