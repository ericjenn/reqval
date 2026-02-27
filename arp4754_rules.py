"""
ARP4754A Validation Rules — single source of truth: rules.json
===============================================================
All validation rules live in rules.json.  This module loads that file once
and exposes constants and helpers consumed by agents.py.

Design principle: ZERO hardcoded rule IDs.  All rule content is derived
from the JSON structure (category names, field presence, tag patterns in
check text) so that adding, removing, or renaming rules in rules.json
requires no code changes here or in agents.py.

Public API
──────────
Constants
  STANDARD, VERSION, PURPOSE
  RULESETS, MANDATORY_MODAL_VERB
  WEAK_MODAL_VERBS, AMBIGUOUS_TERMS, AMBIGUOUS_TERMS_BY_CAT
  FORBIDDEN_PATTERNS, SENTENCE_MORPHOLOGY, STRUCTURAL_RULES
  EARS_PATTERNS, DAL_LEVELS

Functions
  get_ruleset(category)
  get_rules(category)
  get_checks(category, rule_id)
  format_rules_for_prompt(cat)
  format_all_rules_for_prompt()
  format_output_format(cat)         → per-req output-format block
  format_consistency_output_format()
  format_instructions(cat)          → numbered instruction list
  format_consistency_instructions()
  format_wording_output_format()
  format_wording_instructions()
  format_wording_for_prompt()
  format_ears_for_prompt()
  blocking_rules(category)
  rules_by_severity(cat, sev)
  severity_order(sev)
  all_rule_ids()
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

STANDARD : str  = _RAW.get("standard", "ARP4754A")
PURPOSE  : str  = _RAW.get("purpose",  "System Requirements Validation")
VERSION  : str  = _RAW.get("version",  "3.0")
RULESETS : list = _RAW.get("rulesets", [])

_RULESET_BY_CATEGORY: dict = {rs["category"]: rs for rs in RULESETS}

MANDATORY_MODAL_VERB: str = _RAW.get("mandatory_modal_verb", {}).get("verb", "shall")

_EARS_SECTION  = _RAW.get("ears_patterns", {})
EARS_PATTERNS: List[dict] = _EARS_SECTION.get("patterns", [])

DAL_LEVELS = {
    "A": "Catastrophic — Loss of aircraft or multiple fatalities",
    "B": "Hazardous — Large reduction in safety margins, crew distress",
    "C": "Major — Significant reduction in safety margins",
    "D": "Minor — Slight reduction in safety margins",
    "E": "No safety effect",
}

# ─────────────────────────────────────────────────────────────────────────────
# Core accessors
# ─────────────────────────────────────────────────────────────────────────────

def get_ruleset(category: str) -> Optional[dict]:
    return _RULESET_BY_CATEGORY.get(category)

def get_rules(category: str) -> List[dict]:
    rs = get_ruleset(category)
    return rs["rules"] if rs else []

def get_checks(category: str, rule_id: str) -> List[str]:
    for rule in get_rules(category):
        if rule.get("rule_id") == rule_id:
            return rule.get("checks", [])
    return []

def severity_order(severity: str) -> int:
    return {"BLOCKING": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(severity.upper(), 99)

def rules_by_severity(category: str, severity: str) -> List[dict]:
    return [r for r in get_rules(category)
            if r.get("failure_severity", "").upper() == severity.upper()]

def blocking_rules(category: str) -> List[dict]:
    return rules_by_severity(category, "BLOCKING")

def all_rule_ids() -> List[str]:
    return [r["rule_id"] for rs in RULESETS for r in rs.get("rules", [])]

# ── Helper: strip sub-check ID prefix [RULE-XXX.y] ───────────────────────────

def _strip_prefix(text: str) -> str:
    return re.sub(r"^\[RULE-[A-Z0-9]+\.[a-z]\]\s*", "", text).strip()

# ─────────────────────────────────────────────────────────────────────────────
# Prompt formatters — zero hardcoded rule IDs
# ─────────────────────────────────────────────────────────────────────────────

def format_rules_for_prompt(category: str) -> str:
    """Structured prompt block for one category's ruleset."""
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
        if obj:
            lines.append(f"│  Objective : {obj}")
        lines.append("│  Checks    :")
        for chk in checks:
            lines.append(f"│    {chk}")
        lines.append("└" + "─" * 60)
    return "\n".join(lines)

def format_all_rules_for_prompt() -> str:
    return "\n\n".join(format_rules_for_prompt(rs["category"]) for rs in RULESETS)


def format_output_format(category: str) -> str:
    """
    Per-requirement output-format block, generated from rules.json.
    One result line per rule: rule_id, severity, pass/fail, title.
    """
    rules = get_rules(category)
    if not rules:
        return ""
    lines = [
        "─" * 49,
        "[REQ-ID]: [first 60 chars of requirement text...]",
        "─" * 49,
    ]
    for rule in rules:
        rid   = rule["rule_id"]
        sev   = rule["failure_severity"]
        title = rule.get("title", "")
        lines.append(f"  {rid} [{sev}] \u2713/\u2717  \u2014 [{title}: finding]")
    lines.append("  Score: XX/100")
    lines.append("─" * 49)
    return "\n".join(lines)


def format_instructions(category: str) -> str:
    """
    Numbered instruction list for a per-requirement agent.
    Generated from rules.json — each rule becomes one step.
    """
    lines = [
        "Work through EVERY rule below in order.",
        "For each check step: state PASSES (\u2713) or FAILS (\u2717) and quote the offending text.",
        "",
    ]    
    rules = get_rules(category)
    lines = []
    for i, rule in enumerate(rules, 1):
        rid    = rule["rule_id"]
        title  = rule.get("title", "")
        checks = rule.get("checks", [])
        summary = "; ".join(_strip_prefix(c) for c in checks)
        lines.append(f"{i}. {rid} ({title}): {summary}")
    n = len(rules)
    lines.append(f"{n+1}. State PASSES (\u2713) or FAILS (\u2717) with failure_severity per rule.")
    lines.append(f"{n+2}. Give a per-requirement {category} score (0\u2013100).")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction helpers — backward-compatible derived constants
# ─────────────────────────────────────────────────────────────────────────────

def _extract_weak_modals() -> List[str]:
    """
    Extract weak modal verbs from the correctness ruleset.
    Matches check lines: 'Detect <adjective> modal|phrase|paraphrase 'verb''
    No hardcoded rule IDs.
    """
    result   = []
    verb_re  = re.compile(
        r"Detect\s+(?:[\w-]+\s+)*(?:modal|phrase|paraphrase)\s+'([^']+)'",
        re.IGNORECASE,
    )
    for rule in get_rules("correctness"):
        for chk in rule.get("checks", []):
            clean = _strip_prefix(chk)
            m = verb_re.search(clean)
            if m:
                result.append(m.group(1))
    return result


def _extract_ambiguous_terms() -> tuple[dict, List[str]]:
    """
    Extract ambiguous term categories from the completeness ruleset.
    Matches check lines: 'Detect vague <category> terms such as term1, term2, …'
    No hardcoded rule IDs.
    """
    by_cat: dict    = {}
    flat: List[str] = []
    detect_re = re.compile(
        r"Detect\s+(?:vague\s+)?(.+?)\s+terms?\s+such\s+as\s+(.+)$",
        re.IGNORECASE,
    )
    for rule in get_rules("completeness"):
        severity = rule.get("failure_severity", "HIGH")
        for chk in rule.get("checks", []):
            clean = _strip_prefix(chk)
            m = detect_re.match(clean)
            if not m:
                continue
            cat_name = m.group(1).strip().replace(" ", "_")
            terms    = [t.strip() for t in m.group(2).split(",") if t.strip()]
            by_cat[cat_name] = {
                "severity": severity,
                "note": f"Replace '{cat_name.replace('_',' ')}' terms with measurable values",
                "terms": terms,
            }
            flat.extend(terms)
    return by_cat, flat


def _extract_forbidden_patterns() -> List[dict]:
    """
    Forbidden patterns from wording rules ('Detect the literal string X')
    and morphology rules ('[open_ended_lists] Detect …').
    No hardcoded rule IDs.
    """
    result       = []
    literal_re   = re.compile(r"Detect\s+the\s+literal\s+string\s+'([^']+)'", re.IGNORECASE)

    for rule in get_rules("wording"):
        severity = rule.get("failure_severity", "BLOCKING")
        for chk in rule.get("checks", []):
            clean = _strip_prefix(chk)
            m = literal_re.search(clean)
            if m:
                result.append({
                    "pattern":  m.group(1),
                    "severity": severity,
                    "note":     f"{rule['rule_id']} — {rule.get('title','')}",
                })

    for rule in get_rules("morphology"):
        severity = rule.get("failure_severity", "HIGH")
        for chk in rule.get("checks", []):
            clean = _strip_prefix(chk)
            if "open_ended_lists" not in clean.lower():
                continue
            # detect part after "Detect": extract quoted tokens
            detect_part = re.sub(r".*Detect\s+", "", clean, flags=re.IGNORECASE)
            for pat in re.findall(r"'([^']+)'", detect_part):
                result.append({
                    "pattern":  pat,
                    "severity": severity,
                    "note":     f"{rule['rule_id']} — open-ended list marker",
                })

    return result


def _extract_sentence_morphology() -> dict:
    """
    Build {rule_title_snake: {severity, issues: {semantic_tag: description}}}
    from the morphology ruleset. Secondary [semantic_tag] tokens in check text
    (after the sub-check ID prefix is stripped) are extracted as issue keys.
    No hardcoded rule IDs.
    """
    result  = {}
    tag_re  = re.compile(r"^\[([a-z_]+)\]\s+(.+)$")
    for rule in get_rules("morphology"):
        cat_name = rule.get("title", rule["rule_id"]).lower().replace(" ", "_")
        severity = rule.get("failure_severity", "MEDIUM")
        issues   = {}
        for chk in rule.get("checks", []):
            clean = _strip_prefix(chk)
            m = tag_re.match(clean)
            if m:
                issues[m.group(1)] = m.group(2)
        if issues:
            result[cat_name] = {"severity": severity, "issues": issues}
    return result


def _extract_structural_rules() -> List[dict]:
    """
    Morphology rules with word-count thresholds, in legacy STRUCTURAL_RULES shape.
    Detected by presence of a number followed by 'word(s)' in any check.
    No hardcoded rule IDs.
    """
    result   = []
    count_re = re.compile(r"(\d+)\s+words?", re.IGNORECASE)
    for rule in get_rules("morphology"):
        for chk in rule.get("checks", []):
            m = count_re.search(chk)
            if m:
                result.append({
                    "rule_id":    rule["rule_id"],
                    "title":      rule.get("title", ""),
                    "check":      chk,
                    "severity":   rule.get("failure_severity", "MEDIUM"),
                    "note":       rule.get("objective", ""),
                    "threshold":  int(m.group(1)),
                    "comparison": "gt",
                })
                break
    return result


# ── Populate derived constants ────────────────────────────────────────────────

WEAK_MODAL_VERBS:      List[str]  = _extract_weak_modals()
AMBIGUOUS_TERMS_BY_CAT, AMBIGUOUS_TERMS = _extract_ambiguous_terms()
FORBIDDEN_PATTERNS:    List[dict] = _extract_forbidden_patterns()
SENTENCE_MORPHOLOGY:   dict       = _extract_sentence_morphology()
STRUCTURAL_RULES:      List[dict] = _extract_structural_rules()


# ─────────────────────────────────────────────────────────────────────────────
# Wording + EARS formatters
# ─────────────────────────────────────────────────────────────────────────────

def format_ears_for_prompt() -> str:
    """EARS patterns block for the recommender agent prompt."""
    ref = _EARS_SECTION.get("reference", "")
    lines = [
        "EARS PATTERNS — Easy Approach to Requirements Syntax",
        "=" * 65,
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
            f"+- {p['name']}  (keyword: {kw})",
            f"|  Template : {p['template']}",
            f"|  Use when : {p['use_when']}",
            f"|  Example  : {p['example']}",
            "-" * 65,
        ]
    return "\n".join(lines)
