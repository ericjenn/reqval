"""
req_parser.py — JSON Requirements Loader
==========================================
Loads requirements from the canonical JSON format:

{
  "metadata": {
    "project":                   "...",
    "purpose":                   "...",
    "intentionally_non_compliant": true | false,
    "version":                   "1.0"
  },
  "requirements": [
    {
      "req_id":              "REQ-001",
      "req_title":           "...",
      "req_statement":       "The system shall ...",
      "verification_method": "Test" | "Analysis" | "Inspection" | "Demonstration" | null,
      "rationale":           "..."
    },
    ...
  ]
}

Public API
----------
  load(source)        — accepts a JSON string, a file path, or a dict
                        returns (metadata, requirements_list)
  to_internal(reqs)  — converts the JSON list to the internal pipeline format
                        used by all agents (adds computed boolean flags)
  load_internal(src) — convenience: load + to_internal in one call

Internal format (what agents receive)
--------------------------------------
  id                      : str   — req_id
  title                   : str   — req_title (or "")
  text                    : str   — req_statement (verbatim)
  verification_method     : str   — normalised method string (or "")
  rationale               : str   — rationale (or "")
  has_identifier          : bool  — True (id was explicitly provided)
  has_verification_method : bool  — True when method is non-null and non-empty
  has_source              : bool  — True when rationale is non-empty
  has_dal                 : bool  — True when statement or rationale mentions DAL A-E
  modal_verb              : str   — dominant modal in statement
  ambiguous_terms_found   : list  — populated by orchestrator LLM call
  compound                : bool  — populated by orchestrator LLM call
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ── Constants ─────────────────────────────────────────────────────────────────

# Verification method aliases → canonical label
_VM_ALIASES: Dict[str, str] = {
    "test":          "Test",
    "t":             "Test",
    "analysis":      "Analysis",
    "a":             "Analysis",
    "inspection":    "Inspection",
    "i":             "Inspection",
    "demonstration": "Demonstration",
    "demo":          "Demonstration",
    "d":             "Demonstration",
    "none":          "",
    "none specified": "",
    "n/a":           "",
    "tbd":           "TBD",
    "tbc":           "TBD",
}

_DAL_RE      = re.compile(r"\bDAL[-\s]?[A-E]\b", re.IGNORECASE)
_MODAL_RE    = re.compile(r"\b(shall|should|must|may|might|could|will)\b", re.IGNORECASE)

# ── JSON field names accepted as synonyms ─────────────────────────────────────
# Allows the loader to handle minor field-name variants without breaking.

_ID_KEYS    = ("req_id", "id", "req-id", "requirement_id", "identifier")
_TITLE_KEYS = ("req_title", "title", "name", "req-title", "requirement_title")
_STMT_KEYS  = ("req_statement", "statement", "text", "description",
               "requirement", "req-statement")
_VM_KEYS    = ("verification_method", "verification", "method",
               "verification_methods", "verif")
_RAT_KEYS   = ("rationale", "source", "justification", "note", "comment")


def _pick(d: dict, keys: tuple, default: Any = None) -> Any:
    """Return the value of the first matching key found in d."""
    for k in keys:
        if k in d:
            return d[k]
    return default


def _normalise_vm(raw: Optional[str]) -> str:
    """Normalise a verification method string to a canonical label."""
    if raw is None:
        return ""
    key = raw.strip().lower()
    return _VM_ALIASES.get(key, raw.strip())   # keep original if not in aliases


def _detect_modal(text: str) -> str:
    """Return the first (dominant) modal verb found in the statement."""
    m = _MODAL_RE.search(text)
    return m.group(1).lower() if m else "none"


# ── Core loader ───────────────────────────────────────────────────────────────

def _parse_json_source(source: Union[str, dict, Path]) -> dict:
    """
    Accept a JSON string, a file path (str or Path), or an already-parsed dict.
    Returns the parsed dict.
    """
    if isinstance(source, dict):
        return source

    if isinstance(source, Path) or (isinstance(source, str) and
                                     not source.strip().startswith("{")):
        # Treat as a file path
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Requirements file not found: {path}")
        text = path.read_text(encoding="utf-8")
    else:
        text = source   # raw JSON string

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc


def load(source: Union[str, dict, Path]) -> Tuple[dict, List[dict]]:
    """
    Load requirements from a JSON string, file path, or dict.

    Returns:
        (metadata, raw_requirements_list)
        where raw_requirements_list contains the original JSON objects.

    Raises:
        ValueError  — if the JSON is structurally invalid
        FileNotFoundError — if a path was given and the file doesn't exist
    """
    data = _parse_json_source(source)

    if not isinstance(data, dict):
        raise ValueError("Top-level JSON value must be an object { ... }")

    metadata = data.get("metadata", {})

    raw_reqs = data.get("requirements")
    if raw_reqs is None:
        # Try bare array at top level
        if isinstance(data, list):
            raw_reqs = data
        else:
            raise ValueError(
                'JSON must have a "requirements" array at the top level. '
                'Keys found: ' + str(list(data.keys()))
            )

    if not isinstance(raw_reqs, list):
        raise ValueError('"requirements" must be a JSON array.')

    if not raw_reqs:
        raise ValueError('"requirements" array is empty — nothing to validate.')

    return metadata, raw_reqs


def to_internal(raw_reqs: List[dict]) -> List[dict]:
    """
    Convert the raw JSON requirement list to the internal pipeline format.

    Computes all boolean flags derivable without an LLM:
      has_identifier, has_verification_method, has_source, has_dal, modal_verb

    The orchestrator LLM will later fill in:
      type, ambiguous_terms_found, compound
    """
    internal: List[dict] = []

    for i, raw in enumerate(raw_reqs):
        if not isinstance(raw, dict):
            raise ValueError(f"Requirement at index {i} is not an object: {raw!r}")

        req_id = str(_pick(raw, _ID_KEYS, f"AUTO-{i+1:03d}")).strip()
        title  = str(_pick(raw, _TITLE_KEYS, "") or "").strip()
        text   = str(_pick(raw, _STMT_KEYS,  "") or "").strip()
        vm_raw = _pick(raw, _VM_KEYS,   None)
        rat    = str(_pick(raw, _RAT_KEYS,  "") or "").strip()

        if not text:
            raise ValueError(
                f'Requirement "{req_id}" has no statement text. '
                f'Expected one of: {_STMT_KEYS}'
            )

        vm = _normalise_vm(vm_raw)

        internal.append({
            # ── Source fields ────────────────────────────────────────
            "id":                   req_id,
            "title":                title,
            "text":                 text,
            "verification_method":  vm,
            "rationale":            rat,
            # ── Computed boolean flags ───────────────────────────────
            "has_identifier":          not req_id.startswith("AUTO-"),
            "has_verification_method": bool(vm),
            "has_source":              bool(rat),
            "has_dal":                 bool(_DAL_RE.search(text) or _DAL_RE.search(rat)),
            "modal_verb":              _detect_modal(text),
            # ── Placeholders filled by orchestrator LLM ─────────────
            "type":                 "unknown",
            "ambiguous_terms_found": [],
            "compound":             False,
        })

    return internal


def load_internal(source: Union[str, dict, Path]) -> Tuple[dict, List[dict]]:
    """
    Convenience: load JSON and return (metadata, internal_requirements_list).
    This is the main entry point for the orchestrator agent.
    """
    metadata, raw_reqs = load(source)
    return metadata, to_internal(raw_reqs)


def summary(metadata: dict, reqs: List[dict]) -> str:
    """Return a one-line human-readable summary for CLI display."""
    project = metadata.get("project", "unnamed project")
    nc      = " [INTENTIONALLY NON-COMPLIANT]" if metadata.get("intentionally_non_compliant") else ""
    return f'Loaded {len(reqs)} requirements from "{project}"{nc}'
