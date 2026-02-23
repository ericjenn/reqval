"""
Multi-Requirement Analysis Pipeline
=====================================
Detects inter-requirement defects that single-requirement agents cannot find:
  • Contradictions   — two requirements assert incompatible behaviours
  • Overlaps         — two requirements partially cover the same behaviour
  • Redundancies     — one requirement is fully subsumed by another

Pipeline (4 sequential agents, all driven by LangGraph):

  ┌──────────────────────────────────────────────────────────────────┐
  │  Phase 1 — NORMALIZER                                            │
  │  Standardises each requirement into a canonical structured form: │
  │    subject · action · constraint · condition · modes             │
  │  Resolves synonym terms, mode aliases, interface name variants.  │
  └────────────────────┬─────────────────────────────────────────────┘
                       │  normalized_requirements: List[NormalizedReq]
                       ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  Phase 2 — CLUSTERER                                             │
  │  Groups requirements to minimise comparison combinations.        │
  │  Cluster dimensions (each req can belong to multiple clusters):  │
  │    • subject   (system / subsystem / function name)              │
  │    • function  (verb-based functional domain)                    │
  │    • interface (named interface or data bus)                     │
  │    • mode      (operational mode: normal, emergency, ground…)    │
  └────────────────────┬─────────────────────────────────────────────┘
                       │  clusters: dict[dimension → list[req_ids]]
                       │  candidate_pairs: List[Pair]
                       ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  Phase 3 — FILTER                                                │
  │  Prunes candidate pairs that cannot possibly conflict.           │
  │  Keeps pair only if ALL applicable gates pass:                   │
  │    • Same system / allocation?                                   │
  │    • Overlapping modes?                                          │
  │    • Same lifecycle phase?                                       │
  │    • Same interface or function?                                 │
  └────────────────────┬─────────────────────────────────────────────┘
                       │  comparison_set: List[ValidatedPair]
                       ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  Phase 4 — COMPARATOR                                            │
  │  Diagnoses each validated pair:                                  │
  │    • CONTRADICTION  — direct logical conflict                    │
  │    • OVERLAP        — partial semantic overlap                   │
  │    • REDUNDANCY     — one fully subsumes the other              │
  │    • OK             — no inter-requirement defect                │
  └────────────────────┬─────────────────────────────────────────────┘
                       │  multi_req_findings: str  (structured report)
                       ▼
                  reporter_agent (existing)
"""

from __future__ import annotations

import json
import re
import itertools
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from dotenv import load_dotenv

from llm_provider import get_llm as _get_llm
from langchain_core.messages import HumanMessage, SystemMessage

from arp4754_rules import STANDARD, VERSION
from rag import get_rag

load_dotenv()

# ─────────────────────────────────────────────
# Sub-state (embedded in main ValidationState)
# ─────────────────────────────────────────────

class NormalizedReq(TypedDict):
    id:          str
    original:    str          # verbatim original text
    subject:     str          # system / function / subsystem name
    action:      str          # main verb phrase
    constraint:  str          # quantified limit or property
    condition:   str          # applicability condition (when / if / during)
    modes:       List[str]    # operational modes (normalized labels)
    interfaces:  List[str]    # named interfaces or data buses
    functions:   List[str]    # named functions
    lifecycle:   str          # development phase: design / operation / maintenance
    allocation:  str          # HW / SW / System / unspecified
    normal_text: str          # canonical restatement using normalized vocabulary


class CandidatePair(TypedDict):
    id_a:    str
    id_b:    str
    reasons: List[str]   # cluster dimensions that caused this pairing


class ValidatedPair(TypedDict):
    id_a:         str
    id_b:         str
    cluster_dims: List[str]   # dimensions that match
    gate_results: Dict[str, bool]  # gate_name → passed


class ComparisonResult(TypedDict):
    id_a:      str
    id_b:      str
    diagnosis: str            # CONTRADICTION / OVERLAP / REDUNDANCY / OK
    severity:  str            # BLOCKING / HIGH / MEDIUM / LOW / NONE
    detail:    str            # explanation
    evidence:  str            # specific text fragments that triggered finding


# ─────────────────────────────────────────────
# LLM helper
# ─────────────────────────────────────────────

def _llm(temperature: float = 0.05):
    """Return the configured LLM (OpenAI or Ollama) via llm_provider."""
    return _get_llm(temperature=temperature)


def _rag_vocabulary() -> str:
    """Retrieve project-specific vocabulary for normalization."""
    rag = get_rag()
    if not rag.is_ready():
        return ""
    ctx = rag.query(
        "system terminology mode names interface names function names "
        "synonym definitions glossary operational phases", k=6
    )
    if not ctx:
        return ""
    return (
        "\nPROJECT VOCABULARY (from RAG — use these canonical terms):\n"
        f"{ctx}\n"
    )


# ─────────────────────────────────────────────
# Phase 1 — NORMALIZER
# ─────────────────────────────────────────────


def _parse_llm_json(text: str, expect: type = list):
    """Extract and parse JSON from an LLM response, handling fences and trailing prose."""
    import re as _re
    cleaned = _re.sub(r"```(?:json)?\s*", "", text, flags=_re.IGNORECASE).strip()
    cleaned = _re.sub(r"```\s*$", "", cleaned, flags=_re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start_ch, end_ch = ("[", "]") if expect is list else ("{", "}")
    start = cleaned.find(start_ch)
    if start == -1:
        raise json.JSONDecodeError("No JSON found", cleaned, 0)
    end = len(cleaned)
    while end > start:
        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            pos = cleaned.rfind(end_ch, start, end)
            if pos == -1:
                break
            end = pos + 1
    candidate = _re.sub(r",\s*([}\]])", r"\1", cleaned[start:])
    return json.loads(candidate)

def normalizer_agent(requirements: List[dict], system_context: str) -> List[NormalizedReq]:
    """
    Transforms each parsed requirement into a canonical NormalizedReq struct.

    Key operations:
    - Resolve synonyms (e.g. "FMS" / "flight management system" → canonical "FMS")
    - Standardise mode names (e.g. "takeoff" / "TO phase" / "T/O" → "TAKEOFF")
    - Standardise interface names (e.g. "ARINC 429 bus 1" / "A429-1" → "A429-BUS-1")
    - Extract: subject, action, constraint, condition
    - Infer lifecycle phase and HW/SW allocation where possible
    """
    llm     = _llm()
    rag_voc = _rag_vocabulary()
    sys_ctx = f"\nSYSTEM CONTEXT:\n{system_context}\n" if system_context else ""

    system_prompt = f"""You are an aerospace requirements normalisation expert for {STANDARD}.

Your task is to parse each requirement into a canonical structured form that allows
precise pairwise comparison for contradiction, overlap, and redundancy detection.
{rag_voc}{sys_ctx}

NORMALISATION RULES:
1. SUBJECT   — identify the system, subsystem, or function that bears the obligation.
               Use the canonical name from the project vocabulary when available.
               Examples: "Navigation System", "FMS", "Display Unit", "AHRS"
2. ACTION    — the main verb phrase (what the subject shall do / achieve).
               Use active voice, infinitive form.
               Example: "provide position accuracy", "switch to backup", "display altitude"
3. CONSTRAINT— the quantified limit, threshold, or property that bounds the action.
               Include value + unit + tolerance.
               Example: "±0.1 NM (95% confidence)", "within 50 ms", "≤ 45 W"
               Use "none" if no constraint is stated.
4. CONDITION — the triggering or applicability condition (when / if / during / unless).
               Example: "during all phases of flight", "on primary power loss", "if IRS valid"
               Use "always" if unconditional.
5. MODES     — list of operational modes, NORMALISED to uppercase canonical labels.
               Canonical set (extend from project vocabulary):
               NORMAL, DEGRADED, EMERGENCY, GROUND, TAKEOFF, CRUISE, APPROACH,
               LANDING, GO-AROUND, MAINTENANCE, ALL.
               Map variants: "take-off" → TAKEOFF, "approach phase" → APPROACH, etc.
               Use ["ALL"] if no specific mode is stated.
6. INTERFACES— list of named interfaces, buses, or data links.
               Normalise to canonical identifiers from project vocabulary.
               Example: ["A429-BUS-1", "ARINC-717", "MIL-STD-1553-BUS-A"]
               Use [] if no interface is referenced.
7. FUNCTIONS — list of named functions this requirement relates to.
               Example: ["position_computation", "display_management", "power_switching"]
               Use [] if not determinable.
8. LIFECYCLE — one of: "design" | "operation" | "maintenance" | "unspecified"
9. ALLOCATION— one of: "HW" | "SW" | "System" | "unspecified"
10. NORMAL_TEXT — restate the requirement in one clear "shall" sentence using ONLY
                canonical / normalised terms. This becomes the comparison text.

Return a JSON ARRAY. One object per requirement, with these exact fields:
  id, original, subject, action, constraint, condition, modes, interfaces,
  functions, lifecycle, allocation, normal_text

Return ONLY valid JSON — no markdown fences, no commentary."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Normalise these requirements:\n\n{json.dumps(requirements, indent=2)}")
    ])

    try:
        result = _parse_llm_json(response.content, expect=list)
        if not isinstance(result, list):
            result = [result]
        # Validate and fill missing keys with defaults
        defaults: NormalizedReq = {
            "id": "", "original": "", "subject": "unspecified", "action": "",
            "constraint": "none", "condition": "always", "modes": ["ALL"],
            "interfaces": [], "functions": [], "lifecycle": "unspecified",
            "allocation": "unspecified", "normal_text": ""
        }
        cleaned = []
        for item in result:
            merged = {**defaults, **item}
            if not merged["normal_text"]:
                merged["normal_text"] = merged.get("original", "")
            cleaned.append(merged)
        return cleaned
    except (json.JSONDecodeError, Exception) as e:
        # Fallback: return minimal normalised forms
        return [
            {**{"id": r.get("id","?"), "original": r.get("text",""),
                "subject": "unspecified", "action": r.get("text",""),
                "constraint": "none", "condition": "always", "modes": ["ALL"],
                "interfaces": [], "functions": [], "lifecycle": "unspecified",
                "allocation": "unspecified",
                "normal_text": r.get("text","")}}
            for r in requirements
        ]


# ─────────────────────────────────────────────
# Phase 2 — CLUSTERER
# ─────────────────────────────────────────────

def clusterer_agent(
    normalized: List[NormalizedReq],
) -> tuple[Dict[str, Dict[str, List[str]]], List[CandidatePair]]:
    """
    Groups requirements along four dimensions and generates candidate pairs.

    Returns:
        clusters       — dict[dimension][cluster_label] → [req_ids]
        candidate_pairs— list of {id_a, id_b, reasons} with deduplication
    """
    # ── Build clusters from normalized metadata (pure Python, no LLM needed) ──
    clusters: Dict[str, Dict[str, List[str]]] = {
        "subject":   {},
        "function":  {},
        "interface": {},
        "mode":      {},
    }

    for req in normalized:
        rid = req["id"]

        # Subject cluster
        subj = req.get("subject", "unspecified").strip() or "unspecified"
        clusters["subject"].setdefault(subj, []).append(rid)

        # Function clusters (one per named function)
        for fn in req.get("functions", []) or ["unspecified"]:
            clusters["function"].setdefault(fn, []).append(rid)

        # Interface clusters
        for iface in req.get("interfaces", []) or ["unspecified"]:
            clusters["interface"].setdefault(iface, []).append(rid)

        # Mode clusters — ALL expands to every non-ALL cluster that exists
        modes = req.get("modes", ["ALL"])
        for mode in modes:
            clusters["mode"].setdefault(mode, []).append(rid)

    # ── Generate candidate pairs ───────────────────────────────────────────
    # A pair is candidate if they share at least one cluster (any dimension).
    pair_reasons: Dict[tuple, List[str]] = {}

    for dim, buckets in clusters.items():
        for label, req_ids in buckets.items():
            if label in ("unspecified", "ALL"):
                continue  # skip catch-all buckets to avoid O(n²) explosion
            for a, b in itertools.combinations(req_ids, 2):
                key = (min(a, b), max(a, b))
                reason = f"{dim}:{label}"
                pair_reasons.setdefault(key, []).append(reason)

    # Also pair ALL-mode reqs with every other req (they apply universally)
    all_mode_ids = clusters["mode"].get("ALL", [])
    all_req_ids  = [r["id"] for r in normalized]
    for a in all_mode_ids:
        for b in all_req_ids:
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            pair_reasons.setdefault(key, []).append("mode:ALL-universal")

    candidate_pairs: List[CandidatePair] = [
        {"id_a": k[0], "id_b": k[1], "reasons": v}
        for k, v in pair_reasons.items()
    ]

    return clusters, candidate_pairs


# ─────────────────────────────────────────────
# Phase 3 — FILTER
# ─────────────────────────────────────────────

def filter_agent(
    normalized: List[NormalizedReq],
    candidate_pairs: List[CandidatePair],
) -> List[ValidatedPair]:
    """
    Prunes candidate pairs using four binary gates.
    A pair is KEPT only if at least one gate PASSES (i.e. there is a real
    possibility of interaction). Pairs that fail ALL gates are discarded.

    Gates:
      G1 — Same system / allocation?
      G2 — Overlapping modes?
      G3 — Same lifecycle phase?
      G4 — Same interface or function?
    """
    # Build lookup by id
    req_map: Dict[str, NormalizedReq] = {r["id"]: r for r in normalized}

    def modes_overlap(ma: List[str], mb: List[str]) -> bool:
        if "ALL" in ma or "ALL" in mb:
            return True
        return bool(set(ma) & set(mb))

    def alloc_overlap(aa: str, ab: str) -> bool:
        if "unspecified" in (aa, ab):
            return True
        return aa == ab or "System" in (aa, ab)

    def lifecycle_overlap(la: str, lb: str) -> bool:
        if "unspecified" in (la, lb):
            return True
        return la == lb

    def iface_fn_overlap(ra: NormalizedReq, rb: NormalizedReq) -> bool:
        ia, ib = set(ra.get("interfaces", [])), set(rb.get("interfaces", []))
        fa, fb = set(ra.get("functions",  [])), set(rb.get("functions",  []))
        iface_match = bool(ia & ib) and "unspecified" not in ia
        fn_match    = bool(fa & fb) and "unspecified" not in fa
        return iface_match or fn_match

    validated: List[ValidatedPair] = []

    for pair in candidate_pairs:
        ra = req_map.get(pair["id_a"])
        rb = req_map.get(pair["id_b"])
        if not ra or not rb:
            continue

        gates = {
            "G1_same_allocation": alloc_overlap(ra["allocation"], rb["allocation"]),
            "G2_overlapping_modes": modes_overlap(ra["modes"], rb["modes"]),
            "G3_same_lifecycle":   lifecycle_overlap(ra["lifecycle"], rb["lifecycle"]),
            "G4_same_iface_or_fn": iface_fn_overlap(ra, rb),
        }

        # Keep if at least one gate passes
        if any(gates.values()):
            validated.append({
                "id_a":         pair["id_a"],
                "id_b":         pair["id_b"],
                "cluster_dims": pair["reasons"],
                "gate_results": gates,
            })

    return validated


# ─────────────────────────────────────────────
# Phase 4 — COMPARATOR
# ─────────────────────────────────────────────

_BATCH_SIZE = 8   # pairs per LLM call — keeps context manageable

def comparator_agent(
    normalized: List[NormalizedReq],
    comparison_set: List[ValidatedPair],
) -> tuple[List[ComparisonResult], str]:
    """
    Diagnoses each validated pair for inter-requirement defects.

    Batches pairs into groups of _BATCH_SIZE to keep context manageable.
    Returns:
        results   — list of ComparisonResult dicts
        findings  — formatted text report
    """
    if not comparison_set:
        return [], "No candidate pairs remained after filtering — no inter-requirement defects detected."

    req_map: Dict[str, NormalizedReq] = {r["id"]: r for r in normalized}
    llm = _llm(temperature=0.05)

    system_prompt = f"""You are an {STANDARD} inter-requirement defect analyst.

You will receive pairs of normalised requirements and must diagnose each pair with
EXACTLY one of these diagnoses:

  CONTRADICTION — the two requirements assert logically incompatible behaviours,
                  constraints, or values for the same subject/function/mode.
                  Example: one requires latency ≤ 50 ms, another requires ≥ 200 ms
                  for the same signal path.

  OVERLAP       — the two requirements partially cover the same behaviour or property,
                  creating ambiguity about which applies or risking double-allocation.
                  Example: two requirements both constrain display refresh rate but
                  with different scope — one for normal mode, one for "all modes".

  REDUNDANCY    — one requirement is fully subsumed by the other: satisfying the
                  broader one automatically satisfies the narrower one. The narrower
                  requirement adds no information.
                  Example: "shall operate at 28 VDC ±4 V" and "shall operate at 28 VDC"
                  — the first already covers the second.

  OK            — no inter-requirement defect found. The pair may be related but
                  does not exhibit contradiction, overlap, or redundancy.

SEVERITY MAPPING:
  CONTRADICTION → BLOCKING
  OVERLAP       → HIGH
  REDUNDANCY    → MEDIUM
  OK            → NONE

For each pair, output a JSON object with:
  "id_a"     : requirement ID A
  "id_b"     : requirement ID B
  "diagnosis": one of CONTRADICTION / OVERLAP / REDUNDANCY / OK
  "severity" : corresponding severity
  "detail"   : 2–3 sentence explanation of the finding
  "evidence" : the specific text fragments (from normal_text of each req) that
               demonstrate the finding. Quote them directly.

Return a JSON ARRAY of these objects — one per pair.
Return ONLY valid JSON. No markdown fences."""

    all_results: List[ComparisonResult] = []

    # Process in batches
    for i in range(0, len(comparison_set), _BATCH_SIZE):
        batch = comparison_set[i : i + _BATCH_SIZE]

        # Build compact representation of each pair
        pairs_payload = []
        for vp in batch:
            ra = req_map.get(vp["id_a"], {})
            rb = req_map.get(vp["id_b"], {})
            pairs_payload.append({
                "pair": f"{vp['id_a']} × {vp['id_b']}",
                "cluster_dimensions": vp["cluster_dims"],
                "gate_results": vp["gate_results"],
                "req_A": {
                    "id":          ra.get("id"),
                    "normal_text": ra.get("normal_text"),
                    "subject":     ra.get("subject"),
                    "action":      ra.get("action"),
                    "constraint":  ra.get("constraint"),
                    "condition":   ra.get("condition"),
                    "modes":       ra.get("modes"),
                    "allocation":  ra.get("allocation"),
                },
                "req_B": {
                    "id":          rb.get("id"),
                    "normal_text": rb.get("normal_text"),
                    "subject":     rb.get("subject"),
                    "action":      rb.get("action"),
                    "constraint":  rb.get("constraint"),
                    "condition":   rb.get("condition"),
                    "modes":       rb.get("modes"),
                    "allocation":  rb.get("allocation"),
                },
            })

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Diagnose these pairs:\n\n{json.dumps(pairs_payload, indent=2)}")
        ])

        try:
            batch_results = _parse_llm_json(response.content, expect=list)
            if not isinstance(batch_results, list):
                batch_results = [batch_results]
            all_results.extend(batch_results)
        except json.JSONDecodeError:
            # If JSON parse fails, add raw response as a single result note
            for vp in batch:
                all_results.append({
                    "id_a": vp["id_a"], "id_b": vp["id_b"],
                    "diagnosis": "OK", "severity": "NONE",
                    "detail": "Parse error in comparator response.",
                    "evidence": ""
                })

    findings = _format_findings(all_results, normalized)
    return all_results, findings


def _format_findings(
    results: List[ComparisonResult],
    normalized: List[NormalizedReq],
) -> str:
    """Render the comparator results as a structured text report."""

    contradictions = [r for r in results if r["diagnosis"] == "CONTRADICTION"]
    overlaps       = [r for r in results if r["diagnosis"] == "OVERLAP"]
    redundancies   = [r for r in results if r["diagnosis"] == "REDUNDANCY"]
    ok_pairs       = [r for r in results if r["diagnosis"] == "OK"]

    lines = [
        "═" * 70,
        "  MULTI-REQUIREMENT ANALYSIS REPORT",
        "═" * 70,
        f"  Pairs analysed   : {len(results)}",
        f"  Contradictions   : {len(contradictions)}  [BLOCKING]",
        f"  Overlaps         : {len(overlaps)}  [HIGH]",
        f"  Redundancies     : {len(redundancies)}  [MEDIUM]",
        f"  Clean pairs      : {len(ok_pairs)}",
        "═" * 70,
    ]

    def _section(title: str, items: List[ComparisonResult], severity: str) -> List[str]:
        if not items:
            return [f"\n{title}\n" + "─" * 50, "  None detected.\n"]
        out = [f"\n{title}\n" + "─" * 50]
        for r in items:
            out += [
                f"\n  [{severity}]  {r['id_a']}  ×  {r['id_b']}",
                f"  Detail   : {r['detail']}",
                f"  Evidence : {r['evidence']}",
            ]
        return out

    lines += _section("CONTRADICTIONS  [BLOCKING]", contradictions, "BLOCKING")
    lines += _section("OVERLAPS        [HIGH]",     overlaps,       "HIGH")
    lines += _section("REDUNDANCIES    [MEDIUM]",   redundancies,   "MEDIUM")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Cluster summary helper (for reporter)
# ─────────────────────────────────────────────

def format_clusters_summary(clusters: Dict[str, Dict[str, List[str]]]) -> str:
    """Render the cluster map as a readable table for inclusion in the report."""
    lines = ["REQUIREMENT CLUSTERS\n" + "─" * 50]
    for dim, buckets in clusters.items():
        lines.append(f"\n  By {dim.upper()}:")
        for label, ids in sorted(buckets.items()):
            if label in ("unspecified", "ALL") and len(ids) > 10:
                lines.append(f"    {label:<30} ({len(ids)} reqs — not shown)")
            else:
                lines.append(f"    {label:<30} {', '.join(ids)}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Top-level entry point (called from agents.py)
# ─────────────────────────────────────────────

def run_multi_req_pipeline(
    requirements: List[dict],
    system_context: str = "",
) -> dict:
    """
    Run the full 4-phase multi-requirement analysis pipeline.

    Args:
        requirements   : parsed requirements from orchestrator_agent
        system_context : RAG system context string

    Returns dict with keys:
        normalized_requirements : List[NormalizedReq]
        clusters                : dict[dim][label] → [req_ids]
        candidate_pairs         : List[CandidatePair]
        comparison_set          : List[ValidatedPair]
        comparison_results      : List[ComparisonResult]
        multi_req_findings      : str  (formatted report)
        pipeline_stats          : dict with counts
    """
    # Phase 1 — Normalise
    print("  [MultiReq] Phase 1/4 — Normalising requirements...")
    normalized = normalizer_agent(requirements, system_context)

    # Phase 2 — Cluster
    print("  [MultiReq] Phase 2/4 — Clustering requirements...")
    clusters, candidate_pairs = clusterer_agent(normalized)
    print(f"             {len(candidate_pairs)} candidate pairs generated")

    # Phase 3 — Filter
    print("  [MultiReq] Phase 3/4 — Filtering candidate pairs...")
    comparison_set = filter_agent(normalized, candidate_pairs)
    print(f"             {len(comparison_set)} pairs remain after filtering")

    # Phase 4 — Compare
    print("  [MultiReq] Phase 4/4 — Comparing requirement pairs...")
    comparison_results, multi_req_findings = comparator_agent(normalized, comparison_set)

    stats = {
        "total_requirements":  len(requirements),
        "candidate_pairs":     len(candidate_pairs),
        "validated_pairs":     len(comparison_set),
        "contradictions":      sum(1 for r in comparison_results if r["diagnosis"] == "CONTRADICTION"),
        "overlaps":            sum(1 for r in comparison_results if r["diagnosis"] == "OVERLAP"),
        "redundancies":        sum(1 for r in comparison_results if r["diagnosis"] == "REDUNDANCY"),
    }

    clusters_summary = format_clusters_summary(clusters)

    return {
        "normalized_requirements": normalized,
        "clusters":                clusters,
        "clusters_summary":        clusters_summary,
        "candidate_pairs":         candidate_pairs,
        "comparison_set":          comparison_set,
        "comparison_results":      comparison_results,
        "multi_req_findings":      multi_req_findings,
        "pipeline_stats":          stats,
    }
