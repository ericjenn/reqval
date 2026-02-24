"""
ARP4754A Requirements Validation System
Multi-Agent Architecture using LangChain + OpenAI + RAG

All validation rules are loaded dynamically from rules.json via arp4754_rules.py.
Each agent's system prompt is built from the JSON structure, including:
  - rule_id, title, objective
  - checks  (the actual steps to perform per rule)
  - failure_severity  (BLOCKING / HIGH / MEDIUM / LOW)

"""

import os
import json
import operator
import re
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from llm_provider import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from arp4754_rules import (
    format_rules_for_prompt,
    format_wording_for_prompt,
    format_ears_for_prompt,
    get_rules,
    blocking_rules,
    AMBIGUOUS_TERMS,
    AMBIGUOUS_TERMS_BY_CAT,
    WEAK_MODAL_VERBS,
    FORBIDDEN_PATTERNS,
    SENTENCE_MORPHOLOGY,
    MANDATORY_MODAL_VERB,
    EARS_PATTERNS,
    DAL_LEVELS,
    STANDARD,
    VERSION,
)
from rag import get_rag, RAGEngine
from multi_req_agents import run_multi_req_pipeline
from req_parser import load_internal, summary as req_summary

load_dotenv()


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

class ValidationState(TypedDict):
    # ── Original fields ──────────────────────────────────────────────
    requirements:             Annotated[List[dict], lambda old, new: new]
    raw_input:                str
    input_metadata:           Annotated[dict, lambda old, new: new]  # project/version/purpose from JSON header
    system_context:           str
    completeness_findings:    str
    consistency_findings:     str
    verifiability_findings:   str
    traceability_findings:    str
    correctness_findings:     str
    recommendations:          str
    final_report:             str
    rag_available:            bool
    # ── Multi-requirement pipeline fields ────────────────────────────
    normalized_requirements:  Annotated[List[dict], lambda old, new: new]  # NormalizedReq structs
    clusters_summary:         str          # human-readable cluster table
    multi_req_findings:       str          # formatted comparator report
    multi_req_stats:          Annotated[dict, lambda old, new: new]  # counts: pairs, contradictions…
    messages:                 Annotated[list, add_messages]


# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

class ValidationState(TypedDict):
    # ── Inputs ───────────────────────────────────────────────────────────
    requirements:             Annotated[List[dict], lambda old, new: new]
    raw_input:                str
    input_metadata:           Annotated[dict, lambda old, new: new]
    system_context:           str
    rag_available:            bool
    # ── Per-requirement findings (dict keyed by req_id) ──────────────────
    # Each value is the verbatim LLM output for that single requirement.
    completeness_findings:    Annotated[dict, lambda old, new: {**old, **new}]
    verifiability_findings:   Annotated[dict, lambda old, new: {**old, **new}]
    traceability_findings:    Annotated[dict, lambda old, new: {**old, **new}]
    correctness_findings:     Annotated[dict, lambda old, new: {**old, **new}]
    wording_findings:         Annotated[dict, lambda old, new: {**old, **new}]
    # ── Cross-requirement findings (single string — inherently bulk) ──────
    consistency_findings:     str
    # ── Per-requirement rewrites (dict keyed by req_id) ──────────────────
    recommendations:          Annotated[dict, lambda old, new: {**old, **new}]
    # ── Final output ─────────────────────────────────────────────────────
    final_report:             str
    # ── Multi-requirement pipeline ────────────────────────────────────────
    normalized_requirements:  Annotated[List[dict], lambda old, new: new]
    clusters_summary:         str
    multi_req_findings:       str
    multi_req_stats:          Annotated[dict, lambda old, new: new]
    messages:                 Annotated[list, add_messages]

# Progress callback hook
# ─────────────────────────────────────────────
# main.py registers a callable here before calling validate_requirements().
# Signature:  callback(event, current, total, label)
# Events:
#   "agent_start"   — agent is about to run          (label = human-readable name)
#   "agent_done"    — agent finished                 (label = human-readable name)
#   "req_progress"  — per-req progress within agent  (current/total = req counts)
#   "pair_progress" — per-batch in multi-req phase 4 (current/total = pair counts)
_progress_cb = None   # replaced by main.py via set_progress_callback()

def _emit(event: str, current: int = 0, total: int = 0, label: str = "") -> None:
    if _progress_cb is not None:
        try:
            _progress_cb(event=event, current=current, total=total, label=label)
        except Exception:
            pass  # progress errors must never abort the pipeline


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

# get_llm is imported from llm_provider — supports OpenAI and Ollama


def _parse_llm_json(text: str, expect: type = list) -> any:
    """
    Robustly extract and parse JSON from an LLM response.

    Handles the three most common LLM formatting failures:
      1. Markdown fences  — ```json ... ``` or ``` ... ```
      2. Leading/trailing prose — text before '[' or after ']' / '}'
      3. Truncated output — finds the last valid closing bracket

    Args:
        text   : raw LLM response string
        expect : expected top-level type (list or dict)

    Returns:
        Parsed Python object.

    Raises:
        json.JSONDecodeError if no valid JSON can be extracted.
    """
    # 1. Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()

    # 2. Try direct parse first (fast path)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Find the outermost JSON structure
    if expect is list:
        start_ch, end_ch = "[", "]"
    else:
        start_ch, end_ch = "{", "}"

    start = cleaned.find(start_ch)
    if start == -1:
        # Try the other bracket type as fallback
        start_ch, end_ch = ("{", "}") if expect is list else ("[", "]")
        start = cleaned.find(start_ch)
    if start == -1:
        raise json.JSONDecodeError("No JSON structure found", cleaned, 0)

    # Walk from the end to find the matching close bracket
    end = len(cleaned)
    while end > start:
        candidate = cleaned[start:end]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            end = cleaned.rfind(end_ch, start, end)
            if end == -1:
                break
            end += 1  # include the bracket itself

    # 4. Last resort: attempt common repairs
    candidate = cleaned[start:]
    # Remove trailing comma before closing bracket (common LLM mistake)
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    # Remove JavaScript-style comments
    candidate = re.sub(r"//[^\n]*\n", "\n", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    raise json.JSONDecodeError(
        f"Could not extract valid JSON from LLM response "
        f"(length={len(text)}, first 200 chars: {text[:200]!r})",
        text, 0
    )




def _rag_block(query: str, k: int = 5) -> str:
    """Query RAG and return a formatted context block, or '' if RAG not ready."""
    rag = get_rag()
    if not rag.is_ready():
        return ""
    ctx = rag.query(query, k=k)
    if not ctx:
        return ""
    return (
        "\n\n════════════════════════════════════════\n"
        "SYSTEM REFERENCE DOCUMENTS (RAG knowledge base)\n"
        "Use this project-specific context to ground your analysis in the actual\n"
        "system vocabulary, architecture, and constraints.\n"
        "════════════════════════════════════════\n"
        f"{ctx}\n"
        "════════════════════════════════════════\n"
    )


def _sys_ctx(state: ValidationState) -> str:
    ctx = state.get("system_context", "")
    return f"\nSYSTEM CONTEXT (from project documents):\n{ctx}\n" if ctx else ""


def _severity_legend() -> str:
    return (
        "SEVERITY LEVELS:\n"
        "  BLOCKING — airworthiness-critical; must be resolved before approval\n"
        "  HIGH     — significant compliance gap; must be resolved\n"
        "  MEDIUM   — compliance gap; should be resolved\n"
        "  LOW      — minor issue; recommended improvement\n"
    )


# ─────────────────────────────────────────────
# Agent 1: Orchestrator / Parser
# ─────────────────────────────────────────────

def orchestrator_agent(state: ValidationState) -> ValidationState:
    """
    Phase 0 — JSON loading (pure Python, no LLM):
      • Calls req_parser.load_internal() which reads the JSON structure,
        resolves field-name aliases, normalises verification method labels,
        and computes boolean flags (has_identifier, has_dal, modal_verb…).
      • Extracts project metadata from the "metadata" block.

    Phase 1 — Semantic enrichment (LLM):
      • Infers requirement type (functional / safety / performance / …).
      • Detects ambiguous terms in each statement.
      • Identifies compound requirements (multiple "shall" obligations).

    The LLM receives already-structured input and never needs to parse format.
    """
    llm = get_llm()
    rag = get_rag()
    rag_available = rag.is_ready()

    # ── RAG system context ────────────────────────────────────────────────
    system_context = ""
    if rag_available:
        overview = rag.query(
            "system overview architecture functions interfaces safety requirements "
            "operational concept design constraints", k=8
        )
        glossary = rag.get_glossary(max_chunks=4)
        if overview or glossary:
            system_context = (
                "=== SYSTEM OVERVIEW (from project documents) ===\n"
                f"{overview}\n\n"
                "=== PROJECT GLOSSARY & TERMINOLOGY ===\n"
                f"{glossary}"
            )

    # ── Phase 0: JSON pre-loading ─────────────────────────────────────────
    try:
        metadata, pre_loaded = load_internal(state["raw_input"])
    except (ValueError, json.JSONDecodeError) as exc:
        # Hard failure — cannot continue without valid JSON
        raise RuntimeError(
            f"Failed to parse requirements JSON: {exc}\n\n"
            "Expected format: {{\n"
            '  "metadata": {{ "project": "...", ... }},\n'
            '  "requirements": [\n'
            '    {{ "req_id": "REQ-001", "req_title": "...",\n'
            '       "req_statement": "The system shall ...",\n'
            '       "verification_method": "Test" | null,\n'
            '       "rationale": "..." }},\n'
            "    ...\n"
            "  ]\n"
            "}}"
        ) from exc

    print(f"  [Orchestrator] {req_summary(metadata, pre_loaded)}")
    _emit("agent_done", label="Orchestrator")

    # ── Phase 1: LLM semantic enrichment ─────────────────────────────────
    #TODO: the req type should be provided by a dedicated field...
    
    rag_hint       = _rag_block("system functions components interfaces naming conventions", k=4)
    ambiguous_list = ", ".join(AMBIGUOUS_TERMS)

    # Build a compact input for the LLM — only what it needs to enrich
    llm_input_items = []
    for r in pre_loaded:
        item = f"REQ-ID: {r['id']}"
        if r["title"]:
            item += f"\nTITLE:  {r['title']}"
        item += f"\nSTATEMENT: {r['text']}"
        if r["verification_method"]:
            item += f"\nVERIF: {r['verification_method']}"
        if r["rationale"]:
            item += f"\nRATIONALE: {r['rationale']}"
        llm_input_items.append(item)
    llm_input = "\n\n".join(llm_input_items)

    system_prompt = f"""You are a requirements engineering expert for {STANDARD} (v{VERSION}).

The requirements below have already been loaded from a JSON file.
All IDs, statement text, verification methods, and rationale are already extracted.

Your ONLY task is SEMANTIC ENRICHMENT — add three fields per requirement:
  1. type                  — classify as one of:
                             "functional" | "safety" | "performance" |
                             "interface" | "environmental" | "derived"
  2. ambiguous_terms_found — list any ambiguous/subjective terms found in the
                             STATEMENT from this list: {ambiguous_list}
                             (empty list [] if none)
  3. compound              — true if the STATEMENT contains more than one distinct
                             "shall" obligation; false otherwise
{rag_hint}
Return a JSON ARRAY with one object per requirement.
Each object must have EXACTLY these fields (copy id/title/text/
verification_method/rationale verbatim — do not alter them):

  "id", "title", "text", "verification_method", "rationale",
  "has_identifier", "has_verification_method", "has_source", "has_dal",
  "modal_verb", "type", "ambiguous_terms_found", "compound"

For the boolean and modal_verb fields, use the values provided in the
pre-loaded data below (do not recompute them — just copy them through).

Return ONLY valid JSON — no markdown fences, no commentary."""
    # Build pre-loaded booleans as context for the LLM to copy through
    pre_json = json.dumps([
        {
            "id":                   r["id"],
            "title":                r["title"],
            "text":                 r["text"],
            "verification_method":  r["verification_method"],
            "rationale":            r["rationale"],
            "has_identifier":       r["has_identifier"],
            "has_verification_method": r["has_verification_method"],
            "has_source":           r["has_source"],
            "has_dal":              r["has_dal"],
            "modal_verb":           r["modal_verb"],
        }
        for r in pre_loaded
    ], indent=2)

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Pre-loaded requirements (add type, ambiguous_terms_found, compound):\n\n"
            f"{pre_json}"
        ))
    ])

    try:
        requirements = _parse_llm_json(response.content, expect=list)
        if not isinstance(requirements, list):
            requirements = [requirements]
        # Safety net: guarantee all pre-loaded fields survive even if LLM dropped them
        pre_map = {r["id"]: r for r in pre_loaded}
        for req in requirements:
            pp = pre_map.get(req.get("id"), {})
            for field in ("id","title","text","verification_method","rationale",
                          "has_identifier","has_verification_method","has_source",
                          "has_dal","modal_verb"):
                req.setdefault(field, pp.get(field, ""))
            req.setdefault("type", "functional")
            req.setdefault("ambiguous_terms_found", [])
            req.setdefault("compound", False)
    except json.JSONDecodeError:
        # Fallback: use pre-loaded data directly with type defaulting to functional
        requirements = [
            {**r, "type": "functional", "ambiguous_terms_found": [], "compound": False}
            for r in pre_loaded
        ]

    return {
        "requirements":   requirements,
        "input_metadata": metadata,
        "system_context": system_context,
        "rag_available":  rag_available,
    }



# ─────────────────────────────────────────────
# Per-requirement agent helper
# ─────────────────────────────────────────────

def _run_per_req(
    state: ValidationState,
    agent_label: str,
    system_prompt_fn,          # callable(req, rag_ctx) → system_prompt str
    human_msg_fn,              # callable(req, rag_ctx) → human_message str
    result_key: str,
) -> dict:
    """
    Generic per-requirement LLM loop used by all single-req analysis agents.

    Sends one LLM call per requirement.
    Returns {result_key: {req_id: llm_response_text, ...}}.
    """
    print(f"  [{agent_label}]")
    _emit("agent_start", label=agent_label)
    llm    = get_llm()
    reqs   = state["requirements"]
    total  = len(reqs)
    out    = {}

    for idx, req in enumerate(reqs):
        rid = req.get("id", f"REQ-{idx+1}")
        _emit("req_progress", current=idx, total=total, label=agent_label)

        rag_ctx      = ""
        rag_query    = system_prompt_fn(req, "")  # first call gives us query hint
        if get_rag().is_ready():
            rag_ctx  = _rag_block(rag_query[:200], k=4)

        sys_prompt   = system_prompt_fn(req, rag_ctx)
        human_msg    = human_msg_fn(req, rag_ctx)

        response     = llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=human_msg),
        ])
        out[rid]     = response.content
        _emit("req_progress", current=idx + 1, total=total, label=agent_label)

    _emit("agent_done", label=agent_label)
    return {result_key: out}


# ─────────────────────────────────────────────
# Agent 2: Completeness  (§5.3)  — per req
# ─────────────────────────────────────────────

def completeness_agent(state: ValidationState) -> ValidationState:
    """ARP4754A §5.3 — Completeness. One LLM call per requirement."""
    rules_text = format_rules_for_prompt("completeness")

    def sys_fn(req, rag_ctx):
        if len(rag_ctx) < 10:   # called as query-hint pass
            return (
                "system functions safety requirements interface definitions "
                "environmental conditions acceptance criteria test methods"
            )
        return f"""You are an {STANDARD} §5.3 completeness auditor for aerospace systems.

{_severity_legend()}

COMPLETENESS RULES (from rules.json):
{rules_text}

{rag_ctx}
{_sys_ctx(state)}

INSTRUCTIONS — analyse THIS SINGLE REQUIREMENT only:
1. Work through each rule's CHECK STEPS explicitly.
2. For each check step: state PASSES (✓) or FAILS (✗) and why.
3. For FAILS: quote the exact problematic text and state the failure_severity.

OUTPUT FORMAT:
─────────────────────────────────────────────────
[REQ-ID]: [first 60 chars of requirement text...]
─────────────────────────────────────────────────
  REQ-XXX [SEVERITY] ✓/✗  — [explanation]

─────────────────────────────────────────────────"""

    def human_fn(req, rag_ctx):
        return f"Perform completeness analysis on this single requirement:\n\n{json.dumps(req, indent=2)}"

    return _run_per_req(state, "Completeness §5.3", sys_fn, human_fn, "completeness_findings")


# ─────────────────────────────────────────────
# Agent 3: Consistency  (§5.4)  — BULK (cross-req by nature)
# ─────────────────────────────────────────────

def consistency_agent(state: ValidationState) -> ValidationState:
    """ARP4754A §5.4 — Consistency. Bulk call — cross-requirement analysis."""
    print("  [Consistency §5.4]")
    _emit("agent_start", label="Consistency §5.4")
    llm        = get_llm()
    rules_text = format_rules_for_prompt("consistency")
    dal_text   = "\n".join(f"  DAL-{k}: {v}" for k, v in DAL_LEVELS.items())
    rag_ctx    = _rag_block(
        "DAL levels FHA failure conditions safety architecture performance "
        "budgets timing latency power consumption units", k=5
    )

    system_prompt = f"""You are an {STANDARD} §5.4 consistency auditor for aerospace systems.

{_severity_legend()}

CONSISTENCY RULES (from rules.json):
{rules_text}

DAL LEVEL DEFINITIONS:
{dal_text}
{rag_ctx}
{_sys_ctx(state)}

INSTRUCTIONS — analyse the FULL SET of requirements together:
1. REQ-K01: pairwise contradiction analysis — cite both req IDs.
2. REQ-K02: extract key terms; flag inconsistent usage across requirements.
3. REQ-K03: extract all physical quantities; verify unit and tolerance consistency.
4. REQ-K04: compare every DAL assignment against stated failure conditions.
5. REQ-K05/K06: identify timing and performance conflicts.
6. State PASSES (✓) or FAILS (✗) with failure_severity per check.

OUTPUT FORMAT:
─────────────────────────────────────────────────
REQ-K01 [SEVERITY] — Contradiction Analysis
  [pair analysis or "No contradictions found"]
REQ-K02 [SEVERITY] — Terminology Consistency
  [findings]
REQ-K03 [SEVERITY] — Units and Tolerances
  [findings]
REQ-K04 [SEVERITY] — DAL Consistency
  [findings]
REQ-K05 [SEVERITY] — Performance Feasibility
  [findings]
REQ-K06 [SEVERITY] — Timing Consistency
  [findings]
─────────────────────────────────────────────────
SUMMARY: [2-3 sentence paragraph]"""

    req_text = json.dumps(state["requirements"], indent=2)
    _emit("req_progress", current=0, total=len(state["requirements"]), label="Consistency §5.4")
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Perform consistency analysis:\n\n{req_text}"),
    ])
    _emit("req_progress", current=len(state["requirements"]),
          total=len(state["requirements"]), label="Consistency §5.4")
    _emit("agent_done", label="Consistency §5.4")
    return {"consistency_findings": response.content}


# ─────────────────────────────────────────────
# Agent 4: Verifiability  (§5.5)  — per req
# ─────────────────────────────────────────────

def verifiability_agent(state: ValidationState) -> ValidationState:
    """ARP4754A §5.5 — Verifiability. One LLM call per requirement."""
    rules_text = format_rules_for_prompt("verifiability")

    def sys_fn(req, rag_ctx):
        if len(rag_ctx) < 10:
            return (
                "test plan verification methods acceptance criteria performance thresholds "
                "measurement tolerances testability analysis inspection demonstration"
            )
        return f"""You are an {STANDARD} §5.5 verifiability auditor for aerospace systems.

{_severity_legend()}

VERIFIABILITY RULES (from rules.json):
{rules_text}

VERIFICATION METHODS defined in {STANDARD}:
  T — Test: formal testing with stimuli and measured response
  A — Analysis: mathematical / logical proof or simulation
  I — Inspection: visual or physical examination
  D — Demonstration: operational demonstration without precise measurement
{rag_ctx}
{_sys_ctx(state)}

INSTRUCTIONS — analyse THIS SINGLE REQUIREMENT only:
1. REQ-V01: state Yes / Partially / No for verifiability and why.
2. REQ-V02: identify numeric thresholds present; flag vague performance statements.
3. REQ-V03: list any subjective terms; confirm measurable criteria exist.
4. REQ-V04: assign the most appropriate method(s): T / A / I / D.
5. REQ-V05: assess feasibility; flag any unrealistic verification demands.
6. REQ-V06: detect impossible states or undefined references.
7. State PASSES (✓) or FAILS (✗) with failure_severity per rule.
8. Give a per-requirement verifiability score (0–100).

OUTPUT FORMAT:
─────────────────────────────────────────────────
[REQ-ID]: [first 60 chars...]
─────────────────────────────────────────────────
  REQ-V01 [SEVERITY] ✓/✗  — [verifiable? explanation]
  REQ-V02 [SEVERITY] ✓/✗  — [threshold analysis]
  REQ-V03 [SEVERITY] ✓/✗  — [subjective terms found or none]
  REQ-V04 [SEVERITY] ✓/✗  — [recommended method: T/A/I/D]
  REQ-V05 [SEVERITY] ✓/✗  — [feasibility assessment]
  REQ-V06 [SEVERITY] ✓/✗  — [impossible conditions check]
─────────────────────────────────────────────────"""

    def human_fn(req, rag_ctx):
        return f"Perform verifiability analysis on this single requirement:\n\n{json.dumps(req, indent=2)}"

    return _run_per_req(state, "Verifiability §5.5", sys_fn, human_fn, "verifiability_findings")


# ─────────────────────────────────────────────
# Agent 5: Traceability  (§5.6)  — per req
# ─────────────────────────────────────────────

def traceability_agent(state: ValidationState) -> ValidationState:
    """ARP4754A §5.6 — Traceability. One LLM call per requirement."""
    rules_text = format_rules_for_prompt("traceability")

    def sys_fn(req, rag_ctx):
        if len(rag_ctx) < 10:
            return (
                "FHA failure conditions PSSA SSA parent requirements regulatory basis "
                "hardware software allocation derived requirements justification"
            )
        return f"""You are an {STANDARD} §5.6 traceability auditor for aerospace systems.

{_severity_legend()}

TRACEABILITY RULES (from rules.json):
{rules_text}

TRACEABILITY HIERARCHY:
  Aircraft Level → System Level → Item Level (HW / SW)
  Safety Assessments: FHA → PSSA → SSA
{rag_ctx}
{_sys_ctx(state)}

INSTRUCTIONS — analyse THIS SINGLE REQUIREMENT only:
1. REQ-T01: identify whether an upstream source is stated in the text.
2. REQ-T02: for derived requirements, verify a justification link exists.
3. REQ-T03: assess whether downward traceability to design artifacts is implied.
4. REQ-T04: determine whether HW or SW allocation is specified or inferable.
5. REQ-T05: for safety requirements, verify a FHA failure condition is referenced.
6. State PASSES (✓) or FAILS (✗) with failure_severity per rule.

Note: assess only what is PRESENT IN THE TEXT.

OUTPUT FORMAT:
─────────────────────────────────────────────────
[REQ-ID]: [first 60 chars...]
─────────────────────────────────────────────────
  REQ-T01 [SEVERITY] ✓/✗  — [upstream source found / missing]
  REQ-T02 [SEVERITY] ✓/✗  — [derivation justification]
  REQ-T03 [SEVERITY] ✓/✗  — [downward trace]
  REQ-T04 [SEVERITY] ✓/✗  — [HW/SW allocation]
  REQ-T05 [SEVERITY] ✓/✗  — [FHA failure condition reference]
─────────────────────────────────────────────────"""

    def human_fn(req, rag_ctx):
        return f"Perform traceability analysis on this single requirement:\n\n{json.dumps(req, indent=2)}"

    return _run_per_req(state, "Traceability §5.6", sys_fn, human_fn, "traceability_findings")


# ─────────────────────────────────────────────
# Agent 6: Correctness  (§5.2)  — per req
# ─────────────────────────────────────────────

def correctness_agent(state: ValidationState) -> ValidationState:
    """ARP4754A §5.2 — Correctness. One LLM call per requirement."""
    rules_text = format_rules_for_prompt("correctness")

    def sys_fn(req, rag_ctx):
        if len(rag_ctx) < 10:
            return (
                "system functions capabilities what the system does operational modes "
                "design decisions assumptions constraints"
            )
        return f"""You are an {STANDARD} §5.2 correctness auditor for aerospace systems.

{_severity_legend()}

CORRECTNESS RULES (from rules.json):
{rules_text}
{rag_ctx}
{_sys_ctx(state)}

INSTRUCTIONS — Verify the satisfaction of each of the previous rules 
1. Check each rule. The rule passes if all checks pass/
2. State PASSES (✓) or FAILS (✗) with failure_severity per rule.

OUTPUT FORMAT:
─────────────────────────────────────────────────
[REQ-ID]: [first 60 chars...]
─────────────────────────────────────────────────
  REQ-XXX [SEVERITY] ✓/✗  — [explanation]
─────────────────────────────────────────────────"""

    def human_fn(req, rag_ctx):
        return f"Perform correctness analysis on this single requirement:\n\n{json.dumps(req, indent=2)}"

    return _run_per_req(state, "Correctness §5.2", sys_fn, human_fn, "correctness_findings")


# ─────────────────────────────────────────────
# Agent 7: Wording & Morphology  — per req  (NEW)
# ─────────────────────────────────────────────

def wording_agent(state: ValidationState) -> ValidationState:
    """
    Dedicated wording and sentence morphology agent.
    Checks all rules from wording_rules.json per requirement:
      • Ambiguous / unverifiable terms (all 11 categories)
      • Weak modal verbs
      • Forbidden patterns
      • Sentence morphology bad practices (all 7 categories)
      • Structural rules WORD-S01 through WORD-S06
    One LLM call per requirement.
    """
    wording_ref = format_wording_for_prompt()

    def sys_fn(req, rag_ctx):
        if len(rag_ctx) < 10:
            return "wording requirements vocabulary ambiguous terms morphology"
        return f"""You are an {STANDARD} wording and sentence morphology auditor.

{_severity_legend()}

{wording_ref}
{rag_ctx}

INSTRUCTIONS — Analyse THIS SINGLE REQUIREMENT for wording and morphology only.
Work through EVERY category given before.

OUTPUT FORMAT — use EXACTLY this structure:
─────────────────────────────────────────────────
[REQ-ID]: [first 60 chars of text...]
─────────────────────────────────────────────────
WEAK MODALS:
  REQ-R03 [LOW] ✓/✗  — [terms found, or "none"]

AMBIGUOUS TERMS:
  REQ-C02 [HIGH] ✓/✗  — [term "X" [category]: description, or "none"]
  (one line per term found; ✓ if none in category)

FORBIDDEN PATTERNS:
  [pattern] [SEVERITY] ✓/✗  — [occurrence quoted, or "none"]

MORPHOLOGY:
  negation_issues           [SEVERITY] ✓/✗  — [finding or "none"]
  structural_complexity     [SEVERITY] ✓/✗  — [finding or "none"]
  ambiguity_prone_structures [SEVERITY] ✓/✗  — [finding or "none"]
  passive_voice             [SEVERITY] ✓/✗  — [finding or "none"]
  modality_issues           [SEVERITY] ✓/✗  — [finding or "none"]
  logical_issues            [SEVERITY] ✓/✗  — [finding or "none"]
  vagueness_in_structure    [SEVERITY] ✓/✗  — [finding or "none"]

STRUCTURAL:
  WORD-S01 [SEVERITY] ✓/✗  — [finding or "pass"]
  WORD-S02 [SEVERITY] ✓/✗  — [finding or "pass"]
  WORD-S03 [SEVERITY] ✓/✗  — [finding or "pass"]
  WORD-S04 [SEVERITY] ✓/✗  — [finding or "pass"]
  WORD-S05 [SEVERITY] ✓/✗  — [finding or "pass"]
  WORD-S06 [SEVERITY] ✓/✗  — [finding or "pass"]

WORDING SCORE: XX/100
─────────────────────────────────────────────────"""

    def human_fn(req, rag_ctx):
        return (
            f"Perform wording and morphology analysis on this single requirement:\n\n"
            f"{json.dumps(req, indent=2)}"
        )

    return _run_per_req(state, "Wording & Morphology", sys_fn, human_fn, "wording_findings")


# ─────────────────────────────────────────────
# Agent 8: Recommender  — per req
# ─────────────────────────────────────────────

def recommender_agent(state: ValidationState) -> ValidationState:
    """
    Produces per-requirement corrected rewrites.
    Receives per-req findings dicts; passes only the relevant req's findings
    to each LLM call so context stays tight.
    Returns recommendations: Dict[req_id, rewrite_text].
    """
    print("  [Recommendation creation]")
    _emit("agent_start", label="Recommender")
    llm = get_llm(temperature=0.15)
    rag = get_rag()

    # Build compact rule reference
    all_rules_summary = []
    for cat in ("correctness", "completeness", "consistency", "verifiability", "traceability"):
        for rule in get_rules(cat):
            rid      = rule["rule_id"]
            title    = rule["title"]
            severity = rule["failure_severity"]
            checks   = "; ".join(rule.get("checks", []))
            all_rules_summary.append(f"  {rid} [{severity}] {title}: {checks}")
    rules_ref   = "\n".join(all_rules_summary)
    ears_ref    = format_ears_for_prompt()
    wording_ref = format_wording_for_prompt()

    system_prompt = f"""You are a senior aerospace requirements engineer rewriting
non-compliant requirements to conform to {STANDARD} (v{VERSION}).

COMPLETE RULES REFERENCE:
{rules_ref}

{wording_ref}

{ears_ref}

REWRITING RULES — apply ALL of the following:

1. EARS PATTERN SELECTION — choose the most appropriate pattern:
   • No trigger/state/condition present   → Ubiquitous
   • Discrete event or stimulus           → Event-Driven  (WHEN)
   • Fault, failure, off-nominal event    → Unwanted Behavior  (IF … THEN)
   • Continuous operating state or mode   → State-Driven  (WHILE)
   • Optional feature or configuration    → Optional Feature  (WHERE)
   • Multiple triggers / states           → Complex
   Do NOT default to Ubiquitous when a trigger, state, fault scenario,
   or applicability condition is present or implied.

2. MODAL VERB — use "shall" for every binding obligation.
3. ONE OBLIGATION PER STATEMENT — split compound requirements into -A, -B, … each with own EARS pattern.
4. MEASURABLE CRITERIA — replace every ambiguous/vague term with numeric value + unit + tolerance.
5. SENTENCE MORPHOLOGY — positive active voice, named subject, ≤30 words, no open-ended lists.
6. WHAT NOT HOW — behaviour not implementation.
7. PERFORMANCE VALUES — numeric value + unit + tolerance for every criterion.
8. SAFETY — reference DAL level for safety-critical requirements.
9. VERIFICATION — append [Verification: Test/Analysis/Inspection/Demonstration].
10. VOCABULARY — use exact terminology from RAG project documents where available.

OUTPUT FORMAT — produce EXACTLY this block:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIREMENT: [ID]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ORIGINAL:
  [verbatim original text]

EARS PATTERN SELECTED: [pattern name] — [one-sentence justification]

VIOLATIONS ADDRESSED:
  • [rule_id / WORD-Sxx / morphology_category] [SEVERITY] — [violation description]

CORRECTED REWRITE:
  [ID][-A/-B/…]: [EARS-structured statement] [Verification: T/A/I/D]

CHANGES EXPLAINED:
  • "[original wording]" → "[new wording]"  (fixes [rule_id]: [check])

RAG VOCABULARY USED:
  • [term from project docs, or "none"]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If fully compliant: ✓ [ID]: COMPLIANT — [EARS pattern] already applied."""

    reqs    = state["requirements"]
    total   = len(reqs)
    out     = {}

    # Per-req findings dicts from all agents
    comp_f   = state.get("completeness_findings",  {})
    verif_f  = state.get("verifiability_findings", {})
    trace_f  = state.get("traceability_findings",  {})
    corr_f   = state.get("correctness_findings",   {})
    word_f   = state.get("wording_findings",       {})
    consist  = state.get("consistency_findings",   "")

    for idx, req in enumerate(reqs):
        rid = req.get("id", f"REQ-{idx+1}")
        _emit("req_progress", current=idx, total=total, label="Recommender")

        # RAG context for this specific requirement
        req_rag_ctx = ""
        if rag.is_ready():
            query = (
                f"{req.get('text', '')} "
                f"{req.get('type', '')} performance values thresholds terminology"
            )
            req_rag_ctx = _rag_block(query, k=4)

        # Gather only THIS requirement's findings from each agent
        req_findings = (
            f"=== COMPLETENESS (§5.3) ===\n{comp_f.get(rid, '(not analysed)')}\n\n"
            f"=== VERIFIABILITY (§5.5) ===\n{verif_f.get(rid, '(not analysed)')}\n\n"
            f"=== TRACEABILITY (§5.6) ===\n{trace_f.get(rid, '(not analysed)')}\n\n"
            f"=== CORRECTNESS (§5.2) ===\n{corr_f.get(rid, '(not analysed)')}\n\n"
            f"=== WORDING & MORPHOLOGY ===\n{word_f.get(rid, '(not analysed)')}\n\n"
            f"=== CONSISTENCY (§5.4 — cross-req) ===\n{consist}"
        )

        user_message = (
            f"REQUIREMENT:\n{json.dumps(req, indent=2)}\n\n"
            f"{req_rag_ctx}"
            f"SYSTEM CONTEXT:\n{state.get('system_context', 'N/A')}\n\n"
            f"AUDIT FINDINGS FOR {rid}:\n{req_findings}"
        )

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ])
        out[rid] = response.content
        _emit("req_progress", current=idx + 1, total=total, label="Recommender")

    _emit("agent_done", label="Recommender")
    return {"recommendations": out}


# ─────────────────────────────────────────────
# Agent 9: Reporter
# ─────────────────────────────────────────────

def reporter_agent(state: ValidationState) -> ValidationState:
    """
    Produces the final compliance report with NO synthesis — pure assembly.

    Report layout:
      Section 1 — ANALYSIS METADATA + EXECUTIVE SUMMARY + SCORES  (LLM, concise)
      Section 2 — PER-REQUIREMENT FULL AUDIT REPORT  (Python, verbatim, one block per req)
                    For each req:
                      ORIGINAL TEXT
                      COMPLETENESS findings   (verbatim from completeness_agent)
                      VERIFIABILITY findings  (verbatim from verifiability_agent)
                      TRACEABILITY findings   (verbatim from traceability_agent)
                      CORRECTNESS findings    (verbatim from correctness_agent)
                      WORDING & MORPHOLOGY    (verbatim from wording_agent)
                      REWRITE PROPOSAL        (verbatim from recommender_agent)
      Section 3 — CONSISTENCY FINDINGS  (verbatim, bulk — cross-req)
      Section 4 — MULTI-REQUIREMENT ANALYSIS  (verbatim)
      Section 5 — REQUIREMENT CLUSTERS  (verbatim)
    """
    print("  [Report creation]")
    llm = get_llm(temperature=0.2)
    rag = get_rag()

    rag_status = (
        f"RAG Vector Store  : ACTIVE — {rag.chunk_count()} chunks indexed\n"
        f"Indexed sources   : {', '.join(rag.list_sources()) or 'none'}"
        if state.get("rag_available")
        else "RAG Vector Store  : NOT LOADED (generic analysis)"
    )
    blocking_ids = []
    for cat in ("completeness", "consistency", "verifiability", "traceability", "correctness"):
        blocking_ids.extend(r["rule_id"] for r in blocking_rules(cat))

    n_reqs = len(state.get("requirements", []))
    meta   = state.get("input_metadata", {})

    # ── Section 1: LLM writes only metadata + summary + scores ──────────
    llm_system = f"""You are a senior aerospace systems engineer writing ONLY the
summary header of a formal {STANDARD} Requirements Validation Report (rules v{VERSION}).

{rag_status}
BLOCKING rules: {', '.join(blocking_ids)}

Write ONLY the sections below. Be concise. Reference specific Req IDs and Rule IDs.
The full per-requirement detail follows in code — do NOT repeat individual findings.

══════════════════════════════════════════════════════════════════════
          {STANDARD} REQUIREMENTS VALIDATION REPORT
══════════════════════════════════════════════════════════════════════

ANALYSIS METADATA
─────────────────
  Standard              : {STANDARD} v{VERSION}
  Project               : {meta.get("project", "—")}
  Purpose               : {meta.get("purpose", "—")}
  Requirements analysed : {n_reqs}
  RAG knowledge base    : [ACTIVE — N chunks / NOT LOADED]
  Indexed source docs   : [list or "none"]

EXECUTIVE SUMMARY
─────────────────
[3–4 sentences: overall compliance posture, the 2–3 most critical issues
(cite Req ID + Rule ID), validation status.]

OVERALL COMPLIANCE SCORE: XX/100
  Weighted: Completeness 25% + Verifiability 25% + Consistency 20%
          + Correctness 15% + Traceability 15%

COMPLIANCE BREAKDOWN
────────────────────
  §5.2 Correctness        : XX/100
  §5.3 Completeness       : XX/100
  §5.4 Consistency        : XX/100
  §5.5 Verifiability      : XX/100
  §5.6 Traceability       : XX/100
  Wording & Morphology    : XX/100

BLOCKING FINDINGS  (must resolve before approval)
──────────────────────────────────────────────────
  [Req ID] | [Rule ID] | [one-line description]   (or "None detected")

CRITICAL FINDINGS  (HIGH severity)
───────────────────────────────────
  [Req ID] | [Rule ID] | [one-line description]

MAJOR FINDINGS  (MEDIUM severity)
──────────────────────────────────
  [Req ID] | [Rule ID] | [one-line description]

MINOR FINDINGS  (LOW severity)
────────────────────────────────
  [Req ID] | [Rule ID] | [one-line description]

PRIORITISED ACTION PLAN
────────────────────────
[Numbered list BLOCKING → HIGH → MEDIUM → LOW.
 Each: N. [Priority] [Req ID] [Rule ID] — one-line action]

VALIDATION STATUS: [PASS / CONDITIONAL PASS / FAIL]
[Two-sentence justification.]"""

    # Feed the per-req dicts as summaries to the LLM (for scoring)
    reqs       = state.get("requirements", [])
    comp_f     = state.get("completeness_findings",  {})
    verif_f    = state.get("verifiability_findings", {})
    trace_f    = state.get("traceability_findings",  {})
    corr_f     = state.get("correctness_findings",   {})
    word_f     = state.get("wording_findings",       {})
    consist    = state.get("consistency_findings",   "")
    recs       = state.get("recommendations",        {})

    # Compact summary for the LLM (just scores + violations, not full text)
    findings_summary = ""
    for req in reqs:
        rid = req.get("id", "")
        findings_summary += f"\n--- {rid} ---\n"
        for label, fd in [
            ("Completeness", comp_f), ("Verifiability", verif_f),
            ("Traceability", trace_f), ("Correctness", corr_f),
            ("Wording", word_f),
        ]:
            block = fd.get(rid, "")
            # Extract just score lines and ✗ lines for the summary
            score_lines = [l for l in block.splitlines()
                           if "Score:" in l or "✗" in l or "SCORE:" in l.upper()]
            findings_summary += f"  [{label}] " + " | ".join(score_lines[:6]) + "\n"

    llm_user = (
        f"CONSISTENCY (§5.4 — bulk):\n{consist}\n\n"
        f"MULTI-REQ:\n{state.get('multi_req_findings', 'Not run.')}\n\n"
        f"PER-REQ FINDINGS SUMMARY:\n{findings_summary}"
    )

    summary_block = llm.invoke([
        SystemMessage(content=llm_system),
        HumanMessage(content=llm_user),
    ]).content.rstrip()

    # ── Section 2: Per-requirement full audit report (Python, verbatim) ──
    W   = 70
    SEP = "━" * W
    DIV = "\n" + "═" * W + "\n"
    HR  = "─" * W

    per_req_lines = [
        "═" * W,
        "  PER-REQUIREMENT FULL AUDIT REPORT",
        "  (verbatim agent outputs — no synthesis, no omissions)",
        "═" * W,
    ]

    for req in reqs:
        rid      = req.get("id", "?")
        orig_txt = req.get("text", "")
        rewrite  = recs.get(rid, "")

        per_req_lines += [
            "",
            SEP,
            f"  REQUIREMENT: {rid}",
            SEP,
            f"  ORIGINAL TEXT:",
            f"    {orig_txt}",
            "",
        ]

        # Five per-req agent outputs verbatim
        for section_label, findings_dict in [
            ("§5.3 COMPLETENESS",      comp_f),
            ("§5.5 VERIFIABILITY",     verif_f),
            ("§5.6 TRACEABILITY",      trace_f),
            ("§5.2 CORRECTNESS",       corr_f),
            ("WORDING & MORPHOLOGY",   word_f),
        ]:
            block = findings_dict.get(rid, "").strip()
            per_req_lines.append(f"  ┌─ {section_label}")
            if block:
                for line in block.splitlines():
                    per_req_lines.append("  │  " + line)
            else:
                per_req_lines.append("  │  (no findings)")
            per_req_lines.append("  └" + "─" * (W - 3))
            per_req_lines.append("")

        # Rewrite proposal verbatim
        per_req_lines.append("  ┌─ REWRITE PROPOSAL")
        if rewrite.strip():
            for line in rewrite.strip().splitlines():
                per_req_lines.append("  │  " + line)
        else:
            per_req_lines.append("  │  (no rewrite generated)")
        per_req_lines.append("  └" + "─" * (W - 3))

    per_req_block = "\n".join(per_req_lines)

    # ── Section 3: Consistency (bulk) ────────────────────────────────────
    consistency_block = (
        DIV
        + "  §5.4 CONSISTENCY FINDINGS  (cross-requirement — bulk analysis)\n"
        + "═" * W + "\n"
        + (consist or "Not run.")
    )

    # ── Section 4: Multi-req findings ────────────────────────────────────
    multi_block = (
        DIV
        + "  MULTI-REQUIREMENT ANALYSIS  (contradictions / overlaps / redundancies)\n"
        + "═" * W + "\n"
        + (state.get("multi_req_findings") or "Not run.")
    )

    # ── Section 5: Clusters ───────────────────────────────────────────────
    clusters_block = (
        DIV
        + "  REQUIREMENT CLUSTERS\n"
        + "═" * W + "\n"
        + (state.get("clusters_summary") or "Not run.")
    )

    final_report = (
        summary_block
        + "\n\n" + per_req_block
        + consistency_block
        + multi_block
        + clusters_block
        + "\n" + "═" * W
    )

    return {"final_report": final_report}



# ─────────────────────────────────────────────
# Multi-Requirement Analysis Node
# ─────────────────────────────────────────────

def multi_req_agent(state: ValidationState) -> ValidationState:
    """
    LangGraph node wrapping the full 4-phase multi-requirement pipeline.
    Runs after all single-requirement agents.
    """
    _emit("agent_start", label="Multi-req analysis")
    result = run_multi_req_pipeline(
        requirements=state["requirements"],
        system_context=state.get("system_context", ""),
        progress_cb=_progress_cb,
    )
    _emit("agent_done", label="Multi-req analysis")
    return {
        "normalized_requirements": result["normalized_requirements"],
        "clusters_summary":        result["clusters_summary"],
        "multi_req_findings":      result["multi_req_findings"],
        "multi_req_stats":         result["pipeline_stats"],
    }


# ─────────────────────────────────────────────
# Graph Assembly
# ─────────────────────────────────────────────

def build_validation_graph() -> StateGraph:
    """LangGraph workflow — one node per agent."""
    workflow = StateGraph(ValidationState)

    workflow.add_node("orchestrator",  orchestrator_agent)
    workflow.add_node("completeness",  completeness_agent)
    workflow.add_node("consistency",   consistency_agent)
    workflow.add_node("verifiability", verifiability_agent)
    workflow.add_node("traceability",  traceability_agent)
    workflow.add_node("correctness",   correctness_agent)
    workflow.add_node("wording",       wording_agent)       # ← NEW
    workflow.add_node("multi_req",     multi_req_agent)
    workflow.add_node("recommender",   recommender_agent)
    workflow.add_node("reporter",      reporter_agent)

    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator",  "completeness")
    workflow.add_edge("completeness",  "consistency")
    workflow.add_edge("consistency",   "verifiability")
    workflow.add_edge("verifiability", "traceability")
    workflow.add_edge("traceability",  "correctness")
    workflow.add_edge("correctness",   "wording")           # ← NEW edge
    workflow.add_edge("wording",       "multi_req")
    workflow.add_edge("multi_req",     "recommender")
    workflow.add_edge("recommender",   "reporter")
    workflow.add_edge("reporter",      END)

    return workflow.compile()


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def set_progress_callback(cb) -> None:
    """Register a progress callback before calling validate_requirements."""
    global _progress_cb
    _progress_cb = cb


def validate_requirements(requirements_text: str) -> dict:
    """
    Main entry point. Runs the full multi-agent validation pipeline.

    Args:
        requirements_text: JSON string in canonical format:
                           { "metadata": {...}, "requirements": [{...}, ...] }

    Returns:
        Full state dict including all agent findings and the final report.
    """
    graph = build_validation_graph()

    initial_state: ValidationState = {
        "raw_input":               requirements_text,
        "input_metadata":          {},
        "requirements":            [],
        "system_context":          "",
        "rag_available":           get_rag().is_ready(),
        # Per-req findings dicts — start empty, agents merge into them
        "completeness_findings":   {},
        "verifiability_findings":  {},
        "traceability_findings":   {},
        "correctness_findings":    {},
        "wording_findings":        {},
        # Bulk findings
        "consistency_findings":    "",
        # Per-req rewrites
        "recommendations":         {},
        # Final output
        "final_report":            "",
        # Multi-requirement
        "normalized_requirements": [],
        "clusters_summary":        "",
        "multi_req_findings":      "",
        "multi_req_stats":         {},
        "messages":                [],
    }

    return graph.invoke(initial_state)
