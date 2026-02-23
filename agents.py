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
    get_rules,
    blocking_rules,
    AMBIGUOUS_TERMS,
    WEAK_MODAL_VERBS,
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
# Agent 2: Completeness  (§5.3)
# ─────────────────────────────────────────────

def completeness_agent(state: ValidationState) -> ValidationState:
    """ARP4754A §5.3 — Completeness. Rules and checks loaded from rules.json."""
    print("  [Completeness]")
    llm        = get_llm()
    rules_text = format_rules_for_prompt("completeness")
    ambiguous  = ", ".join(AMBIGUOUS_TERMS)
    rag_ctx    = _rag_block(
        "system functions safety requirements interface definitions "
        "environmental conditions acceptance criteria test methods", k=5
    )

    system_prompt = f"""You are an {STANDARD} §5.3 completeness auditor for aerospace systems.

{_severity_legend()}

COMPLETENESS RULES (from rules.json):
{rules_text}

AMBIGUOUS TERMS TO DETECT: {ambiguous}
{rag_ctx}
{_sys_ctx(state)}

INSTRUCTIONS — for EACH requirement, execute the checks listed under every rule:
1. Work through each rule's CHECK STEPS explicitly.
2. For each check step: state whether it PASSES (✓) or FAILS (✗) and why.
3. For FAILS: quote the exact problematic text and state the failure_severity.
4. Give a per-requirement completeness score (0-100%).

OUTPUT FORMAT per requirement:
─────────────────────────────────────────────────
[REQ-ID]: [first 60 chars of requirement text...]
─────────────────────────────────────────────────
  REQ-C01 [SEVERITY] ✓/✗  — [explanation + check results]
  REQ-C02 [SEVERITY] ✓/✗  — [explanation + check results]
  REQ-C03 [SEVERITY] ✓/✗  — [explanation]
  REQ-C04 [SEVERITY] ✓/✗  — [explanation]
  REQ-C05 [SEVERITY] ✓/✗  — [explanation]
  REQ-C06 [SEVERITY] ✓/✗  — [explanation]
  REQ-C07 [SEVERITY] ✓/✗  — [explanation]
  REQ-C08 [SEVERITY] ✓/✗  — [explanation]
  REQ-C09 [SEVERITY] ✓/✗  — [explanation]
  REQ-C10 [SEVERITY] ✓/✗  — [explanation]
  Ambiguous terms found: [list or "none"]
  Score: XX/100
─────────────────────────────────────────────────

End with:
OVERALL COMPLETENESS SCORE: XX/100
SUMMARY: [2-3 sentence paragraph]"""

    req_text = json.dumps(state["requirements"], indent=2)
    response  = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Perform completeness analysis:\n\n{req_text}")
    ])
    return {"completeness_findings": response.content}


# ─────────────────────────────────────────────
# Agent 3: Consistency  (§5.4)
# ─────────────────────────────────────────────

def consistency_agent(state: ValidationState) -> ValidationState:
    """ARP4754A §5.4 — Consistency. Rules and checks loaded from rules.json."""
    print("  [Consistency]")
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
1. For each rule, execute every listed CHECK STEP across all requirements.
2. REQ-K01: perform pairwise contradiction analysis — cite both requirement IDs.
3. REQ-K02: extract key terms and flag inconsistent usage across requirements.
4. REQ-K03: extract all physical quantities; verify unit and tolerance consistency.
5. REQ-K04: compare every DAL assignment against stated failure conditions.
6. REQ-K05/K06: identify timing and performance conflicts.
7. State PASSES (✓) or FAILS (✗) per check, with failure_severity.

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
OVERALL CONSISTENCY SCORE: XX/100
SUMMARY: [2-3 sentence paragraph]"""

    req_text = json.dumps(state["requirements"], indent=2)
    response  = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Perform consistency analysis:\n\n{req_text}")
    ])
    return {"consistency_findings": response.content}


# ─────────────────────────────────────────────
# Agent 4: Verifiability  (§5.5)
# ─────────────────────────────────────────────

def verifiability_agent(state: ValidationState) -> ValidationState:
    """ARP4754A §5.5 — Verifiability. Rules and checks loaded from rules.json."""
    print("  [Verifiability]")
    llm        = get_llm()
    rules_text = format_rules_for_prompt("verifiability")
    rag_ctx    = _rag_block(
        "test plan verification methods acceptance criteria performance thresholds "
        "measurement tolerances testability analysis inspection demonstration", k=5
    )

    system_prompt = f"""You are an {STANDARD} §5.5 verifiability auditor for aerospace systems.

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

INSTRUCTIONS — for EACH requirement, execute every check step:
1. REQ-V01: state Yes / Partially / No for verifiability and why.
2. REQ-V02: identify numeric thresholds present; flag vague performance statements.
3. REQ-V03: list any subjective terms; confirm measurable criteria exist.
4. REQ-V04: assign the most appropriate method(s): T / A / I / D.
5. REQ-V05: assess feasibility; flag any unrealistic verification demands.
6. REQ-V06: detect impossible states or undefined references.
7. State PASSES (✓) or FAILS (✗) with failure_severity per rule.
8. Give a per-requirement verifiability score (0-100%).

OUTPUT FORMAT per requirement:
─────────────────────────────────────────────────
[REQ-ID]: [first 60 chars...]
─────────────────────────────────────────────────
  REQ-V01 [SEVERITY] ✓/✗  — [verifiable? explanation]
  REQ-V02 [SEVERITY] ✓/✗  — [threshold analysis]
  REQ-V03 [SEVERITY] ✓/✗  — [subjective terms found or none]
  REQ-V04 [SEVERITY] ✓/✗  — [recommended method: T/A/I/D]
  REQ-V05 [SEVERITY] ✓/✗  — [feasibility assessment]
  REQ-V06 [SEVERITY] ✓/✗  — [impossible conditions check]
  Score: XX/100
─────────────────────────────────────────────────

OVERALL VERIFIABILITY SCORE: XX/100
SUMMARY: [2-3 sentence paragraph]"""

    req_text = json.dumps(state["requirements"], indent=2)
    response  = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Perform verifiability analysis:\n\n{req_text}")
    ])
    return {"verifiability_findings": response.content}


# ─────────────────────────────────────────────
# Agent 5: Traceability  (§5.6)
# ─────────────────────────────────────────────

def traceability_agent(state: ValidationState) -> ValidationState:
    """ARP4754A §5.6 — Traceability. Rules and checks loaded from rules.json."""
    print("  [Traceability]")    
    llm        = get_llm()
    rules_text = format_rules_for_prompt("traceability")
    rag_ctx    = _rag_block(
        "FHA failure conditions PSSA SSA parent requirements regulatory basis "
        "hardware software allocation derived requirements justification", k=5
    )

    system_prompt = f"""You are an {STANDARD} §5.6 traceability auditor for aerospace systems.

{_severity_legend()}

TRACEABILITY RULES (from rules.json):
{rules_text}

TRACEABILITY HIERARCHY:
  Aircraft Level → System Level → Item Level (HW / SW)
  ↕ bidirectional
  Safety Assessments: FHA → PSSA → SSA
{rag_ctx}
{_sys_ctx(state)}

INSTRUCTIONS — for EACH requirement, execute every check step:
1. REQ-T01: identify whether an upstream source is stated in the text.
2. REQ-T02: for derived requirements, verify a justification link exists.
3. REQ-T03: assess whether downward traceability to design artifacts is implied.
4. REQ-T04: determine whether HW or SW allocation is specified or inferable.
5. REQ-T05: for safety requirements, verify a FHA failure condition is referenced.
6. State PASSES (✓) or FAILS (✗) with failure_severity per rule.
7. Give a per-requirement traceability score (0-100%).

Note: assess only what is PRESENT IN THE TEXT — do not assume external databases.

OUTPUT FORMAT per requirement:
─────────────────────────────────────────────────
[REQ-ID]: [first 60 chars...]
─────────────────────────────────────────────────
  REQ-T01 [SEVERITY] ✓/✗  — [upstream source found / missing]
  REQ-T02 [SEVERITY] ✓/✗  — [derivation justification]
  REQ-T03 [SEVERITY] ✓/✗  — [downward trace]
  REQ-T04 [SEVERITY] ✓/✗  — [HW/SW allocation]
  REQ-T05 [SEVERITY] ✓/✗  — [FHA failure condition reference]
  Score: XX/100
─────────────────────────────────────────────────

OVERALL TRACEABILITY SCORE: XX/100
SUMMARY: [2-3 sentence paragraph]"""

    req_text = json.dumps(state["requirements"], indent=2)
    response  = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Perform traceability analysis:\n\n{req_text}")
    ])
    return {"traceability_findings": response.content}


# ─────────────────────────────────────────────
# Agent 6: Correctness  (§5.2)
# ─────────────────────────────────────────────

def correctness_agent(state: ValidationState) -> ValidationState:
    """ARP4754A §5.2 — Correctness. Rules and checks loaded from rules.json."""
    print("  [Correctness]")
    llm        = get_llm()
    rules_text = format_rules_for_prompt("correctness")
    weak_verbs = ", ".join(WEAK_MODAL_VERBS)
    rag_ctx    = _rag_block(
        "system functions capabilities what the system does operational modes "
        "design decisions assumptions constraints", k=5
    )

    system_prompt = f"""You are an {STANDARD} §5.2 correctness auditor for aerospace systems.

{_severity_legend()}

CORRECTNESS RULES (from rules.json):
{rules_text}

WEAK MODAL VERBS (flag these — not mandatory): {weak_verbs}
REQUIRED MANDATORY VERB: "shall"
{rag_ctx}
{_sys_ctx(state)}

INSTRUCTIONS — for EACH requirement, execute every check step:
1. REQ-R01: compare requirement intent with the system definition; flag mismatches.
2. REQ-R02: detect HOW (implementation) language vs WHAT (behaviour) language.
3. REQ-R03: verify "shall" is used; flag "should", "may", "can", "must", etc.
4. REQ-R04: detect compound requirements — count distinct "shall" obligations.
5. REQ-R05: assess abstraction level relative to system hierarchy.
6. REQ-R06: detect implicit assumptions; verify they are explicitly stated.
7. State PASSES (✓) or FAILS (✗) with failure_severity per rule.
8. Give a per-requirement correctness score (0-100%).

OUTPUT FORMAT per requirement:
─────────────────────────────────────────────────
[REQ-ID]: [first 60 chars...]
─────────────────────────────────────────────────
  REQ-R01 [SEVERITY] ✓/✗  — [intent vs system definition]
  REQ-R02 [SEVERITY] ✓/✗  — [HOW vs WHAT analysis]
  REQ-R03 [SEVERITY] ✓/✗  — [modal verb: "X" used]
  REQ-R04 [SEVERITY] ✓/✗  — [N obligations found]
  REQ-R05 [SEVERITY] ✓/✗  — [abstraction assessment]
  REQ-R06 [SEVERITY] ✓/✗  — [implicit assumptions found or none]
  Score: XX/100
─────────────────────────────────────────────────

OVERALL CORRECTNESS SCORE: XX/100
SUMMARY: [2-3 sentence paragraph]"""

    req_text = json.dumps(state["requirements"], indent=2)
    response  = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Perform correctness analysis:\n\n{req_text}")
    ])
    return {"correctness_findings": response.content}


# ─────────────────────────────────────────────
# Agent 7: Recommender
# ─────────────────────────────────────────────

def recommender_agent(state: ValidationState) -> ValidationState:
    """
    Produces per-requirement corrected rewrites using:
      - Violations flagged by the five analysis agents
      - Checks and severities from rules.json (for precise rule references)
      - RAG context for project-specific vocabulary and values
    """
    print("  [Recommendation creation]")
    llm = get_llm(temperature=0.15)
    rag = get_rag()

    # Build compact rule reference from rules.json for the rewriter
    all_rules_summary = []
    for cat in ("correctness", "completeness", "consistency", "verifiability", "traceability"):
        for rule in get_rules(cat):
            rid      = rule["rule_id"]
            title    = rule["title"]
            severity = rule["failure_severity"]
            checks   = "; ".join(rule.get("checks", []))
            all_rules_summary.append(f"  {rid} [{severity}] {title}: {checks}")
    rules_ref = "\n".join(all_rules_summary)

    all_findings = (
        f"=== COMPLETENESS (§5.3) ===\n{state['completeness_findings']}\n\n"
        f"=== CONSISTENCY (§5.4) ===\n{state['consistency_findings']}\n\n"
        f"=== VERIFIABILITY (§5.5) ===\n{state['verifiability_findings']}\n\n"
        f"=== TRACEABILITY (§5.6) ===\n{state['traceability_findings']}\n\n"
        f"=== CORRECTNESS (§5.2) ===\n{state['correctness_findings']}\n\n"
        f"=== MULTI-REQUIREMENT ANALYSIS ===\n{state.get('multi_req_findings', '')}"
    )

    system_prompt = f"""You are a senior aerospace requirements engineer rewriting
non-compliant requirements to conform to {STANDARD} (v{VERSION}).

COMPLETE RULES REFERENCE (rule_id [severity] title: check steps):
{rules_ref}

REWRITING RULES:
- Use "shall" for every mandatory statement                          (fixes REQ-R03)
- One obligation per requirement; split compound ones               (fixes REQ-R04)
- Replace all ambiguous/subjective terms with measurable values     (fixes REQ-C02, REQ-V02, REQ-V03)
- State WHAT the system shall achieve, not HOW                      (fixes REQ-R02)
- Include units AND tolerances for all performance values           (fixes REQ-V02)
- Reference the DAL level for safety-critical requirements          (fixes REQ-C05)
- Add [Verification: T/A/I/D] tag to each rewritten statement      (fixes REQ-V04)
- Use exact terminology and numeric values from RAG project docs    (vocabulary consistency)
- Make implicit assumptions explicit                                (fixes REQ-R06)

OUTPUT FORMAT — one block per requirement:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIREMENT: [ID]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ORIGINAL:
  [verbatim original text]

VIOLATIONS:
  • [REQ-Xxx] [SEVERITY] — [one-line description of the violation and which check failed]

CORRECTED REWRITE:
  (split into -A, -B, ... if compound)
  [ID]: [rewritten statement] [Verification: T/A/I/D]

CHANGES EXPLAINED:
  • "[original wording]" → "[new wording]"  (fixes REQ-Xxx: [check step that was failing])

RAG VOCABULARY USED:
  • [term or value sourced from project documents, or "none"]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If a requirement is fully compliant across ALL rules, write exactly:
✓ [ID]: COMPLIANT — no rewrite needed."""

    parts = []
    for req in state["requirements"]:
        req_rag_ctx = ""
        if rag.is_ready():
            query = (
                f"{req.get('text', '')} "
                f"{req.get('type', '')} performance values thresholds terminology"
            )
            req_rag_ctx = _rag_block(query, k=4)

        user_message = (
            f"REQUIREMENT:\n{json.dumps(req, indent=2)}\n\n"
            f"{req_rag_ctx}"
            f"SYSTEM CONTEXT:\n{state.get('system_context', 'N/A')}\n\n"
            f"AUDIT FINDINGS:\n{all_findings}"
        )

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])
        parts.append(response.content)

    return {"recommendations": "\n\n".join(parts)}


# ─────────────────────────────────────────────
# Agent 8: Reporter
# ─────────────────────────────────────────────

def reporter_agent(state: ValidationState) -> ValidationState:
    """
    Synthesises all agent findings into the formal compliance report.
    The CORRECTED REWRITES section is appended by direct string concatenation —
    NOT passed through the LLM — to prevent placeholder substitution failure.
    """
    print("  [Report creation]")
    llm = get_llm(temperature=0.2)

    rag = get_rag()
    rag_status = (
        f"RAG Vector Store  : ACTIVE — {rag.chunk_count()} chunks indexed\n"
        f"Indexed sources   : {', '.join(rag.list_sources()) or 'none'}"
        if state.get("rag_available")
        else "RAG Vector Store  : NOT LOADED (generic analysis — no project documents)"
    )

    # List BLOCKING rule IDs from rules.json for the report header
    blocking_ids = []
    for cat in ("completeness", "consistency", "verifiability", "traceability", "correctness"):
        blocking_ids.extend(r["rule_id"] for r in blocking_rules(cat))
    blocking_note = f"BLOCKING rules (must all pass for approval): {', '.join(blocking_ids)}"

    system_prompt = f"""You are a senior aerospace systems engineer producing a formal
{STANDARD} Requirements Validation Report (rules v{VERSION}).

{rag_status}
{blocking_note}

Generate ONLY the sections below. Do NOT include a corrected rewrites section —
it will be appended separately in code.

══════════════════════════════════════════════════════════════════════
          {STANDARD} REQUIREMENTS VALIDATION REPORT
══════════════════════════════════════════════════════════════════════

EXECUTIVE SUMMARY
─────────────────
[3-4 sentences: overall compliance status, most critical issues, RAG impact]

ANALYSIS METADATA
─────────────────
  Standard              : {STANDARD} v{VERSION}
  Project               : {state.get("input_metadata", {}).get("project", "—")}
  Purpose               : {state.get("input_metadata", {}).get("purpose", "—")}
  Requirements analysed : [N]
  RAG knowledge base    : [ACTIVE — N chunks / NOT LOADED]
  Indexed source docs   : [list or "none"]

OVERALL COMPLIANCE SCORE: XX/100
  Weighted: Completeness 25% + Verifiability 25% + Consistency 20%
          + Correctness 15% + Traceability 15%

COMPLIANCE BREAKDOWN
────────────────────
  §5.2 Correctness     : XX/100  [●●●○○]
  §5.3 Completeness    : XX/100  [●●●○○]
  §5.4 Consistency     : XX/100  [●●●○○]
  §5.5 Verifiability   : XX/100  [●●●○○]
  §5.6 Traceability    : XX/100  [●●●○○]

BLOCKING FINDINGS  (must resolve before approval)
──────────────────────────────────────────────────
  [Req ID] | [Rule ID] | [One-line description]
  (or "None" if no BLOCKING violations found)

CRITICAL FINDINGS  (HIGH severity — must fix)
─────────────────────────────────────────────
  [Req ID] | [Rule ID] | [Description]

MAJOR FINDINGS  (MEDIUM severity — should fix)
───────────────────────────────────────────────
  [Req ID] | [Rule ID] | [Description]

MINOR FINDINGS  (LOW severity — recommended)
─────────────────────────────────────────────
  [Req ID] | [Rule ID] | [Description]

DETAILED FINDINGS BY SECTION
─────────────────────────────

§5.2 CORRECTNESS
[3-5 sentence prose summary referencing specific rule IDs and req IDs]

§5.3 COMPLETENESS
[3-5 sentence prose summary]

§5.4 CONSISTENCY
[3-5 sentence prose summary]

§5.5 VERIFIABILITY
[3-5 sentence prose summary]

§5.6 TRACEABILITY
[3-5 sentence prose summary]

MULTI-REQUIREMENT ANALYSIS
[Summarise contradiction / overlap / redundancy findings.
 State: N pairs analysed, N contradictions (list req IDs), N overlaps, N redundancies.
 Contradictions go directly into BLOCKING FINDINGS above.]

PRIORITISED ACTION PLAN
────────────────────────
[Numbered list ordered BLOCKING → HIGH → MEDIUM → LOW.
 Each item: N. [Priority] [Req ID] [Rule ID] — one-line action to take]

VALIDATION STATUS: [PASS / CONDITIONAL PASS / FAIL]
[Two-sentence justification. Rule: FAIL if any BLOCKING rule is violated;
 CONDITIONAL PASS if only HIGH/MEDIUM violations; PASS if only LOW or none.]"""

    user_message = f"""Synthesise these findings into the formal report:

REQUIREMENTS:
{json.dumps(state['requirements'], indent=2)}

=== COMPLETENESS (§5.3) ===
{state['completeness_findings']}

=== CONSISTENCY (§5.4) ===
{state['consistency_findings']}

=== VERIFIABILITY (§5.5) ===
{state['verifiability_findings']}

=== TRACEABILITY (§5.6) ===
{state['traceability_findings']}

=== CORRECTNESS (§5.2) ===
{state['correctness_findings']}

=== MULTI-REQUIREMENT ANALYSIS (contradictions / overlaps / redundancies) ===
{state.get('multi_req_stats', {})}
{state.get('multi_req_findings', 'Not run.')}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ])

    # ── Direct-concatenation sections (never via LLM to avoid placeholder failure) ──

    # Clusters summary
    clusters_block = (
        "\n\nREQUIREMENT CLUSTERS & COMPARISON PIPELINE\n"
        "───────────────────────────────────────────\n"
        + (state.get("clusters_summary", "") or "Not run.")
    )

    # Full multi-req findings (already formatted by comparator)
    multi_block = (
        "\n\nFULL MULTI-REQUIREMENT FINDINGS\n"
        "────────────────────────────────\n"
        + (state.get("multi_req_findings", "") or "Not run.")
    )

    # Corrected rewrites
    rewrites = state.get("recommendations", "").strip()
    rewrites_section = (
        "\n\nCORRECTED REQUIREMENT REWRITES\n"
        "───────────────────────────────\n"
        + (rewrites if rewrites else "No rewrites generated.")
    )

    final_report = (
        response.content.rstrip()
        + clusters_block
        + multi_block
        + rewrites_section
        + "\n\n══════════════════════════════════════════════════════════════════════"
    )

    return {"final_report": final_report}


# ─────────────────────────────────────────────
# Multi-Requirement Analysis Node
# ─────────────────────────────────────────────

def multi_req_agent(state: ValidationState) -> ValidationState:
    """
    LangGraph node that wraps the full 4-phase multi-requirement pipeline:
      Phase 1 — Normalizer  : canonical structured form per requirement
      Phase 2 — Clusterer   : group by subject / function / interface / mode
      Phase 3 — Filter      : prune pairs that cannot conflict
      Phase 4 — Comparator  : diagnose contradiction / overlap / redundancy

    Runs after all single-requirement agents so it can reference their
    findings if needed, and before the recommender so its results feed
    into corrective rewrites.
    """
    result = run_multi_req_pipeline(
        requirements=state["requirements"],
        system_context=state.get("system_context", ""),
    )
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
    """
    LangGraph workflow
    """
    workflow = StateGraph(ValidationState)

    workflow.add_node("orchestrator",  orchestrator_agent)
    workflow.add_node("completeness",  completeness_agent)
    workflow.add_node("consistency",   consistency_agent)
    workflow.add_node("verifiability", verifiability_agent)
    workflow.add_node("traceability",  traceability_agent)
    workflow.add_node("correctness",   correctness_agent)
    workflow.add_node("multi_req",     multi_req_agent)     # ← NEW 4-phase pipeline
    workflow.add_node("recommender",   recommender_agent)
    workflow.add_node("reporter",      reporter_agent)

    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator",  "completeness")
    workflow.add_edge("completeness",  "consistency")
    workflow.add_edge("consistency",   "verifiability")
    workflow.add_edge("verifiability", "traceability")
    workflow.add_edge("traceability",  "correctness")
    workflow.add_edge("correctness",   "multi_req")         # ← runs after single-req agents
    workflow.add_edge("multi_req",     "recommender")       # ← multi-req findings feed recommender
    workflow.add_edge("recommender",   "reporter")
    workflow.add_edge("reporter",      END)

    return workflow.compile()


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def validate_requirements(requirements_text: str) -> dict:
    """
    Main entry point. Runs the full multi-agent validation pipeline.

    Args:
        requirements_text: JSON string (or file path) in the canonical format:
                           { "metadata": {...}, "requirements": [{...}, ...] }

    Returns:
        Full state dict including all agent findings, per-requirement
        corrected rewrites, and the final compliance report.
    """
    graph = build_validation_graph()

    initial_state: ValidationState = {
        "raw_input":               requirements_text,
        "input_metadata":          {},
        "requirements":            [],
        "system_context":          "",
        "completeness_findings":   "",
        "consistency_findings":    "",
        "verifiability_findings":  "",
        "traceability_findings":   "",
        "correctness_findings":    "",
        "recommendations":         "",
        "final_report":            "",
        "rag_available":           get_rag().is_ready(),
        # Multi-requirement pipeline fields
        "normalized_requirements": [],
        "clusters_summary":        "",
        "multi_req_findings":      "",
        "multi_req_stats":         {},
        "messages":                [],
    }

    return graph.invoke(initial_state)
