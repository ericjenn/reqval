# ARP4754A Requirements Validation System  v2

A multi-agent AI system that validates requirements against ARP4754A *(Guidelines for Development of Civil Aircraft
and Systems)*.

---

## What's new in v1
---

## Architecture


---

## Installation

```bash

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your OpenAI API key
Customize file `.env`

---

## Usage

### Ingest system documents + validate

```bash
python main.py --docs ./system_docs/ --file requirements.txt
```

The `system_docs/` folder can contain any mix of:
- System Description / Concept of Operations (TXT, DOCX, PDF)
- Functional Hazard Assessment (FHA) — failure conditions, DAL levels
- Interface Control Document (ICD) — data formats, protocols, timing
- System Safety Assessment (PSSA/SSA)* — probability targets
- Glossary / Acronym List
- Any other project reference document

### Validate with existing vector store (no re-ingestion)

```bash
python main.py --file requirements.txt
```

### Add new documents to the store

```bash
python main.py --docs ./new_docs/ --file reqs.json
```

### Reset the store and rebuild from scratch

```bash
python main.py --docs ./system_docs/ --clear-store --file reqs.json
```

### Check what documents are currently indexed

```bash
python main.py --list-docs
```


### Save report + verbose intermediate output

```bash
python main.py --docs ./docs/ --file reqs.json --output report.txt --verbose
```

---


---

## ARP4754A Rules Covered

| Section | Area | Key checks |
|---------|------|-----------|
| §5.2 | Correctness | "shall" usage, compound reqs, HOW vs WHAT, abstraction level |
| §5.3 | Completeness | Unique IDs, acceptance criteria, DAL refs, source/rationale, interface definitions |
| §5.4 | Consistency | Contradictions, terminology, unit conflicts, DAL alignment |
| §5.5 | Verifiability | Quantifiable thresholds, T/A/I/D method mapping, measurability |
| §5.6 | Traceability | Upstream source, FHA linkage, derived req. justification, HW/SW allocation |

---

## File Structure


---

## Optional: LangSmith Tracing

```bash
# Add to .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=arp4754-validator
```

This gives you a visual trace of every agent call and RAG retrieval in the LangSmith UI.
