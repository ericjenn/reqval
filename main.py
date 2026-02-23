import warnings
# Suppress pydantic v1 deprecation warnings emitted by langchain/langgraph
# on Python 3.12+ — these are noise and do not affect functionality.
warnings.filterwarnings("ignore", message=".*pydantic.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*FieldInfo.*", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")

"""
ARP4754A Requirements Validator — CLI Entry Point (v2 — with RAG)

Usage examples:
    # With RAG: ingest system docs, then validate
    python main.py --docs ./system_docs/ --file reqs.json

    # Add docs to an existing store, then validate
    python main.py --docs ./new_docs/ --file reqs.json

    # Validate only (reuse previously built store)
    python main.py --file reqs.json

    # Reset the vector store, re-ingest, then validate
    python main.py --docs ./system_docs/ --clear-store --file reqs.json

    # Show what documents are currently indexed
    python main.py --list-docs

    # Save report to file + verbose intermediate agent output
    python main.py --docs ./docs/ --file reqs.json --output report.txt --verbose
"""

import json
import time
import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.text import Text
from rich.rule import Rule

from agents import validate_requirements
from rag import get_rag, reset_rag
from llm_provider import validate_provider, provider_info

app     = typer.Typer(help="ARP4754A Requirements Validation System — Multi-Agent + RAG")
console = Console()



# ─────────────────────────────────────────────
# Display Helpers
# ─────────────────────────────────────────────

def print_banner():
    info = provider_info()
    provider_str = info["provider"]
    model_str    = info["llm_model"]
    embed_str    = info["embed_model"]
    banner = Text()
    banner.append("  ARP4754A Requirements Validation System  v2\n", style="bold cyan")
    banner.append(f"  Multi-Agent LangGraph  |  {provider_str}: {model_str}  |  Embeddings: {embed_str}\n", style="dim")
    console.print(Panel(banner, border_style="cyan", padding=(1, 4)))


def print_rag_status(rag):
    if rag.is_ready():
        sources = rag.list_sources()
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_row("[green]●[/green] RAG Status", "[green]ACTIVE[/green]")
        table.add_row("  Chunks indexed", str(rag.chunk_count()))
        table.add_row("  Source documents", ", ".join(sources) if sources else "none")
        console.print(Panel(table, title="[bold]Knowledge Base[/bold]", border_style="green"))
    else:
        console.print(Panel(
            "[yellow]● RAG NOT LOADED[/yellow]\n"
            "  Running in generic mode. Use [bold]--docs <folder>[/bold] to ingest system documents\n"
            "  for vocabulary-grounded analysis and context-aware recommendations.",
            title="[bold]Knowledge Base[/bold]", border_style="yellow"
        ))


# ─────────────────────────────────────────────
# Main CLI Command
# ─────────────────────────────────────────────

@app.command()
def main(
    file: Path = typer.Option(
        None, "--file", "-f",
        help="Path to requirements text file"
    ),
    docs: Path = typer.Option(
        None, "--docs", "-d",
        help="Folder of system documents to ingest into the RAG vector store"
    ),
    clear_store: bool = typer.Option(
        False, "--clear-store",
        help="Delete the existing vector store before ingesting new documents"
    ),
    list_docs: bool = typer.Option(
        False, "--list-docs",
        help="List documents currently indexed in the vector store, then exit"
    ),
    output: Path = typer.Option(
        None, "--output", "-o",
        help="Save the final report to a text file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show intermediate agent findings in addition to the final report"
    ),
    store_dir: str = typer.Option(
        "./faiss_store", "--store-dir",
        help="Directory for the FAISS persistent vector store"
    ),
):
    print_banner()

    # ── Validate LLM provider config before doing anything else ──
    try:
        validate_provider()
    except ValueError as exc:
        console.print(f"\n[red bold]Provider Error:[/red bold] {exc}")
        raise typer.Exit(1)

    # ── Load / prepare RAG ──
    rag = get_rag(store_dir=store_dir)

    if clear_store:
        console.print("\n[yellow]► Clearing existing vector store...[/yellow]")
        rag.clear()
        reset_rag()
        rag = get_rag(store_dir=store_dir)

    if docs:
        if not docs.is_dir():
            console.print(f"[red]Error: --docs path '{docs}' is not a directory.[/red]")
            raise typer.Exit(1)
        console.print(f"\n[bold cyan]► Ingesting system documents from:[/bold cyan] {docs}")
        n = rag.ingest_folder(str(docs))
        console.print(f"  [green]✓ {n} new chunks added to vector store[/green]")

    if list_docs:
        print_rag_status(rag)
        raise typer.Exit(0)

    print_rag_status(rag)

    # ── Get requirements text ──
    if file:
        if not file.exists():
            console.print(f"[red]Error: File '{file}' not found.[/red]")
            raise typer.Exit(1)
        requirements_text = file.read_text(encoding="utf-8")
        # Validate JSON and show a summary before starting the pipeline
        try:
            import json as _json
            from req_parser import load_internal, summary as _req_summary
            _meta, _reqs = load_internal(requirements_text)
            console.print(f"\n[bold yellow]► {_req_summary(_meta, _reqs)}[/bold yellow]")
            if _meta.get("intentionally_non_compliant"):
                console.print("[dim yellow]  ⚠  Dataset is marked intentionally non-compliant[/dim yellow]")
        except Exception as _e:
            console.print(f"[red]Error reading requirements JSON: {_e}[/red]")
            raise typer.Exit(1)
    else:
        console.print("[red]No requirements provided. Use --file.[/red]")
        raise typer.Exit(1)

    # ── Run validation ──
    try:
        result = validate_requirements(requirements_text)
    except ValueError as e:
        console.print(f"\n[red bold]Configuration Error:[/red bold] {e}")
        console.print("[dim]Check your .env file — set OPENAI_API_KEY or configure Ollama (LLM_PROVIDER=ollama)[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red bold]Error during validation:[/red bold] {e}")
        raise typer.Exit(1)

    # ── Verbose intermediate output ──
    if verbose:
        console.print("\n")
        console.rule("[dim]Intermediate Agent Findings[/dim]")
        sections = [
            ("Parsed Requirements (Orchestrator)",    _fmt(result.get("requirements", []))),
            ("System Context (from RAG)",             result.get("system_context", "") or "[No RAG context]"),
            ("§5.3 Completeness Findings",            result.get("completeness_findings", "")),
            ("§5.4 Consistency Findings",             result.get("consistency_findings", "")),
            ("§5.5 Verifiability Findings",           result.get("verifiability_findings", "")),
            ("§5.6 Traceability Findings",            result.get("traceability_findings", "")),
            ("§5.2 Correctness Findings",             result.get("correctness_findings", "")),
            ("Corrected Rewrites (Recommender)",      result.get("recommendations", "")),
            ("Multi-Req Clusters",                    result.get("clusters_summary", "")),
            ("Multi-Req Findings (Contradictions/Overlaps/Redundancies)", result.get("multi_req_findings", "")),
        ]
        for title, content in sections:
            console.print(Panel(
                str(content)[:4000],
                title=f"[bold cyan]{title}[/bold cyan]",
                border_style="dim", padding=(0, 2)
            ))

    # ── Final report ──
    console.print("\n")
    console.rule("[cyan bold]FINAL VALIDATION REPORT[/cyan bold]")
    console.print()
    report = result.get("final_report", "No report generated.")
    console.print(report)

    # ── Save to file ──
    if output:
        output.write_text(report, encoding="utf-8")
        console.print(f"\n[green]✓ Report saved to:[/green] {output}")

    console.print()
    console.rule("[dim]Validation Complete[/dim]")


def _fmt(data) -> str:
    try:
        return json.dumps(data, indent=2)
    except Exception:
        return str(data)


if __name__ == "__main__":
    app()
