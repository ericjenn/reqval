import warnings
warnings.filterwarnings("ignore", message=".*pydantic.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*FieldInfo.*",  category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")

"""
ARP4754A Requirements Validator — CLI Entry Point (v2 — with RAG)

Usage examples:
    python main.py --docs ./system_docs/ --file reqs.json
    python main.py --file reqs.json
    python main.py --docs ./system_docs/ --clear-store --file reqs.json
    python main.py --list-docs
    python main.py --docs ./docs/ --file reqs.json --output report.txt --verbose
"""

import json
import math
import time
import typer
from pathlib import Path
from rich.console import Console
from rich.panel   import Panel
from rich.progress import (
    Progress, SpinnerColumn, TextColumn,
    BarColumn, MofNCompleteColumn, TaskProgressColumn, TimeElapsedColumn,
)
from rich.table import Table
from rich.text   import Text
from rich.rule   import Rule

from agents import validate_requirements, set_progress_callback
from rag    import get_rag, reset_rag

app     = typer.Typer(help="ARP4754A Requirements Validation System — Multi-Agent + RAG")
console = Console()


# ─────────────────────────────────────────────
# Display Helpers
# ─────────────────────────────────────────────

def print_banner():
    from llm_provider import provider_info  # noqa: local import to avoid hard dep at module level
    info   = provider_info()
    banner = Text()
    banner.append("  ARP4754A Requirements Validation System  v2\n", style="bold cyan")
    banner.append(
        f"  Multi-Agent LangGraph  |  {info['provider'].upper()}: {info['llm_model']}"
        f"  |  Embeddings: {info['embed_model']}\n",
        style="dim",
    )
    console.print(Panel(banner, border_style="cyan", padding=(1, 4)))


def print_rag_status(rag):
    if rag.is_ready():
        sources = rag.list_sources()
        table   = Table(show_header=False, box=None, padding=(0, 2))
        table.add_row("[green]●[/green] RAG Status",  "[green]ACTIVE[/green]")
        table.add_row("  Chunks indexed",  str(rag.chunk_count()))
        table.add_row("  Source documents", ", ".join(sources) if sources else "none")
        console.print(Panel(table, title="[bold]Knowledge Base[/bold]", border_style="green"))
    else:
        console.print(Panel(
            "[yellow]● RAG NOT LOADED[/yellow]\n"
            "  Running in generic mode. Use [bold]--docs <folder>[/bold] to ingest system documents\n"
            "  for vocabulary-grounded analysis and context-aware recommendations.",
            title="[bold]Knowledge Base[/bold]", border_style="yellow",
        ))


# ─────────────────────────────────────────────
# Main CLI Command
# ─────────────────────────────────────────────

@app.command()
def main(
    file: Path = typer.Option(
        None, "--file", "-f", help="Path to requirements JSON file"
    ),
    docs: Path = typer.Option(
        None, "--docs", "-d", help="Folder of system documents to ingest into RAG"
    ),
    clear_store: bool = typer.Option(
        False, "--clear-store", help="Delete the existing vector store before ingesting"
    ),
    list_docs: bool = typer.Option(
        False, "--list-docs", help="List documents currently indexed, then exit"
    ),
    output: Path = typer.Option(
        None, "--output", "-o", help="Save the final report to a text file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show intermediate agent findings"
    ),
    store_dir: str = typer.Option(
        "./faiss_store", "--store-dir", help="Directory for the FAISS vector store"
    ),
):
    print_banner()

    # ── RAG setup ──────────────────────────────────────────────────────────
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

    # ── Load requirements ──────────────────────────────────────────────────
    if not file:
        console.print("[red]No requirements provided. Use --file.[/red]")
        raise typer.Exit(1)
    if not file.exists():
        console.print(f"[red]Error: File '{file}' not found.[/red]")
        raise typer.Exit(1)

    requirements_text = file.read_text(encoding="utf-8")
    try:
        from req_parser import load_internal, summary as _req_summary
        _meta, _reqs = load_internal(requirements_text)
        console.print(f"\n[bold yellow]► {_req_summary(_meta, _reqs)}[/bold yellow]")
        if _meta.get("intentionally_non_compliant"):
            console.print("[dim yellow]  ⚠  Dataset is marked intentionally non-compliant[/dim yellow]")
    except Exception as _e:
        console.print(f"[red]Error reading requirements JSON: {_e}[/red]")
        raise typer.Exit(1)

    n_reqs = len(_reqs)

    # ── Validate with live progress ────────────────────────────────────────
    #
    # Two progress bars:
    #
    #   [1] Per-requirement analysis
    #       Total = 6 agents × n_reqs  (completeness, consistency, verifiability,
    #       traceability, correctness each do 1 bulk LLM call = 1 agent-pass;
    #       recommender loops per requirement = n_reqs calls)
    #       We advance by n_reqs each time an agent finishes.
    #
    #   [2] Multi-req pair comparison
    #       Total is unknown until clustering runs; we initialise to n_reqs*(n_reqs-1)//2
    #       (worst-case) and correct it when the first pair_progress event arrives.
    #       Advances by batch of 8 pairs after each comparator LLM call.

    N_SINGLE_REQ_AGENTS = 7            # completeness, verif, trace, correctness, wording, recommender + consistency(bulk)
    req_total  = N_SINGLE_REQ_AGENTS * n_reqs
    pair_total = max(n_reqs * (n_reqs - 1) // 2, 1)

    # Track per-agent progress so each agent advances its own slice cleanly.
    # _agent_req_done[label] = number of req-units emitted so far for that agent.
    _agent_req_done: dict[str, int] = {}

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=32),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=10,
            transient=False,
        ) as progress:

            task_reqs  = progress.add_task(
                "[bold cyan]Per-requirement analysis  [/bold cyan]",
                total=req_total,
            )
            task_pairs = progress.add_task(
                "[bold cyan]Multi-req pair comparison [/bold cyan]",
                total=pair_total,
            )

            # ── Progress callback (called from inside graph.invoke) ──────
            def _on_progress(event: str, current: int, total: int, label: str) -> None:

                if event == "agent_start":
                    # Update the description to show the active agent name
                    progress.update(
                        task_reqs,
                        description=f"[bold cyan]{label:<28}[/bold cyan]",
                    )

                elif event == "req_progress":
                    # current = reqs done so far in THIS agent's pass (0..n_reqs)
                    # We track how much we have already advanced for this agent
                    # so we only advance by the new delta.
                    done_before = _agent_req_done.get(label, 0)
                    delta       = current - done_before
                    if delta > 0:
                        progress.advance(task_reqs, delta)
                        _agent_req_done[label] = current
                    # Live label update
                    progress.update(
                        task_reqs,
                        description=(
                            f"[bold cyan]{label:<20}[/bold cyan]"
                            f"[dim] {current}/{total} reqs[/dim]"
                        ),
                    )

                elif event == "agent_done":
                    # Snap this agent's slice to n_reqs (handles bulk single-call agents)
                    done_before = _agent_req_done.get(label, 0)
                    gap = n_reqs - done_before
                    if gap > 0:
                        progress.advance(task_reqs, gap)
                        _agent_req_done[label] = n_reqs
                    progress.update(
                        task_reqs,
                        description=f"[bold cyan]{label:<28}[/bold cyan][dim] ✓[/dim]",
                    )

                elif event == "pair_progress":
                    # Correct the total if we now know the real pair count
                    if progress.tasks[task_pairs].total != total:
                        progress.update(task_pairs, total=total)
                    progress.update(
                        task_pairs,
                        completed=current,
                        description=(
                            "[bold cyan]Multi-req pair comparison [/bold cyan]"
                            f"[dim]{current}/{total} pairs[/dim]"
                        ),
                    )

            # ── Run the pipeline ─────────────────────────────────────────
            set_progress_callback(_on_progress)
            result = validate_requirements(requirements_text)
            set_progress_callback(None)

            # Snap both bars to 100%
            progress.update(task_reqs,  completed=req_total)
            progress.update(task_pairs, completed=progress.tasks[task_pairs].total)

    except ValueError as e:
        console.print(f"\n[red bold]Configuration Error:[/red bold] {e}")
        console.print("[dim]Check your .env — set OPENAI_API_KEY or LLM_PROVIDER=ollama[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red bold]Error during validation:[/red bold] {e}")
        raise typer.Exit(1)

    # ── Verbose intermediate output ────────────────────────────────────────
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
            ("Multi-Req Findings",                    result.get("multi_req_findings", "")),
        ]
        for title, content in sections:
            console.print(Panel(
                str(content)[:4000],
                title=f"[bold cyan]{title}[/bold cyan]",
                border_style="dim", padding=(0, 2),
            ))

    # ── Final report ───────────────────────────────────────────────────────
    console.print("\n")
    console.rule("[cyan bold]FINAL VALIDATION REPORT[/cyan bold]")
    console.print()
    report = result.get("final_report", "No report generated.")
    console.print(report)

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
