"""Command Line Interface for the Self-Evaluating Lesson Content Generator."""

import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from src.config import Config
from src.pipeline import run_pipeline
from src.models import CheckpointID, PipelineOutput

console = Console()


def display_rubric_table(evaluation, attempt_num: int):
    """Renders a formatted table of the 6 checkpoints."""
    table = Table(
        title=f"📋 Rubric Evaluation (Attempt #{attempt_num})",
        header_style="bold cyan",
        show_lines=True
    )
    table.add_column("Checkpoint", style="bold", width=24)
    table.add_column("Decision", justify="center", width=10)
    table.add_column("Reason / Evidence", width=42)
    table.add_column("Remediation", width=36)

    for chk in evaluation.checkpoints:
        status_str = "[bold green]PASS[/bold green]" if chk.passed else "[bold red]FAIL[/bold red]"
        table.add_row(
            chk.name,
            status_str,
            f"{chk.reason}\n[dim]{chk.evidence}[/dim]",
            chk.remediation if not chk.passed else "[green]Criteria satisfied[/green]"
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Self-Evaluating Lesson Content Generator — Agentic Loop with Hard Checkpoint Rubrics"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=Config.DEFAULT_TOPIC,
        help=f"The lesson topic to generate (default: '{Config.DEFAULT_TOPIC}')"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["mock", "gemini", "openai"],
        default=None,
        help="LLM Provider to use (default: auto-detected from environment or 'mock')"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=Config.MAX_RETRIES,
        help=f"Maximum regeneration retries (default: {Config.MAX_RETRIES})"
    )

    args = parser.parse_args()

    console.print(Panel.fit(
        f"[bold cyan]🏗️ Self-Evaluating Lesson Content Generator[/bold cyan]\n"
        f"[yellow]Topic:[/yellow] {args.topic}\n"
        f"[yellow]Loop Strategy:[/yellow] Generate ➔ Evaluate (6 Hard Binary Checkpoints) ➔ Regenerate (Max {args.max_retries} retries) ➔ Self-Evolve\n"
        f"[yellow]Persistence:[/yellow] JSON Memory & Evolved Guidelines",
        border_style="cyan"
    ))

    console.print("\n[bold blue]🚀 Starting Agentic Pipeline...[/bold blue]")
    
    output: PipelineOutput = run_pipeline(
        topic=args.topic,
        max_retries=args.max_retries,
        provider_type=args.provider
    )

    console.print("\n" + "=" * 80)
    console.print(f"[bold]🏁 Pipeline Execution Complete | Final Status: [/bold] [bold green]{output.status}[/bold green]")
    console.print(f"[bold]Total Attempts Executed:[/bold] {output.total_attempts}")
    console.print(f"[bold]Total Rejections Encountered:[/bold] {len(output.rejection_log)}")
    console.print("=" * 80 + "\n")

    # Display Rejection Log if rejections occurred
    if output.rejection_log:
        console.print(Panel.fit("[bold red]🛑 REJECTION AUDIT LOG[/bold red]", border_style="red"))
        for entry in output.rejection_log:
            rejection_text = (
                f"[bold]Attempt #{entry.attempt} Failed at Checkpoints:[/bold] {', '.join(entry.failed_checkpoints)}\n\n"
                f"[bold]Evaluator Critique:[/bold] {entry.critique_summary}\n\n"
                f"[bold]Remediation Fed Into Next Prompt:[/bold]\n" +
                "\n".join(f"  • {item}" for item in entry.remediation_plan)
            )
            console.print(Panel(rejection_text, title=f"Rejection Record #{entry.attempt}", border_style="yellow"))

    # Display Final Rubric Matrix
    display_rubric_table(output.final_evaluation, output.total_attempts)

    # Display Evolved Guidelines
    if output.evolved_guidelines_used:
        console.print(Panel.fit("[bold magenta]🧠 PERSISTENT EVOLVED GUIDELINES (Memory)[/bold magenta]", border_style="magenta"))
        for g in output.evolved_guidelines_used:
            console.print(f"  ✨ {g}")

    # Output Files
    console.print("\n[bold green]📦 OUTPUT ARTIFACTS SAVED:[/bold green]")
    console.print(f"  📄 [bold]Final Lesson:[/bold] {Config.FINAL_LESSON_OUTPUT_PATH.resolve()}")
    console.print(f"  📋 [bold]Rejection Log:[/bold] {Config.REJECTION_LOG_OUTPUT_PATH.resolve()}")
    console.print(f"  💾 [bold]Run History Memory:[/bold] {Config.RUN_HISTORY_PATH.resolve()}")
    console.print(f"  🧠 [bold]Evolved Guidelines:[/bold] {Config.EVOLVED_GUIDELINES_PATH.resolve()}")

    console.print("\n[bold cyan]📖 Final Passing Lesson Preview:[/bold cyan]\n")
    console.print(Markdown(output.final_lesson))


if __name__ == "__main__":
    main()

