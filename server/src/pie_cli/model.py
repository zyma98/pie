"""Model management commands for Pie CLI.

Implements: pie model list|download|remove
Uses HuggingFace Hub as the source for models.
"""

import typer
from huggingface_hub import scan_cache_dir, snapshot_download
from rich.console import Console
from rich.panel import Panel

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.text import Text

from pie.model import (
    get_hf_cache_dir,
    parse_repo_id_from_dirname,
    get_model_config,
    check_pie_compatibility,

)
import typing


class TqdmProgress:
    """Adapter to map tqdm calls to a global rich Progress instance."""

    _progress: typing.Optional[Progress] = None

    def __init__(
        self,
        iterable=None,
        desc: str | None = None,
        total: float | None = None,
        unit: str = "it",
        unit_scale: bool = False,
        miniters: int | None = None,
        ncols: int | None = None,
        **kwargs,
    ):
        self.iterable = iterable
        self.desc = desc
        self.total = total
        self.task_id = None

        # Filter out questionable HF bars that don't progress
        if desc and "incomplete total" in desc:
            return

        if self._progress:
            self.task_id = self._progress.add_task(
                description=desc or "",
                total=total,
                visible=True,
                unit=unit,
            )

    def update(self, n: float = 1) -> None:
        if self._progress and self.task_id is not None:
            self._progress.update(self.task_id, advance=n)

    def close(self) -> None:
        if self._progress and self.task_id is not None:
            self._progress.update(self.task_id, visible=False)

    def __iter__(self):
        if self.iterable:
            for item in self.iterable:
                yield item
                self.update(1)

    @classmethod
    def get_lock(cls):
        # Rich is thread-safe, but tqdm expects a lock.
        # We return a dummy context manager or a real lock.
        # huggingface_hub uses this to acquire lock before printing sometimes.
        import threading
        if not hasattr(cls, "_lock"):
            cls._lock = threading.Lock()
        return cls._lock

    @classmethod
    def set_lock(cls, lock):
        cls._lock = lock

    def refresh(self, nolock=False, lock_args=None):
        if self._progress:
            self._progress.refresh()

    def reset(self, total=None):
        if self._progress and self.task_id is not None:
            self._progress.update(self.task_id, total=total, completed=0)

    @classmethod
    def write(cls, s, file=None, end="\n", nolock=False):
        # rich Console print is thread safe
        # console is global here
        console.print(s, end=end)

    def set_description(self, desc=None, refresh=True):
        self.desc = desc
        if self._progress and self.task_id is not None:
            self._progress.update(self.task_id, description=desc or "")
            if refresh:
                self.refresh()

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        # rich handles speed and other metrics automatically.
        # If we really want postfix, we could append to description or use a custom column.
        # For now, we'll ignore it to keep the bar clean, as HF usually puts speed/size here 
        # which rich already shows.
        pass

    def disable(self):
        pass


class SmartDownloadColumn(ProgressColumn):
    """Renders file size if unit is 'B', otherwise 'completed/total unit'."""

    def render(self, task: "Task") -> Text:
        unit = task.fields.get("unit", "it")
        if unit == "B":
            return DownloadColumn().render(task)
        
        if task.total is None:
            return Text(f"{int(task.completed)} {unit}", style="progress.download")
        
        return Text(f"{int(task.completed)}/{int(task.total)} {unit}", style="progress.download")


class SmartTransferSpeedColumn(ProgressColumn):
    """Renders transfer speed if unit is 'B', otherwise 'speed unit/s'."""

    def render(self, task: "Task") -> Text:
        unit = task.fields.get("unit", "it")
        if unit == "B":
            return TransferSpeedColumn().render(task)
        
        if task.speed is None:
            return Text("?", style="progress.data.speed")
        
        return Text(f"{task.speed:.1f} {unit}/s", style="progress.data.speed")





console = Console()
app = typer.Typer(help="Manage models from HuggingFace")


@app.command("list")
def model_list() -> None:
    """List locally cached HuggingFace models."""
    cache_dir = get_hf_cache_dir()

    if not cache_dir.exists():
        console.print(
            Panel(
                "[dim]No HuggingFace cache found[/dim]",
                title="Models",
                title_align="left",
                border_style="dim",
            )
        )
        return

    # Collect models
    models: list[tuple[str, bool, str]] = []  # (repo_id, compatible, info)

    for entry in cache_dir.iterdir():
        if not entry.is_dir():
            continue
        repo_id = parse_repo_id_from_dirname(entry.name)
        if repo_id is None:
            continue

        config = get_model_config(cache_dir, repo_id)
        compatible, info = check_pie_compatibility(config)
        models.append((repo_id, compatible, info))

    if not models:
        console.print(
            Panel(
                "[dim]No models found[/dim]",
                title="Models",
                title_align="left",
                border_style="dim",
            )
        )
        return

    # Build display
    lines = Text()
    for i, (repo_id, compatible, info) in enumerate(sorted(models)):
        if i > 0:
            lines.append("\n")

        if compatible:
            lines.append("✓ ", style="green")
            lines.append(repo_id, style="white")
            lines.append(f" ({info})", style="dim")
        else:
            lines.append("○ ", style="dim")
            lines.append(repo_id, style="dim")
            lines.append(f" ({info})", style="dim")

    console.print(Panel(lines, title="Models", title_align="left", border_style="dim"))


@app.command("download")
def model_download(
    repo_id: str = typer.Argument(
        ..., help="HuggingFace repo ID (e.g., meta-llama/Llama-3.2-1B-Instruct)"
    )
) -> None:
    """Download a model from HuggingFace."""
    console.print()
    console.print(f"[bold]Downloading:[/bold] {repo_id}")

    try:
        # Configuration for the progress bar
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            SmartDownloadColumn(),
            SmartTransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        TqdmProgress._progress = progress

        with progress:
            local_path = snapshot_download(
                repo_id,
                local_files_only=False,
                tqdm_class=TqdmProgress,
            )
        
        # Cleanup global reference
        TqdmProgress._progress = None

        console.print(f"[green]✓[/green] Downloaded to {local_path}")

        # Check compatibility
        cache_dir = get_hf_cache_dir()
        config = get_model_config(cache_dir, repo_id)
        compatible, info = check_pie_compatibility(config)

        console.print()
        if compatible:
            console.print(f"[green]✓[/green] Pie compatible (arch: {info})")
            console.print("Add to config.toml:")
            console.print(f'  hf_repo = "{repo_id}"')
        else:
            console.print(f"[yellow]![/yellow] Not Pie compatible ({info})")

    except Exception as e:
        console.print(f"[red]✗[/red] Download failed: {e}")
        raise typer.Exit(1)


@app.command("remove")
def model_remove(
    repo_id: str = typer.Argument(..., help="HuggingFace repo ID to remove")
) -> None:
    """Remove a locally cached model."""
    try:
        cache_info = scan_cache_dir()
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to scan cache: {e}")
        raise typer.Exit(1)

    # Find the repo in cache
    target_repo = None
    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            target_repo = repo
            break

    if target_repo is None:
        console.print(f"[red]✗[/red] Model '{repo_id}' not found in cache")
        raise typer.Exit(1)

    # Confirm deletion
    size_mb = target_repo.size_on_disk / (1024 * 1024)
    if not typer.confirm(f"Remove {repo_id} ({size_mb:.1f} MB)?"):
        console.print("[dim]Aborted.[/dim]")
        return

    # Delete using huggingface_hub
    try:
        delete_strategy = cache_info.delete_revisions(
            *[rev.commit_hash for rev in target_repo.revisions]
        )
        delete_strategy.execute()
        console.print(f"[green]✓[/green] Removed {repo_id}")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to remove: {e}")
        raise typer.Exit(1)
