from __future__ import annotations

import logging
import random
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler

class BookingObjectiveGenerator:
    """Generate booking objectives using cities drawn from the flights database."""

    def __init__(
        self,
        db_path: Optional[Path | str] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self._rng = random.Random()
        seed = self.config.get("seed")
        if seed is not None:
            self._rng.seed(seed)

        self._db_path = self._resolve_db_path(db_path)
        self._origins, self._destinations = self._load_locations()

    def _resolve_db_path(self, db_path: Optional[Path | str]) -> Path:
        if not db_path:
            raise ValueError("BookingObjectiveGenerator requires a database path.")

        path = Path(db_path)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            path = project_root / path

        if not path.exists():
            raise FileNotFoundError(f"Flights database not found at {path}")
        return path

    def _connect(self) -> sqlite3.Connection:
        try:
            return sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise RuntimeError(f"Failed to open flights database: {exc}") from exc

    def _load_locations(self) -> tuple[List[str], List[str]]:
        with self._connect() as conn:
            origins = self._fetch_city_names(conn, "departure_airport")
            destinations = self._fetch_city_names(conn, "arrival_airport")

        if not origins:
            raise RuntimeError(f"No departure cities available in database: {self._db_path}")
        if not destinations:
            raise RuntimeError(f"No arrival cities available in database: {self._db_path}")

        return origins, destinations

    def _fetch_city_names(self, conn: sqlite3.Connection, airport_column: str) -> List[str]:
        query = f"""
            SELECT DISTINCT c.name
            FROM flights f
            JOIN cities c ON c.airport_code = f.{airport_column}
            WHERE c.name IS NOT NULL AND c.name != ''
            ORDER BY c.name COLLATE NOCASE
        """
        try:
            cursor = conn.execute(query)
            return [row[0] for row in cursor.fetchall() if row[0]]
        except sqlite3.Error as exc:
            raise RuntimeError(f"Failed to load city names from database: {exc}") from exc

    def reset_seed(self, seed: Optional[int]) -> None:
        """Reset the RNG seed (useful for reproducible experiments)."""
        if seed is None:
            self._rng = random.Random()
            return
        self._rng.seed(seed)

    def generate(self) -> str:
        """Generate a booking objective containing only origin and destination."""
        origin = self._rng.choice(self._origins)
        destination = self._choose_destination(origin)
        return f"I want to book a flight from {origin} to {destination}."

    def _choose_destination(self, origin: str) -> str:
        candidates = [city for city in self._destinations if city != origin]
        if not candidates:
            raise RuntimeError("No suitable destination city found different from the origin.")
        return self._rng.choice(candidates)



class ConsoleReporter:
    """Generalized console display helper for training outputs."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()

    def configure_logging(self, output_dir: Path) -> None:
        """Route logs to both the Rich console and a persistent file."""
        log_path = Path(output_dir) / "training.log"
        root_logger = logging.getLogger()

        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

        rich_handler = RichHandler(
            console=self.console,
            rich_tracebacks=True,
            show_time=False,
            show_path=False,
            markup=True,
        )
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        rich_handler.setFormatter(logging.Formatter("%(message)s"))

        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(rich_handler)
        root_logger.addHandler(file_handler)

        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("datasets").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    def show_config(self, algorithm_name: str, config: Dict[str, Any]) -> None:
        algo_config = config[algorithm_name]
        table = Table(title="Training Configuration", box=box.ROUNDED)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Algorithm", algorithm_name.upper())
        cfg_path = config["environment"]["auditor"]["config_path"]
        from training.setup import load_config  # Lazy import to avoid cycles

        auditor_cfg = load_config(str(cfg_path))
        table.add_row("Auditor Model", auditor_cfg.get("model_name", "<unknown>"))
        table.add_row("Learning Rate", str(algo_config.get("learning_rate")))
        table.add_row("Batch Size", str(algo_config.get("batch_size")))
        total_eps = algo_config.get(
            "total_episodes",
            algo_config.get("max_epochs", 1)
            * algo_config.get("max_conversations_per_epoch", 1),
        )
        table.add_row("Total Episodes", str(total_eps))
        self.console.print(table)
        self.console.print()

    def show_turn(
        self,
        episode_index: int,
        turn_index: int,
        user_message: str,
        assistant_message: Optional[str],
    ) -> None:
        max_len = 280

        def _format_text(text: Optional[str]) -> str:
            if not text:
                return "[dim]<no response>[/dim]"
            stripped = text.strip()
            truncated = stripped if len(stripped) <= max_len else stripped[: max_len - 3] + "..."
            return escape(truncated)

        user_display = _format_text(user_message)
        assistant_display = _format_text(assistant_message)
        panel_content = (
            f"[bold cyan]User:[/bold cyan] {user_display}\n\n"
            f"[bold magenta]Agent:[/bold magenta] {assistant_display}"
        )
        self.console.print(
            Panel(
                panel_content,
                title=f"Episode {episode_index} · Turn {turn_index}",
                expand=False,
            )
        )

    def show_episode(
        self,
        episode_index: int,
        metrics: Any,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        table = Table(title=f"Episode {episode_index} Summary", box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Terminal Reward", f"{metrics.terminal_reward:.3f}")
        table.add_row("Shaped Reward (mean)", f"{metrics.shaped_reward_mean:.3f}")
        table.add_row("Steps", str(metrics.total_steps))
        table.add_row("Loss", f"{metrics.algo_loss:.4f}")
        table.add_row("KL Divergence", f"{metrics.kl_divergence:.4f}")
        if getattr(metrics, "policy_loss", None):
            table.add_row("Policy Loss", f"{metrics.policy_loss:.4f}")
        if getattr(metrics, "value_loss", None):
            table.add_row("Value Loss", f"{metrics.value_loss:.4f}")
        if extra:
            for k, v in extra.items():
                table.add_row(k, f"{v}")
        self.console.print(table)
        self.console.print()

    def show_final(self, history: List[Any], output_dir: Path) -> None:
        table = Table(title="Training Results", box=box.DOUBLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        if history:
            final = history[-1]
            avg_terminal = sum(rec.terminal_reward for rec in history) / len(history)
            avg_shaped = sum(rec.shaped_reward_mean for rec in history) / len(history)
            success_rate = sum(1 for rec in history if rec.success) / len(history)
            total_steps = sum(rec.total_steps for rec in history)
            table.add_row("Final Episode", str(final.episode_index))
            table.add_row("Final Terminal Reward", f"{final.terminal_reward:.3f}")
            table.add_row("Avg Terminal Reward", f"{avg_terminal:.3f}")
            table.add_row("Avg Shaped Reward", f"{avg_shaped:.3f}")
            table.add_row("Success Rate", f"{success_rate:.1%}")
            table.add_row("Total Steps", str(total_steps))
        table.add_row("Model Saved To", str(output_dir))
        self.console.print(table)
