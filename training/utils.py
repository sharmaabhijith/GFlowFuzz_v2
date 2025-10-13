from __future__ import annotations

import logging
import random
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table


class BookingObjectiveGenerator:
    """Generate high-level booking objectives with database-backed locations."""

    _DEFAULT_ORIGINS = [
        "New York",
        "Los Angeles",
        "Chicago",
        "Miami",
        "Boston",
        "Seattle",
        "San Francisco",
        "Denver",
        "Dallas",
        "Atlanta",
    ]

    _DEFAULT_DESTINATIONS = [
        "London",
        "Paris",
        "Tokyo",
        "Sydney",
        "Dubai",
        "Singapore",
        "Hong Kong",
        "Rome",
        "Barcelona",
        "Amsterdam",
    ]

    _DEFAULT_AIRLINES = [
        "Delta",
        "United",
        "American Airlines",
        "British Airways",
        "Lufthansa",
        "Air France",
        "Emirates",
        "Qantas",
        "Singapore Airlines",
        "Air Canada",
    ]

    _DEFAULT_CABINS = ["economy", "business", "first"]
    _DEFAULT_TIME_OF_DAY = ["morning", "afternoon", "evening", "red-eye"]
    _DEFAULT_SEAT_TYPES = ["window seat", "aisle seat"]
    _DEFAULT_MONTHS = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

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
        self._airlines = self._load_airlines()
        self._cabin_classes = self._load_cabin_classes()

    def _resolve_db_path(self, db_path: Optional[Path | str]) -> Optional[Path]:
        if not db_path:
            return None

        path = Path(db_path)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            path = project_root / path

        if not path.exists():
            self.logger.warning(f"Objective generator could not find database at {path}")
            return None
        return path

    def _connect(self) -> Optional[sqlite3.Connection]:
        if not self._db_path:
            return None
        try:
            return sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            self.logger.warning(f"Failed to open flights database: {exc}")
            return None

    def _load_locations(self) -> tuple[List[str], List[str]]:
        origins: List[str] = []
        destinations: List[str] = []

        conn = self._connect()
        if conn:
            try:
                origins = self._fetch_city_names(conn, "departure_airport")
                destinations = self._fetch_city_names(conn, "arrival_airport")
            finally:
                conn.close()

        if not origins:
            origins = self._DEFAULT_ORIGINS.copy()
        if not destinations:
            destinations = self._DEFAULT_DESTINATIONS.copy()
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
            names = [row[0] for row in cursor.fetchall() if row[0]]
            return names
        except sqlite3.Error as exc:
            self.logger.warning(f"Failed to load city names from database: {exc}")
            return []

    def _load_airlines(self) -> List[str]:
        conn = self._connect()
        airlines: List[str] = []
        if conn:
            try:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT airline_name
                    FROM flights
                    WHERE airline_name IS NOT NULL AND airline_name != ''
                    ORDER BY airline_name COLLATE NOCASE
                    """
                )
                airlines = [row[0] for row in cursor.fetchall() if row[0]]
            except sqlite3.Error as exc:
                self.logger.warning(f"Failed to load airlines from database: {exc}")
            finally:
                conn.close()

        if not airlines:
            airlines = self._DEFAULT_AIRLINES.copy()
        return airlines

    def _load_cabin_classes(self) -> List[str]:
        conn = self._connect()
        cabins: List[str] = []

        if conn:
            try:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT cabin_class
                    FROM flights
                    WHERE cabin_class IS NOT NULL AND cabin_class != ''
                    ORDER BY cabin_class COLLATE NOCASE
                    """
                )
                cabins = [row[0].lower() for row in cursor.fetchall() if row[0]]
            except sqlite3.Error as exc:
                self.logger.warning(f"Failed to load cabin classes: {exc}")
            finally:
                conn.close()

        if not cabins:
            cabins = self._DEFAULT_CABINS.copy()
        return cabins

    def reset_seed(self, seed: Optional[int]) -> None:
        """Reset the RNG seed (useful for reproducible experiments)."""
        if seed is None:
            self._rng = random.Random()
            return
        self._rng.seed(seed)

    def _weighted_choice(
        self,
        weight_map: Optional[Dict[str, Any]],
        default_value: Optional[str],
        allowed: Optional[Sequence[str]] = None,
    ) -> Optional[str]:
        if not weight_map:
            return default_value

        candidates: List[tuple[str, float]] = []
        allowed_set = {item.lower(): item for item in allowed} if allowed else None
        for key, value in weight_map.items():
            try:
                weight = float(value)
            except (TypeError, ValueError):
                continue
            if weight <= 0:
                continue
            key_str = str(key)
            if allowed_set is not None:
                lookup = key_str.lower()
                if lookup not in allowed_set:
                    continue
                key_str = allowed_set[lookup]
            candidates.append((key_str, weight))

        if not candidates:
            return default_value

        total = sum(weight for _, weight in candidates)
        pick = self._rng.uniform(0, total)
        cumulative = 0.0
        for key, weight in candidates:
            cumulative += weight
            if pick <= cumulative:
                return key
        return candidates[-1][0]

    def _weighted_sample_without_replacement(
        self,
        weight_map: Optional[Dict[str, Any]],
        count: int,
        allowed: Optional[Sequence[str]] = None,
    ) -> List[str]:
        if not weight_map or count <= 0:
            return []

        selections: List[str] = []
        remaining: Dict[str, Any] = dict(weight_map)

        for _ in range(count):
            choice = self._weighted_choice(remaining, None, allowed=allowed)
            if not choice:
                break
            selections.append(choice)
            remaining.pop(choice, None)
        return selections

    def generate(self) -> str:
        """Generate a booking objective string."""
        weights_cfg = self.config.get("weights") or {}
        complexity_cfg = self.config.get("complexity") or {}

        cabin_choice = self._weighted_choice(
            weights_cfg.get("cabin"),
            default_value=self._DEFAULT_CABINS[0],
            allowed=self._cabin_classes,
        )
        cabin = (cabin_choice or self._DEFAULT_CABINS[0]).lower()

        trip_type = (
            self._weighted_choice(weights_cfg.get("trip_type"), "one_way")
            or "one_way"
        ).lower()
        direct_pref = (
            self._weighted_choice(weights_cfg.get("direct"), "prefer")
            or "prefer"
        ).lower()
        passengers_raw = self._weighted_choice(weights_cfg.get("passengers"), "1") or "1"
        try:
            passengers = max(1, int(float(passengers_raw)))
        except (TypeError, ValueError):
            passengers = 1

        origin = self._rng.choice(self._origins)
        destination = self._choose_destination(origin)

        month = self._rng.choice(self._DEFAULT_MONTHS)
        travel_window = self._rng.choice(
            [
                f"{month} 2026",
                f"early {month} 2026",
                f"mid {month} 2026",
                f"late {month} 2026",
            ]
        )

        complexity_min = max(0, int(complexity_cfg.get("min", 0)))
        complexity_max = max(complexity_min, int(complexity_cfg.get("max", 2)))
        if complexity_max <= 0:
            num_extras = 0
        else:
            sampled = self._rng.randint(complexity_min, complexity_max)
            num_extras = min(1, sampled)

        extra_tags = self._weighted_sample_without_replacement(
            weights_cfg.get("extras"), num_extras
        )
        extra_clauses: List[str] = []

        for tag in extra_tags:
            tag_lower = tag.lower()
            if tag_lower == "budget":
                budget = self._rng.choice(list(range(400, 1801, 50)))
                extra_clauses.append(f"keep the total under ${budget}")
            elif tag_lower == "airline":
                airline = self._rng.choice(self._airlines)
                extra_clauses.append(f"prefer to fly with {airline}")
            elif tag_lower == "time_of_day":
                time_pref = self._rng.choice(self._DEFAULT_TIME_OF_DAY)
                if time_pref == "red-eye":
                    extra_clauses.append("open to a red-eye departure")
                else:
                    extra_clauses.append(f"prefer a {time_pref} departure")
            elif tag_lower == "seat":
                seat_pref = self._rng.choice(self._DEFAULT_SEAT_TYPES)
                extra_clauses.append(f"would like a {seat_pref}")
            elif tag_lower == "bags":
                bag_count = self._rng.choice([1, 2, 3])
                label = "bag" if bag_count == 1 else "bags"
                extra_clauses.append(f"traveling with {bag_count} checked {label}")
            elif tag_lower == "flexibility":
                extra_clauses.append("dates are flexible by a few days")

        detail_clauses: List[str] = []
        if direct_pref == "prefer":
            detail_clauses.append("prefer direct flights")
        elif direct_pref in {"allow", "layover_ok", "layover"}:
            detail_clauses.append("okay with a layover if needed")

        passenger_clause = f"for {passengers} passengers" if passengers > 1 else ""
        trip_descriptor = "round-trip" if trip_type == "round_trip" else "one-way"

        base_parts = [
            f"I need a {trip_descriptor} {cabin} flight",
            f"from {origin} to {destination}",
            f"in {travel_window}",
            passenger_clause,
        ]
        base_sentence = " ".join(part for part in base_parts if part).strip()

        extras_sentence = ""
        extra_sentence_parts = detail_clauses + extra_clauses
        if extra_sentence_parts:
            extras_sentence = " Also, " + "; ".join(extra_sentence_parts) + "."

        if not base_sentence.endswith("."):
            base_sentence += "."
        return base_sentence + extras_sentence

    def _choose_destination(self, origin: str) -> str:
        candidates = [city for city in self._destinations if city != origin]
        if not candidates:
            fallback = [city for city in self._DEFAULT_DESTINATIONS if city != origin]
            if fallback:
                candidates = fallback
            else:
                return origin
        return self._rng.choice(candidates)



class ConsoleReporter:
    """Generalized console display helper for training outputs."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()

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
