from __future__ import annotations
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field

from rich.console import Console

from .environment import BookingConversationEnvironment
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from agents.auditor.module import AuditorAgent, AuditorConfig
from .trainer.ppo import PPOAlgorithm
from .trainer.grpo import GRPOAlgorithm


def load_config(config_path: str) -> Dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_metrics(metrics: List[Dict], output_dir: str) -> Path:
    """Save training metrics to the output directory and return the file path."""
    metrics_path = Path(output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics_path

@dataclass
class Trajectory:
    """Structured representation of a single conversation episode."""

    episode_index: int
    states: List[str]
    actions: List[str]
    rewards: List[float]  # Shaped rewards per step
    terminal_reward: float
    episode_reward: float
    num_turns: int
    conversation_history: List[Dict[str, Any]]
    booking_summary: Optional[Any] = None
    objective: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for trainer interfaces and serialization."""
        return {
            "episode_index": self.episode_index,
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "episode_reward": self.episode_reward,
            "terminal_reward": self.terminal_reward,
            "num_turns": self.num_turns,
            "conversation_history": self.conversation_history,
            "booking_summary": self.booking_summary,
            "objective": self.objective,
        }


@dataclass
class EpisodeMetrics:
    """Stores metrics for a single training episode."""

    episode_index: int
    total_steps: int
    terminal_reward: float
    shaped_reward_mean: float
    shaped_reward_sum: float
    algo_loss: float
    mean_reward: float
    kl_divergence: float
    policy_loss: float = 0.0
    value_loss: float = 0.0
    success: bool = False
    raw_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "total_steps": self.total_steps,
            "terminal_reward": self.terminal_reward,
            "shaped_reward_mean": self.shaped_reward_mean,
            "shaped_reward_sum": self.shaped_reward_sum,
            "algo_loss": self.algo_loss,
            "mean_reward": self.mean_reward,
            "kl_divergence": self.kl_divergence,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "success": self.success,
            "raw_metrics": self.raw_metrics,
        }

    def for_display(self) -> Dict[str, Any]:
        return {
            "avg_reward": self.shaped_reward_mean,
            "avg_terminal_reward": self.terminal_reward,
            "avg_process_reward": 0.0,
            "success_rate": 1.0 if self.success else 0.0,
            "loss": self.algo_loss,
            "kl_divergence": self.kl_divergence,
            "value_loss": self.value_loss,
            "policy_loss": self.policy_loss,
            "num_trajectories": 1,
        }

    @classmethod
    def from_trajectory(
        cls,
        episode_index: int,
        trajectory: "Trajectory",
        train_metrics: Dict[str, Any],
    ) -> "EpisodeMetrics":
        rewards = trajectory.rewards or [trajectory.terminal_reward]
        shaped_sum = float(sum(rewards))
        shaped_mean = shaped_sum / len(rewards) if rewards else 0.0
        algo_loss = train_metrics.get("grpo_loss", train_metrics.get("loss", 0.0))
        return cls(
            episode_index=episode_index,
            total_steps=trajectory.num_turns,
            terminal_reward=trajectory.terminal_reward,
            shaped_reward_mean=shaped_mean,
            shaped_reward_sum=shaped_sum,
            algo_loss=algo_loss,
            mean_reward=train_metrics.get("mean_reward", shaped_mean),
            kl_divergence=train_metrics.get("kl_divergence", 0.0),
            policy_loss=train_metrics.get("policy_loss", 0.0),
            value_loss=train_metrics.get("value_loss", 0.0),
            success=trajectory.terminal_reward > 0.5,
            raw_metrics=train_metrics,
        )

class TrainingSetup:
    """Encapsulates environment/model/trainer setup and output configuration."""

    def __init__(self, config: Dict, console: Optional[Console] = None):
        self.config = config
        self.console = console or Console()
        self._environment: Optional[BookingConversationEnvironment] = None
        self._policy: Optional[AuditorAgent] = None
        self._trainer: Optional[PPOAlgorithm | GRPOAlgorithm] = None
        self._output_dir: Optional[Path] = None
        self._auditor_config: Optional[Dict[str, Any]] = None

    def resolve_path(self, path: Path | str | None, *, default: Path | str | None = None) -> Path:
        """Resolve a config path relative to the project root."""
        if path is None:
            if default is None:
                raise ValueError("Cannot resolve an empty path without a default.")
            return self.resolve_path(default)

        candidate = Path(path)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

    def get_booking_agent_paths(self) -> Dict[str, Path]:
        """Return resolved paths required by the booking chat agent."""
        env_cfg = self.config.get("environment", {})
        booking_cfg = env_cfg.get("booking_agent_config", {})
        return {
            "config_path": self.resolve_path(booking_cfg.get("config_path"), default="agents/chat/config.yaml"),
            "db_path": self.resolve_path(booking_cfg.get("db_path"), default="database/flights.db"),
            "server_path": self.resolve_path(booking_cfg.get("server_path"), default="mcp-server/database_server.py"),
        }

    def get_verifier_config_path(self) -> Path:
        """Return the resolved verifier configuration path."""
        env_cfg = self.config.get("environment", {})
        verifier_cfg = env_cfg.get("verifier_config", {})
        return self.resolve_path(verifier_cfg.get("config_path"), default="agents/verifier/config.yaml")

    def get_user_agent_config_path(self) -> Path:
        """Return the resolved user agent configuration path."""
        env_cfg = self.config.get("environment", {})
        user_cfg_path = env_cfg.get("user_agent_config_path", "agents/user/config.yaml")
        return self.resolve_path(user_cfg_path)

    def ensure_database(self) -> Path:
        """Ensure the flights database directory exists and return its path."""
        db_path = self.get_booking_agent_paths()["db_path"]
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    def configure_hf_caches(self) -> None:
        """Ensure Hugging Face cache paths are set to consistent locations."""
        hf_cache = Path(os.environ.get("HF_HOME", Path.home() / "hf_cache"))
        os.environ.setdefault("HF_HOME", str(hf_cache))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "hub"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache / "hub"))
        os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache / "datasets"))

    def prepare_output(self) -> Path:
        """Create output directories and persist configuration."""
        algo = self.config[self.config["algorithm"].lower()]
        output_dir = Path(algo["output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "trajectory_logs").mkdir(parents=True, exist_ok=True)

        cfg_path = output_dir / "training_config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(self.config, f)

        self.console.print(f"[green]Output directory:[/green] {output_dir}")
        self.console.print(f"[green]Trajectory logs:[/green] {output_dir / 'trajectory_logs'}")
        self.console.print()
        self._output_dir = output_dir
        return output_dir

    def environment(self) -> BookingConversationEnvironment:
        if self._environment is None:
            env_cfg = self.config["environment"]
            env = BookingConversationEnvironment(config=env_cfg, auditor_config=None)
            env.initialize()
            self.console.print("[green]✓[/green] Environment initialized")
            self._environment = env
        return self._environment

    def policy(self) -> AuditorAgent:
        if self._policy is None:
            cfg_path = self.config["environment"]["auditor"]["config_path"]
            auditor_cfg = load_config(str(cfg_path))
            policy_cfg = AuditorConfig(
                model_name=auditor_cfg["model_name"],
                max_length=auditor_cfg["max_response_length"],
                temperature=auditor_cfg.get("temperature", 0.7),
                top_p=auditor_cfg.get("top_p", 0.9),
                do_sample=auditor_cfg.get("do_sample", True),
                device=auditor_cfg.get("device", "auto"),
                system_prompt=auditor_cfg.get("system_prompt", ""),
            )
            policy = AuditorAgent(policy_cfg)
            policy.initialize_model()
            self.console.print("[green]✓[/green] Policy model loaded")
            self._policy = policy
        return self._policy

    def trainer(self) -> PPOAlgorithm | GRPOAlgorithm:
        if self._trainer is None:
            algo_name = self.config.get("algorithm", "ppo").lower()
            if algo_name == "grpo":
                trainer = GRPOAlgorithm(self.config, console=self.console)
            else:
                trainer = PPOAlgorithm(self.config, console=self.console)
            trainer.setup_trainer(self.policy())
            self.console.print("[green]✓[/green] Trainer initialized")
            self._trainer = trainer
        return self._trainer

    @property
    def output_dir(self) -> Optional[Path]:
        return self._output_dir
