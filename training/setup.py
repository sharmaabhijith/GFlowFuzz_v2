from __future__ import annotations
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field

from rich import box
from rich.table import Table
from rich.console import Console

from training.environment import BookingConversationEnvironment
from agents.auto_user.module import AutoUserAgent, AutoUserConfig
from training.trainer.ppo import PPOAlgorithm
from training.trainer.grpo import GRPOAlgorithm


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
        self._policy: Optional[AutoUserAgent] = None
        self._trainer: Optional[PPOAlgorithm | GRPOAlgorithm] = None
        self._output_dir: Optional[Path] = None

    def configure_hf_caches(self) -> None:
        """Ensure Hugging Face cache paths are set to consistent locations."""
        hf_cache = Path(os.environ.get("HF_HOME", Path.home() / "hf_cache"))
        os.environ.setdefault("HF_HOME", str(hf_cache))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "hub"))
        os.environ.setdefault("HF_HUB_CACHE", str(hf_cache / "hub"))
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
            env = BookingConversationEnvironment(config=env_cfg, auto_user_config=None)
            env.initialize()
            self.console.print("[green]✓[/green] Environment initialized")
            self._environment = env
        return self._environment

    def policy(self) -> AutoUserAgent:
        if self._policy is None:
            mcfg = self.config["model"]
            policy_cfg = AutoUserConfig(
                model_name=mcfg["base_model"],
                tokenizer_name=mcfg.get("tokenizer", mcfg["base_model"]),
                max_length=mcfg["max_response_length"],
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                device=mcfg.get("device", "auto"),
            )
            policy = AutoUserAgent(policy_cfg)
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


class ConsoleReporter:
    """Generalized console display helper for training.

    Provides consistent, reusable rendering for config, per-episode summaries,
    and final results.
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def show_config(self, algorithm_name: str, config: Dict[str, Any]) -> None:
        algo_config = config[algorithm_name]
        table = Table(title="Training Configuration", box=box.ROUNDED)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Algorithm", algorithm_name.upper())
        table.add_row("Model", config["model"]["base_model"])
        table.add_row("Learning Rate", str(algo_config.get("learning_rate")))
        table.add_row("Batch Size", str(algo_config.get("batch_size")))
        total_eps = algo_config.get(
            "total_episodes",
            algo_config.get("max_epochs", 1) * algo_config.get("max_conversations_per_epoch", 1),
        )
        table.add_row("Total Episodes", str(total_eps))
        self.console.print(table)
        self.console.print()

    def show_episode(self, episode_index: int, metrics: EpisodeMetrics, extra: Optional[Dict[str, Any]] = None) -> None:
        table = Table(title=f"Episode {episode_index} Summary", box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Terminal Reward", f"{metrics.terminal_reward:.3f}")
        table.add_row("Shaped Reward (mean)", f"{metrics.shaped_reward_mean:.3f}")
        table.add_row("Steps", str(metrics.total_steps))
        table.add_row("Loss", f"{metrics.algo_loss:.4f}")
        table.add_row("KL Divergence", f"{metrics.kl_divergence:.4f}")
        if metrics.policy_loss:
            table.add_row("Policy Loss", f"{metrics.policy_loss:.4f}")
        if metrics.value_loss:
            table.add_row("Value Loss", f"{metrics.value_loss:.4f}")
        if extra:
            for k, v in extra.items():
                table.add_row(k, f"{v}")
        self.console.print(table)
        self.console.print()

    def show_final(self, history: List[EpisodeMetrics], output_dir: Path) -> None:
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
