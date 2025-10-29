#!/usr/bin/env python3
"""GRPO-style trainer tuned for the booking environment."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from rich import box
from rich.console import Console
from rich.table import Table

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)


class GRPOAlgorithm:
    """Minimal policy-gradient trainer compatible with the new training loop."""

    def __init__(self, config: Dict[str, Any], console: Optional[Console] = None) -> None:
        self.config = config
        self.console = console or Console()

        self.model = None
        self.tokenizer = None
        self.optimizer: Optional[AdamW] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.grad_accum_steps = 1
        self.max_grad_norm = 1.0
        self.max_prompt_length = 512
        self.max_completion_length = 128
        self.reward_baseline: Optional[float] = None
        self.baseline_momentum = 0.9
        self.global_step = 0

    # ---------------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------------
    def setup_trainer(self, policy_agent) -> "GRPOAlgorithm":
        """Wire the policy agent into the lightweight optimiser."""

        grpo_cfg = self.config.get("grpo", {})

        self.grad_accum_steps = max(1, int(grpo_cfg.get("gradient_accumulation_steps", 1)))
        self.max_grad_norm = float(grpo_cfg.get("max_grad_norm", 1.0))
        self.max_prompt_length = int(grpo_cfg.get("max_prompt_length", 512))
        self.max_completion_length = int(grpo_cfg.get("max_completion_length", 256))
        self.baseline_momentum = min(max(float(grpo_cfg.get("baseline_momentum", 0.9)), 0.0), 0.999)

        learning_rate = float(grpo_cfg.get("learning_rate", 5e-5))
        weight_decay = float(grpo_cfg.get("weight_decay", 0.0))

        self.tokenizer = policy_agent.tokenizer

        if self.tokenizer.pad_token is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = policy_agent.model

        # Ensure LoRA adapters are present so we only update lightweight parameters.
        self.model = self._ensure_lora(policy_agent, grpo_cfg)
        self.device = getattr(policy_agent, "device", self.device)
        self.model.to(self.device)
        self.model.train()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

        self.console.print("[green]✓[/green] GRPO trainer initialised (lightweight policy gradient)")
        self.console.print(f"  - Learning rate: {learning_rate}")
        self.console.print(f"  - Grad accum steps: {self.grad_accum_steps}")
        self.console.print(f"  - Max prompt / completion: {self.max_prompt_length} / {self.max_completion_length}")

        return self

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def prepare_batch(self, trajectories: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten trajectories into state/action/reward triples."""

        samples: List[Dict[str, Any]] = []
        for traj in trajectories:
            states = traj.get("states") or []
            actions = traj.get("actions") or []
            rewards = traj.get("shaped_rewards") or traj.get("rewards") or []

            if not rewards and actions:
                rewards = [traj.get("terminal_reward", 0.0)] * len(actions)

            for state, action, reward in zip(states, actions, rewards):
                if not state or not action or not isinstance(reward, (int, float)):
                    continue
                if not action.strip():
                    continue
                samples.append({
                    "state": state,
                    "action": action,
                    "reward": float(reward),
                })

        return samples

    def _tokenize_pair(self, state: str, action: str) -> Optional[Dict[str, torch.Tensor]]:
        """Tokenise state/action so only action tokens contribute to gradients."""

        prompt_enc = self.tokenizer(
            state,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_length,
        )
        action_enc = self.tokenizer(
            action,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_completion_length,
            add_special_tokens=False,
        )

        action_len = action_enc["input_ids"].shape[1]
        if action_len == 0:
            return None

        input_ids = torch.cat([prompt_enc["input_ids"], action_enc["input_ids"]], dim=1).to(self.device)
        attention_mask = torch.cat([prompt_enc["attention_mask"], action_enc["attention_mask"]], dim=1).to(self.device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_len": prompt_enc["input_ids"].shape[1],
            "action_len": action_len,
        }

    def _policy_loss(self, state: str, action: str, advantage: float) -> Optional[Tuple[torch.Tensor, float, int]]:
        """Compute REINFORCE loss and log-probs for a single sample."""

        tokenised = self._tokenize_pair(state, action)
        if tokenised is None:
            return None

        input_ids = tokenised["input_ids"]
        attention_mask = tokenised["attention_mask"]
        prompt_len = tokenised["prompt_len"]
        action_len = tokenised["action_len"]

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]

        start = max(prompt_len - 1, 0)
        end = start + action_len

        action_logits = logits[:, start:end, :]
        action_labels = labels[:, start:end]

        log_probs = F.log_softmax(action_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, action_labels.unsqueeze(-1)).squeeze(-1)
        logprob_sum = token_log_probs.sum(dim=-1)

        advantage_tensor = torch.tensor(advantage, device=self.device, dtype=logprob_sum.dtype)
        loss = -(advantage_tensor * logprob_sum).mean() / max(action_len, 1)

        return loss, logprob_sum.detach().item(), action_len

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train_step(self, trajectories: List[Dict[str, Any]], epoch: int = 0) -> Dict[str, Any]:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Call setup_trainer before train_step.")

        batch = self.prepare_batch(trajectories)
        if not batch:
            logger.warning("No usable samples in trajectories; skipping GRPO step")
            return {
                "grpo_loss": 0.0,
                "mean_reward": 0.0,
                "kl_divergence": 0.0,
                "num_samples": 0,
            }

        rewards_tensor = torch.tensor([sample["reward"] for sample in batch], device=self.device)
        batch_mean = rewards_tensor.mean().item()
        if self.reward_baseline is None:
            self.reward_baseline = batch_mean
        else:
            self.reward_baseline = (
                self.baseline_momentum * self.reward_baseline + (1.0 - self.baseline_momentum) * batch_mean
            )
        baseline_value = self.reward_baseline
        advantages = rewards_tensor - baseline_value

        if self.console:
            self.console.print()
            self.console.print("[cyan]Running lightweight GRPO update...[/cyan]")

        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        total_logprob = 0.0
        total_tokens = 0
        processed = 0

        for idx, sample in enumerate(batch):
            advantage = advantages[idx].item()
            loss_tuple = self._policy_loss(sample["state"], sample["action"], advantage)
            if loss_tuple is None:
                continue

            loss, logprob_sum, token_count = loss_tuple
            scaled_loss = loss / self.grad_accum_steps
            scaled_loss.backward()

            total_loss += loss.detach().item()
            total_logprob += logprob_sum
            total_tokens += token_count
            processed += 1

            if (processed % self.grad_accum_steps == 0) or (idx == len(batch) - 1):
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

        mean_reward = rewards_tensor.mean().item()
        reward_std = rewards_tensor.std().item() if rewards_tensor.numel() > 1 else 0.0

        if self.console:
            self.console.print(
                f"[green]GRPO update[/green] - samples: {processed}, loss: {total_loss / max(processed,1):.4f}, "
                f"mean reward: {mean_reward:.3f}"
            )

        return {
            "grpo_loss": total_loss / max(processed, 1),
            "mean_reward": mean_reward,
            "reward_std": reward_std,
            "reward_baseline": baseline_value,
            "avg_logprob": (total_logprob / max(total_tokens, 1)) if total_tokens else 0.0,
            "kl_divergence": 0.0,
            "num_samples": processed,
        }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, output_dir: str) -> None:
        if self.model is None or self.tokenizer is None:
            return

        checkpoint_dir = Path(output_dir) / f"checkpoint-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "reward_baseline": self.reward_baseline,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
        }
        with open(checkpoint_dir / "training_state.json", "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)

        if self.console:
            self.console.print(f"[green]✓[/green] Checkpoint saved to {checkpoint_dir}")

    def save_final_model(self, output_dir: str) -> None:
        if self.model is None or self.tokenizer is None:
            return

        final_dir = Path(output_dir) / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        with open(final_dir / "config.json", "w", encoding="utf-8") as handle:
            json.dump(self.config, handle, indent=2)

        if self.console:
            self.console.print(f"[green]✓[/green] Final model saved to {final_dir}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def display_epoch_summary(self, epoch: int, metrics: Dict[str, Any]) -> None:
        if not self.console:
            return

        summary = Table(title=f"Epoch {epoch} Summary - GRPO", box=box.SIMPLE)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="green")

        summary.add_row("Avg Reward", f"{metrics.get('mean_reward', 0.0):.3f}")
        summary.add_row("Reward Baseline", f"{metrics.get('reward_baseline', 0.0):.3f}")
        summary.add_row("Loss", f"{metrics.get('grpo_loss', 0.0):.4f}")
        summary.add_row("Samples", str(metrics.get('num_samples', 0)))
        summary.add_row("Avg LogProb", f"{metrics.get('avg_logprob', 0.0):.4f}")

        self.console.print(summary)
        self.console.print()

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _ensure_lora(self, policy_agent, grpo_cfg: Dict[str, Any]):
        model = policy_agent.model
        if hasattr(model, "peft_config") and getattr(model, "peft_config", None):
            return model

        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=int(grpo_cfg.get("lora_rank", 16)),
            lora_alpha=int(grpo_cfg.get("lora_alpha", 32)),
            target_modules=grpo_cfg.get(
                "lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            lora_dropout=float(grpo_cfg.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        policy_agent.model = model
        return model
