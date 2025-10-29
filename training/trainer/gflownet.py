#!/usr/bin/env python3
"""GFlowNet-style trainer for trajectory balance fine-tuning."""

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


class GFlowNetAlgorithm:
    """Trajectory-balance trainer compatible with the booking environment loop."""

    def __init__(self, config: Dict[str, Any], console: Optional[Console] = None) -> None:
        self.config = config
        self.gfn_cfg = self.config.get("gflownet", {})
        self.console = console or Console()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.grad_accum_steps = max(1, int(self.gfn_cfg.get("gradient_accumulation_steps", 1)))
        self.max_grad_norm = float(self.gfn_cfg.get("max_grad_norm", 1.0))
        self.max_prompt_length = int(self.gfn_cfg.get("max_prompt_length", 512))
        self.max_completion_length = int(self.gfn_cfg.get("max_completion_length", 256))
        self.min_reward = float(self.gfn_cfg.get("min_reward", 1e-4))
        self.reward_exponent = float(self.gfn_cfg.get("reward_exponent", 1.0))
        self.reward_scale = float(self.gfn_cfg.get("reward_scale", 1.0))
        self.use_terminal_reward = bool(self.gfn_cfg.get("use_terminal_reward", True))
        self.use_shaped_rewards = bool(self.gfn_cfg.get("use_shaped_rewards", False))
        self.shaped_reward_coef = float(self.gfn_cfg.get("shaped_reward_coef", 0.0))
        self.entropy_coef = float(self.gfn_cfg.get("entropy_coef", 0.0))

        self.global_step = 0
        self.epsilon = 1e-6
        self.log_z: Optional[torch.nn.Parameter] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup_trainer(self, policy_agent) -> "GFlowNetAlgorithm":

        learning_rate = float(self.gfn_cfg.get("learning_rate", 5e-5))
        log_z_lr = float(self.gfn_cfg.get("log_z_learning_rate", learning_rate))
        weight_decay = float(self.gfn_cfg.get("weight_decay", 0.0))

        self.tokenizer = policy_agent.tokenizer
        if self.tokenizer.pad_token is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = policy_agent.model
        self.model = self._ensure_lora(policy_agent, self.gfn_cfg)
        self.device = getattr(policy_agent, "device", self.device)
        self.model.to(self.device)
        self.model.train()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        init_log_z = float(self.gfn_cfg.get("init_log_z", 0.0))
        self.log_z = torch.nn.Parameter(torch.tensor(init_log_z, dtype=torch.float32, device=self.device))
        opt_groups = [
            {"params": trainable_params, "lr": learning_rate, "weight_decay": weight_decay},
            {"params": [self.log_z], "lr": log_z_lr, "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(opt_groups)

        self.console.print("[green]✓[/green] GFlowNet trainer initialised")
        self.console.print(f"  - Learning rate: {learning_rate}")
        self.console.print(f"  - Log Z lr: {log_z_lr}")
        self.console.print(f"  - Grad accum steps: {self.grad_accum_steps}")
        self.console.print(f"  - Reward floor: {self.min_reward}")
        return self

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def prepare_batch(self, trajectories: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for traj in trajectories:
            states = traj.get("states") or []
            actions = traj.get("actions") or []
            reward = self._compute_reward(
                float(traj.get("terminal_reward", 0.0)),
                traj.get("shaped_rewards") or [],
            )

            pairs: List[Tuple[str, str]] = []
            for state, action in zip(states, actions):
                pairs.append((state, action))
            samples.append({"pairs": pairs, "reward": reward})

        return samples

    def _compute_reward(self, terminal_reward: float, shaped_rewards: Sequence[float]) -> float:
        value = 0.0
        if self.use_terminal_reward:
            value += max(terminal_reward, 0.0)
        if self.use_shaped_rewards and shaped_rewards:
            shaped_sum = sum(max(float(r), 0.0) for r in shaped_rewards)
            value += self.shaped_reward_coef * shaped_sum

        value = self.reward_scale * max(value, self.min_reward)
        if self.reward_exponent != 1.0:
            value = max(self.min_reward, value ** self.reward_exponent)
        return value

    def _tokenize_pair(self, state: str, action: str) -> Optional[Dict[str, torch.Tensor]]:
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

    def _trajectory_logprob(self, pairs: Sequence[Tuple[str, str]]) -> Optional[Tuple[torch.Tensor, int, torch.Tensor]]:
        logprob_terms: List[torch.Tensor] = []
        total_tokens = 0
        entropy_terms: List[torch.Tensor] = []

        for state, action in pairs:
            tokenised = self._tokenize_pair(state, action)
            if tokenised is None:
                continue

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
            logprob_terms.append(token_log_probs.sum(dim=-1))
            total_tokens += action_len

            if self.entropy_coef > 0.0:
                probs = log_probs.exp()
                token_entropy = -(probs * log_probs).sum(dim=-1)
                entropy_terms.append(token_entropy.sum(dim=-1))

        if not logprob_terms:
            return None

        traj_logprob = torch.stack(logprob_terms).sum()
        entropy = torch.stack(entropy_terms).sum() if entropy_terms else torch.tensor(0.0, device=self.device)
        return traj_logprob, total_tokens, entropy

    def _trajectory_loss(self, pairs: Sequence[Tuple[str, str]], reward: float) -> Optional[Tuple[torch.Tensor, float, int, float]]:
        
        logprob_tuple = self._trajectory_logprob(pairs)
        traj_logprob, token_count, entropy = logprob_tuple
        reward_tensor = torch.tensor(reward, device=self.device, dtype=traj_logprob.dtype)
        log_reward = torch.log(reward_tensor + self.epsilon)
        balance = traj_logprob + self.log_z - log_reward
        loss = 0.5 * balance.pow(2)
        if self.entropy_coef > 0.0:
            loss = loss - self.entropy_coef * entropy

        return loss, traj_logprob.detach().item(), token_count, entropy.detach().item()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train_step(self, trajectories: List[Dict[str, Any]], epoch: int = 0) -> Dict[str, Any]:
        if self.model is None or self.optimizer is None or self.log_z is None:
            raise RuntimeError("Call setup_trainer before train_step.")

        batch = self.prepare_batch(trajectories)
        if not batch:
            logger.warning("No usable trajectories; skipping GFlowNet step")
            return {
                "loss": 0.0,
                "mean_reward": 0.0,
                "num_samples": 0,
                "avg_logprob": 0.0,
                "log_z": self.log_z.detach().item(),
            }

        if self.console:
            self.console.print()
            self.console.print("[cyan]Running GFlowNet trajectory-balance update...[/cyan]")

        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        total_logprob = 0.0
        total_tokens = 0
        total_entropy = 0.0
        rewards: List[float] = []
        processed = 0

        for idx, sample in enumerate(batch):
            loss_tuple = self._trajectory_loss(sample["pairs"], sample["reward"])
            if loss_tuple is None:
                continue

            loss, logprob_value, token_count, entropy_value = loss_tuple
            scaled_loss = loss / self.grad_accum_steps
            scaled_loss.backward()

            total_loss += loss.detach().item()
            total_logprob += logprob_value
            total_tokens += token_count
            total_entropy += entropy_value
            rewards.append(sample["reward"])
            processed += 1

            if (processed % self.grad_accum_steps == 0) or (idx == len(batch) - 1):
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

        mean_reward = float(sum(rewards) / len(rewards)) if rewards else 0.0
        reward_std = torch.tensor(rewards).std().item() if len(rewards) > 1 else 0.0

        if self.console:
            self.console.print(
                f"[green]GFlowNet update[/green] - trajectories: {processed}, "
                f"loss: {total_loss / max(processed,1):.4f}, mean reward: {mean_reward:.3f}, "
                f"log Z: {self.log_z.detach().item():.3f}"
            )

        return {
            "loss": total_loss / max(processed, 1),
            "mean_reward": mean_reward,
            "reward_std": reward_std,
            "avg_logprob": (total_logprob / max(total_tokens, 1)) if total_tokens else 0.0,
            "avg_entropy": total_entropy / max(processed, 1) if processed else 0.0,
            "log_z": self.log_z.detach().item(),
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
            "log_z": float(self.log_z.detach().cpu()) if self.log_z is not None else 0.0,
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
    # Internal utilities
    # ------------------------------------------------------------------
    def _ensure_lora(self, policy_agent, gfn_cfg: Dict[str, Any]):
        model = policy_agent.model
        if hasattr(model, "peft_config") and getattr(model, "peft_config", None):
            return model

        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=int(gfn_cfg.get("lora_rank", 16)),
            lora_alpha=int(gfn_cfg.get("lora_alpha", 32)),
            target_modules=gfn_cfg.get(
                "lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            lora_dropout=float(gfn_cfg.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        policy_agent.model = model
        return model

