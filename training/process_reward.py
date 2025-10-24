from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

try:
    import torch
except ImportError as exc:  # pragma: no cover - torch is a runtime dependency
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as exc:  # pragma: no cover - transformers is a runtime dependency
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


@dataclass
class ARMOScorerSettings:
    """Configuration bundle for ARMO process reward scoring."""

    model_path: str
    device: Optional[str] = None
    max_length: int = 1024
    max_history_turns: int = 8
    max_history_chars: int = 512
    score_scale: float = 1.0
    score_offset: float = 0.0
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None
    heuristics: Optional[Dict[str, Any]] = None


class ARMOProcessScorer:
    """
    Thin wrapper around an ARMO reward model checkpoint for per-turn process rewards.

    The scorer expects Hugging Face compatible checkpoints where the reward head is
    exposed via `AutoModelForSequenceClassification`. The model is evaluated in
    inference mode and returns a scalar reward for the current turn that can be
    combined with other shaping signals.
    """

    def __init__(self, **kwargs: Any) -> None:
        if _TRANSFORMERS_IMPORT_ERROR is not None:
            raise RuntimeError(
                "transformers is required for ARMOProcessScorer but is not installed."
            ) from _TRANSFORMERS_IMPORT_ERROR
        if _TORCH_IMPORT_ERROR is not None:
            raise RuntimeError(
                "torch is required for ARMOProcessScorer but is not installed."
            ) from _TORCH_IMPORT_ERROR

        settings = ARMOScorerSettings(**kwargs)
        if not settings.model_path:
            raise ValueError("model_path must be provided for ARMOProcessScorer.")

        self.device = self._resolve_device(settings.device)
        self.max_length = settings.max_length
        self.max_history_turns = settings.max_history_turns
        self.max_history_chars = settings.max_history_chars
        self.score_scale = settings.score_scale
        self.score_offset = settings.score_offset
        self.heuristics = settings.heuristics or {}
        self.clamp_range: Optional[Tuple[Optional[float], Optional[float]]] = (
            settings.clamp_min,
            settings.clamp_max,
        )
        self.role_prefix = {"user": "User", "assistant": "Assistant", "system": "System"}
        self.prompt_prefix = self.heuristics.get(
            "prompt_prefix", "Conversation transcript:"
        )

        logger.info(
            "Loading ARMO reward model from %s on device %s", settings.model_path, self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_path, use_fast=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            settings.model_path
        )
        self.model.to(self.device)
        self.model.eval()

    def score(
        self,
        history_before: Optional[Sequence[Dict[str, str]]],
        action: str,
    ) -> float:
        """
        Compute a shaped process reward for the given action using ARMO logits
        with optional heuristic adjustments.
        """
        if not action or not action.strip():
            empty_penalty = self.heuristics.get("empty_penalty")
            if empty_penalty is not None:
                return float(empty_penalty)
            return 0.0

        prompt = self._build_prompt(history_before or [])
        candidate = action.strip()

        encoded = self.tokenizer(
            prompt,
            candidate,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits

        reward = logits.squeeze().float().item()
        reward = reward * self.score_scale + self.score_offset
        reward += self._apply_heuristics(history_before or [], candidate)

        if self.clamp_range:
            clamp_min, clamp_max = self.clamp_range
            if clamp_min is not None:
                reward = max(clamp_min, reward)
            if clamp_max is not None:
                reward = min(clamp_max, reward)

        return float(reward)

    def _build_prompt(self, history: Sequence[Dict[str, str]]) -> str:
        """Format prior conversation history into a compact prompt."""
        if not history:
            return self.prompt_prefix

        relevant_history = (
            list(history)[-self.max_history_turns :]
            if self.max_history_turns > 0
            else list(history)
        )

        lines = [self.prompt_prefix]
        for message in relevant_history:
            role = message.get("role", "user").lower()
            prefix = self.role_prefix.get(role, role.title())
            content = (message.get("content") or "").strip()
            if self.max_history_chars and len(content) > self.max_history_chars:
                content = content[: self.max_history_chars - 3].rstrip() + "..."
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    def _apply_heuristics(
        self,
        history: Sequence[Dict[str, str]],
        action: str,
    ) -> float:
        """Apply lightweight heuristics to stabilize ARMO scores."""
        heuristics = self.heuristics
        if not heuristics:
            return 0.0

        adjustment = 0.0
        text = action.strip()

        min_chars = heuristics.get("min_char_length")
        if min_chars and len(text) < int(min_chars):
            adjustment += float(heuristics.get("short_penalty", -0.2))

        max_chars = heuristics.get("max_char_length")
        if max_chars and len(text) > int(max_chars):
            adjustment += float(heuristics.get("long_penalty", -0.1))

        question_bonus = heuristics.get("question_bonus")
        if question_bonus and "?" in text:
            adjustment += float(question_bonus)

        keywords = heuristics.get("keywords") or []
        keyword_bonus = float(heuristics.get("keyword_bonus", 0.0))
        if keywords and keyword_bonus:
            lowered = text.lower()
            if any(keyword in lowered for keyword in keywords):
                adjustment += keyword_bonus

        repetition_penalty = heuristics.get("repetition_penalty")
        if repetition_penalty:
            last_user = None
            for message in reversed(history):
                if message.get("role") == "user":
                    last_user = (message.get("content") or "").strip()
                    break
            if last_user and last_user.lower() == text.lower():
                adjustment += float(repetition_penalty)

        return adjustment

    @staticmethod
    def _resolve_device(preferred: Optional[str]) -> torch.device:
        """Resolve a usable torch device string."""
        if torch is None:  # pragma: no cover - guarded at init
            raise RuntimeError("torch must be available to resolve a device.")
        if preferred in (None, "", "auto"):
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():  # type: ignore[attr-defined]
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(preferred)
