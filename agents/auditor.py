#!/usr/bin/env python3

"""
Stateless Auditor Agent - Pure Policy Implementation
Acts as a pure function mapping states to actions without maintaining any internal state.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from dataclasses import dataclass
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class AuditorConfig:
    """Configuration for the Auditor agent"""
    model_name: str
    max_length: int
    temperature: float
    top_p: float
    do_sample: bool
    device: str
    system_prompt: str = ""


class AuditorAgent:
    """
    Stateless Auditor Agent - Pure Policy Function

    This agent acts as a pure function π(s) → a, mapping states to actions
    without maintaining any internal conversation state.

    All conversation context must be provided in the state string from the environment.
    """

    def __init__(self, config: AuditorConfig):
        """
        Initialize the stateless agent with model and tokenizer only.
        No conversation state is maintained.

        Args:
            config: Configuration for the agent
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None

    def _resolve_device(self, device_preference: Optional[str]) -> torch.device:
        """Pick the best available torch device based on preference."""
        preference = (device_preference or "auto").lower()
        if preference == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        try:
            return torch.device(preference)
        except (RuntimeError, ValueError) as exc:
            logger.warning(
                "Requested device '%s' is unavailable (%s); falling back to CPU",
                device_preference,
                exc,
            )
            return torch.device("cpu")

    def initialize_model(self):
        """Load the model and tokenizer, applying PEFT adapter if available"""
        # Determine device
        self.device = self._resolve_device(self.config.device)
        supports_half_precision = self.device.type in {"cuda", "mps"}
        model_dtype = torch.float16 if supports_half_precision else torch.float32

        model_path = Path(self.config.model_name)
        # Load the adapter config to get the base model
        peft_config = PeftConfig.from_pretrained(str(model_path))
        base_model_name = peft_config.base_model_name_or_path

        # Load tokenizer from adapter path (it should have the tokenizer files)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )

        # Load base model
        logger.info(f"Loading base model on {self.device} with dtype {model_dtype}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=model_dtype,
            device_map="auto",
        )

        # Load and merge PEFT adapter
        logger.info(f"Applying PEFT adapter...")
        self.model = PeftModel.from_pretrained(base_model, str(model_path))
        # Merge adapter weights into base model for faster inference
        logger.info(f"Merging adapter weights...")
        self.model = self.model.merge_and_unload()

        # Configure tokenizer
        if self.tokenizer.pad_token is None and hasattr(self.tokenizer, "eos_token"):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.tokenizer, "bos_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.bos_token = self.tokenizer.eos_token
            self.tokenizer.bos_token_id = self.tokenizer.eos_token_id

        self.model.eval()

        # Update config dtype hints
        self.model.config.torch_dtype = model_dtype
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
            self.model.generation_config.bos_token_id = getattr(self.tokenizer, "bos_token_id", None)

        logger.info("Model initialization complete")

    def _build_prompt(self, state: str) -> str:
        """Combine system prompt (if provided) with environment state."""
        if not self.config.system_prompt:
            return state
        system = self.config.system_prompt.strip()
        return f"{system}\n\n{state}"

    def get_action(self, state: str) -> str:
        """
        Generate user response from state using the trained model.
        Uses chat template for proper formatting if available.

        Returns:
            Generated user response (action)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        # Build messages for chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            # Format as chat messages
            messages = []
            if self.config.system_prompt:
                messages.append({"role": "system", "content": self.config.system_prompt})
            messages.append({"role": "", "content": state})

            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to simple prompt building
            prompt = self._build_prompt(state)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=2048,  # Larger context window for full conversation
            truncation=True,
            padding=False
        )

        # Move to device
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the generated part
        generated_tokens = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Clean up response
        response = response.strip()

        # Remove any "User:" or "Agent:" prefixes if accidentally generated
        if response.startswith("User:"):
            response = response[5:].strip()
        if response.startswith("Agent:"):
            response = response[6:].strip()
        if response.startswith("Assistant:"):
            response = response[10:].strip()

        return response

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "not_initialized"}

        return {
            "model_name": self.config.model_name,
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
