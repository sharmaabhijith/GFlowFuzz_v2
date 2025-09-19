#!/usr/bin/env python3

"""
Stateless AutoUser Agent - Pure Policy Implementation
Acts as a pure function mapping states to actions without maintaining any internal state.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AutoUserConfig:
    """Configuration for the AutoUser agent"""
    model_name: str
    tokenizer_name: Optional[str]
    max_length: int
    temperature: float
    top_p: float
    do_sample: bool
    device: str


class AutoUserAgent:
    """
    Stateless AutoUser Agent - Pure Policy Function

    This agent acts as a pure function π(s) → a, mapping states to actions
    without maintaining any internal conversation state.

    All conversation context must be provided in the state string from the environment.
    """

    def __init__(self, config: AutoUserConfig):
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

    def initialize_model(self):
        """Load the model and tokenizer"""
        # Determine device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        # Load tokenizer
        tokenizer_name = self.config.tokenizer_name or self.config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        self.model.eval()

    async def get_action(self, state: str) -> str:
        """
        Returns:
            Generated user response (action)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        # Tokenize the complete state
        inputs = self.tokenizer.encode(
            state,
            return_tensors="pt",
            max_length=512,  # Larger context window for full state
            truncation=True,
            padding=True
        ).to(self.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                attention_mask=inputs.ne(self.tokenizer.pad_token_id)
            )

        # Decode only the generated part
        generated_tokens = outputs[0][inputs.shape[-1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Clean up response
        response = response.strip()

        # Remove any "User:" or "Agent:" prefixes if accidentally generated
        if response.startswith("User:"):
            response = response[5:].strip()
        if response.startswith("Agent:"):
            response = response[6:].strip()

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