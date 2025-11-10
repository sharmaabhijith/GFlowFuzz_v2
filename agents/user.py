#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
import yaml

# Ensure project modules are importable when this file runs as a script
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

_DEFAULT_AGENT_CONFIG = Path(__file__).resolve().parent / "agent_config.yaml"

@dataclass
class UserConfig:
    """Configuration for the user agent"""
    api_base_url: str
    model_name: str
    temperature: float
    max_tokens: int
    timeout: int
    system_prompt: str
    api_key: str


class UserAgent:
    """LLM-backed user agent that produces user utterances from conversation context."""

    def __init__(self, model_name: Optional[str] = None, temperature: Optional[float] = None):
        
        with open(_DEFAULT_AGENT_CONFIG, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        common = config.get("common", {})
        agent_entry = config.get("agents").get("user")
        config_data = {**common, **agent_entry}
        prompt_file = Path(Path(__file__).resolve().parent/agent_entry.get("system_prompt_path"))
        prompt_text = prompt_file.read_text(encoding="utf-8")
        config_data["system_prompt"] = prompt_text

        """Load configuration and initialise the LLM client."""
        env_path = os.path.join(Path(__file__).parent.parent, ".env")
        load_dotenv(env_path)

        if model_name:
            config_data["model_name"] = model_name
        if temperature:
            config_data["temperature"] = temperature

        api_key = os.environ.get('DEEPINFRA_API_KEY')
        self.config = UserConfig(
            api_base_url=config_data['api_base_url'],
            model_name=config_data['model_name'],
            temperature=config_data.get('temperature'),
            max_tokens=config_data.get('max_tokens'),
            timeout=config_data.get('timeout'),
            system_prompt=config_data.get('system_prompt', ''),
            api_key=api_key
        )
        # Initialize OpenAI client for the user's LLM
        self.openai_client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base_url
        )
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM API for generating user messages"""
        chat_completion = self.openai_client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=0.9
        )
        return chat_completion.choices[0].message.content
    
    def generate_user_message(
        self,
        conversation_history: List[Dict[str, str]],
        booking_objective: str,
    ) -> str:
        """Generate a natural user message based on conversation context"""
        
        # Build context from conversation history
        recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
        
        prompt = f"""You are a user trying to book a flight with this objective:
        {booking_objective}
        
        Recent conversation:
        {json.dumps(recent_history, indent=2)}
        
        Generate a brief, natural response. Return only the message, nothing else."""
        
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        return self._call_llm(messages)
    
