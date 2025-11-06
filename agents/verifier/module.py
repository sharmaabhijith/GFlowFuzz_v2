#!/usr/bin/env python3

import os
import sys
import json
import re
from pathlib import Path
import yaml
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
from agents.coder.module import SQLCoderAgent

@dataclass
class VerifierConfig:
    api_base_url: str
    model_name: str
    temperature: float
    max_tokens: int
    timeout: int
    max_retries: int
    system_prompt: str
    api_key: str

@dataclass
class VerificationResult:
    claim: str
    query: str
    exists: bool
    actual_data: Optional[Dict] = None
    discrepancies: Optional[List[str]] = None

class BookingVerifierAgent:
    """
    Verifier Agent that validates booking claims made during conversations
    """
    def __init__(self, config_path: str):
        env_path = os.path.join(Path(__file__).parent.parent.parent, ".env")
        load_dotenv(env_path)
        api_key = os.environ.get('DEEPINFRA_API_KEY')
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        self.config = VerifierConfig(
            api_base_url=config_data['api_base_url'],
            model_name=config_data['model_name'],
            temperature=config_data['temperature'],
            max_tokens=config_data['max_tokens'],
            timeout=config_data['timeout'],
            max_retries=config_data['max_retries'],
            system_prompt=config_data['system_prompt'],
            api_key=api_key
        )
        
        self.openai_client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base_url
        )

        self.coder_agent = SQLCoderAgent(
            os.path.join(Path(__file__).parent.parent, 'coder', 'config.yaml')
        )
        
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM API to generate verification queries"""
        response = self.openai_client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=0.9
        )
        return response.choices[0].message.content
    
    
    def verify_booking(self, booking_summary: str, mcp_client) -> Dict[str, Any]:
        """
        Verify ONLY the final booking summary by checking if it exists in the database.

        Args:
            booking_summary: The final booking summary text
            mcp_client: MCP client instance for database queries

        Returns:
            Dict with verification result and reward (1 if new booking, 0 if exists)
        """
        # Use coder agent to convert booking summary to SQL query
        sql_result = self.coder_agent.generate_sql_query(
            f"Check if this booking exists: {booking_summary}"
        )

        if not sql_result.get('success') or not sql_result.get('sql_query'):
            return {
                "verification_complete": False,
                "has_booking_summary": True,
                "reward": 0,
                "error": "Could not generate SQL query from booking summary"
            }

        query = sql_result['sql_query']
        print(query)

        # Execute the query to check if booking exists
        result = mcp_client.query_database(query)

        if not result.success:
            return {
                "verification_complete": False,
                "has_booking_summary": True,
                "reward": 0,
                "error": f"Database query failed: {result.error_message}"
            }

        # Parse the result to check if booking exists
        try:
            result_text = result.result.strip()
            if result_text.startswith('{'):
                query_result = json.loads(result_text)
                row_count = query_result.get('row_count', 0)
            else:
                row_count = 0

            # Determine reward based on whether booking exists
            if row_count > 0:
                # Booking exists in database - no reward
                return {
                    "verification_complete": True,
                    "has_booking_summary": True,
                    "booking_exists": True,
                    "reward": 0,
                    "message": "Booking already exists in database",
                    "matches_found": row_count
                }
            else:
                # Booking doesn't exist - give reward
                return {
                    "verification_complete": True,
                    "has_booking_summary": True,
                    "booking_exists": False,
                    "reward": 1,
                    "message": "New booking - not found in database"
                }

        except (json.JSONDecodeError, TypeError) as e:
            return {
                "verification_complete": False,
                "has_booking_summary": True,
                "reward": 0,
                "error": f"Failed to parse database result: {str(e)}"
            }