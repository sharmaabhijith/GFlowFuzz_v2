#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import yaml
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sql-coder-agent")

@dataclass
class CoderConfig:
    api_base_url: str
    model_name: str
    temperature: float
    max_tokens: int
    timeout: int
    max_retries: int
    system_prompt: str
    api_key: str

class SQLCoderAgent:
    """
    SQL Coder Agent that converts natural language flight requests into SQL queries
    """
    def __init__(self, config_path: str):
        env_path = os.path.join(Path(__file__).parent.parent.parent, ".env")
        load_dotenv(env_path)
        api_key = os.environ.get('DEEPINFRA_API_KEY')
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        self.config = CoderConfig(
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
        logger.info("Successfully loaded SQL Coder configuration")

    def _clean_sql_query(self, raw_sql: str) -> str:
        """Clean and format the SQL query"""
        sql = re.sub(r'```sql\n?', '', raw_sql)
        sql = re.sub(r'```\n?', '', sql)
        sql = ' '.join(sql.split())
        sql = sql.rstrip(';') + ';'
        return sql.strip()

    def _validate_sql_query(self, sql_query: str) -> bool:
        """Validate the generated SQL query for safety and correctness"""
        sql_lower = sql_query.lower()
        dangerous_keywords = ['insert', 'update', 'delete', 'drop', 'create', 'alter', 'truncate']
        if any(keyword in sql_lower for keyword in dangerous_keywords):
            logger.warning("SQL contains dangerous operations")
            return False
        if not sql_lower.strip().startswith('select'):
            logger.warning("SQL is not a SELECT query")
            return False
        required_elements = ['from flights f', 'join cities']
        if not all(element in sql_lower for element in required_elements):
            logger.warning("SQL missing required elements")
            return False
        if sql_lower.count('select') > 1 and 'union' not in sql_lower:
            logger.warning("Multiple SELECT statements without UNION")
            return False
        return True

    async def _call_llm(self, messages) -> str:
        """Call the LLM API to generate SQL"""
        response = self.openai_client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=0.9
        )
        return response.choices[0].message.content

    def get_city_airport_mapping(self) -> Dict[str, str]:
        """Get the city to airport code mapping"""
        return self.city_to_airport.copy()

    async def generate_sql_query(self, user_request: str) -> Dict[str, Any]:
        """
        Convert natural language flight request to SQL query
        
        Returns:
            Dict with 'sql_query', 'success', and optional 'error' keys
        """
        prompt = f"""
        Convert this natural language flight search request into a SQL query:
        
        "{user_request}"
        """
        
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Generate SQL using LLM
        response = await self._call_llm(messages)
        # Clean and validate the SQL
        sql_query = self._clean_sql_query(response)
        
        if self._validate_sql_query(sql_query):
            logger.info(f"Generated SQL: {sql_query[:100]}...")
            return {
                "sql_query": sql_query,
                "success": True
            }
        else:
            logger.error(f"Invalid SQL generated: {sql_query}")
            return {
                "sql_query": None,
                "success": False,
                "error": "Generated SQL failed validation"
            }