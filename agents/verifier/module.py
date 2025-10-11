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
        
    
    def _extract_booking_claims(self, conversation_history: List[Dict[str, str]]) -> str:
        """Extract all booking-related claims from conversation history"""
        booking_claims = []
        
        for message in conversation_history:
            if message["role"] == "assistant":
                content = message["content"].lower()
                # Look for flight numbers, prices, dates, routes mentioned
                if any(keyword in content for keyword in ['flight', 'book', 'price', '$', 'departure', 'arrival']):
                    booking_claims.append(message["content"])
        
        return "\n\n".join(booking_claims)
    
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
    
    def _parse_verification_queries(self, llm_response: str) -> List[Dict[str, str]]:
        """Parse the LLM response to extract verification queries"""
        try:
            # Try to extract JSON array from the response
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            if json_match:
                queries = json.loads(json_match.group(0))
                return queries
            else:
                # Fallback: try to parse the entire response as JSON
                queries = json.loads(llm_response)
                return queries if isinstance(queries, list) else []
        except (json.JSONDecodeError, AttributeError) as e:
            return []
    
    def _clean_sql_query(self, query: str) -> str:
        """Clean and validate SQL query"""
        # Remove any markdown formatting
        query = re.sub(r'```sql\n?', '', query)
        query = re.sub(r'```\n?', '', query)
        # Remove trailing semicolons to avoid multi-statement issues
        query = query.rstrip(';')
        # Normalize whitespace
        query = ' '.join(query.split())
        return query.strip()
    
    def generate_verification_queries(self, conversation_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Generate SQL queries to verify booking claims from conversation

        Returns:
            List of dicts with 'claim' and 'query' keys
        """
        # Extract booking claims from conversation
        booking_claims = self._extract_booking_claims(conversation_history)

        if not booking_claims:
            return []

        prompt = f"""
        Analyze the following booking claims from a flight booking conversation and generate SQL queries to verify each claim:

        BOOKING CLAIMS:
        {booking_claims}

        Generate verification queries for EACH specific claim (flight numbers, dates, prices, routes).
        Return as JSON array with format: [{{"claim": "description", "query": "SQL query"}}]
        """

        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Generate verification queries using LLM
        response = self._call_llm(messages)
        queries = self._parse_verification_queries(response)

        # Clean each query
        for query_dict in queries:
            if 'query' in query_dict:
                query_dict['query'] = self._clean_sql_query(query_dict['query'])

        return queries
    
    def verify_final_booking_only(self, booking_summary: str, mcp_client, coder_agent) -> Dict[str, Any]:
        """
        Verify ONLY the final booking summary by checking if it exists in the database.

        Args:
            booking_summary: The final booking summary text
            mcp_client: MCP client instance for database queries
            coder_agent: Coder agent to convert summary to SQL

        Returns:
            Dict with verification result and reward (1 if new booking, 0 if exists)
        """

        # If no booking summary, return no reward
        if not booking_summary or booking_summary.strip() == "":
            return {
                "verification_complete": True,
                "has_booking_summary": False,
                "reward": 0,
                "message": "No booking summary to verify"
            }

        try:
            # Use coder agent to convert booking summary to SQL query
            sql_result = coder_agent.generate_sql_query(
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

        except Exception as e:
            return {
                "verification_complete": False,
                "has_booking_summary": True,
                "reward": 0,
                "error": f"Verification failed: {str(e)}"
            }

    def verify_bookings(self, conversation_history: List[Dict[str, str]], mcp_client) -> Dict[str, Any]:
        """
        Verify all booking claims from the conversation
        
        Args:
            conversation_history: List of conversation messages
            mcp_client: MCP client instance for database queries
        
        Returns:
            Verification report with results
        """
        
        # Generate verification queries
        verification_queries = self.generate_verification_queries(conversation_history)
        
        if not verification_queries:
            return {
                "verification_complete": True,
                "claims_found": 0,
                "message": "No specific booking claims found to verify"
            }
        
        verification_results = []
        
        for query_info in verification_queries:
            claim = query_info.get('claim', 'Unknown claim')
            query = query_info.get('query', '')
            
            if not query:
                continue
            
            # Execute verification query
            result = mcp_client.query_database(query)
            
            if not result.success:
                verification_results.append({
                    "claim": claim,
                    "status": "error",
                    "error": result.error_message
                })
                continue
            
            # Parse query results
            try:
                result_text = result.result.strip()
                if result_text.startswith('{'):
                    query_result = json.loads(result_text)
                    results_data = query_result.get('results', [])
                    row_count = query_result.get('row_count', 0)
                else:
                    results_data = []
                    row_count = 0
                
                # Determine verification status
                if row_count > 0:
                    verification_results.append({
                        "claim": claim,
                        "status": "verified",
                        "matches_found": row_count,
                        "sample_data": results_data[:3] if results_data else []
                    })
                else:
                    verification_results.append({
                        "claim": claim,
                        "status": "not_found",
                        "message": "No matching records found in database"
                    })
                    
            except (json.JSONDecodeError, TypeError) as e:
                verification_results.append({
                    "claim": claim,
                    "status": "error",
                    "error": f"Failed to parse result: {str(e)}"
                })
        
        # Prepare summary report
        verified_count = sum(1 for r in verification_results if r['status'] == 'verified')
        not_found_count = sum(1 for r in verification_results if r['status'] == 'not_found')
        error_count = sum(1 for r in verification_results if r['status'] == 'error')
        
        report = {
            "verification_complete": True,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_claims": len(verification_results),
                "verified": verified_count,
                "not_found": not_found_count,
                "errors": error_count
            },
            "details": verification_results
        }
        
        # Add warnings for unverified claims
        if not_found_count > 0:
            report["warnings"] = []
            for result in verification_results:
                if result['status'] == 'not_found':
                    report["warnings"].append(f"⚠️ Unverified claim: {result['claim']}")
        
        return report
    
    def format_verification_report(self, report: Dict[str, Any]) -> str:
        """Format the verification report for display"""
        if not report.get("verification_complete"):
            return "Verification incomplete"
        
        output = []
        output.append("\n" + "="*60)
        output.append("📋 BOOKING VERIFICATION REPORT")
        output.append("="*60)
        
        summary = report.get("summary", {})
        output.append(f"\n📊 Summary:")
        output.append(f"   Total Claims Checked: {summary.get('total_claims', 0)}")
        output.append(f"   ✅ Verified: {summary.get('verified', 0)}")
        output.append(f"   ❌ Not Found: {summary.get('not_found', 0)}")
        output.append(f"   ⚠️ Errors: {summary.get('errors', 0)}")
        
        # Detailed results
        output.append("\n📝 Detailed Results:")
        output.append("-"*40)
        
        for i, detail in enumerate(report.get("details", []), 1):
            output.append(f"\n{i}. Claim: {detail['claim']}")
            
            if detail['status'] == 'verified':
                output.append(f"   Status: ✅ VERIFIED")
                output.append(f"   Matches Found: {detail.get('matches_found', 0)}")
                if detail.get('sample_data'):
                    output.append(f"   Sample Match: {json.dumps(detail['sample_data'][0], indent=6)[:200]}...")
            elif detail['status'] == 'not_found':
                output.append(f"   Status: ❌ NOT FOUND IN DATABASE")
                output.append(f"   Note: {detail.get('message', 'No matching records')}")
            else:
                output.append(f"   Status: ⚠️ ERROR")
                output.append(f"   Error: {detail.get('error', 'Unknown error')}")
        
        # Warnings section
        if report.get("warnings"):
            output.append("\n⚠️ WARNINGS:")
            output.append("-"*40)
            for warning in report["warnings"]:
                output.append(warning)
        
        # Conclusion
        output.append("\n" + "="*60)
        if summary.get('not_found', 0) > 0:
            output.append("⚠️ Some booking claims could not be verified in the database.")
            output.append("This may indicate that the assistant provided incorrect information.")
        elif summary.get('verified', 0) == summary.get('total_claims', 0):
            output.append("✅ All booking claims have been successfully verified!")
        
        output.append("="*60 + "\n")
        
        return "\n".join(output)
