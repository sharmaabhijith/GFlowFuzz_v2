#!/usr/bin/env python3

import asyncio
import json
import os
import sys
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(os.path.join(Path(__file__).parent.parent.parent,"mcp-client"))
sys.path.append(os.path.join(Path(__file__).parent.parent,"coder"))
from mcp_client import MCPClient, ToolResult
from coder.module import SQLCoderAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat-agent")

@dataclass
class ChatConfig:
    api_base_url: str
    model_name: str
    temperature: float
    max_tokens: int
    timeout: int
    max_retries: int
    system_prompt: str
    api_key: str

class FlightBookingChatAgent:
    def __init__(self, config_path: str, db_path: str, server_path: str):
        env_path = os.path.join(Path(__file__).parent.parent.parent, ".env")
        load_dotenv(env_path)
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        api_key = os.environ.get('DEEPINFRA_API_KEY')
        self.config = ChatConfig(
            api_base_url=config_data['api_base_url'],
            model_name=config_data['model_name'],
            temperature=config_data['temperature'],
            max_tokens=config_data['max_tokens'],
            timeout=config_data['timeout'],
            max_retries=config_data['max_retries'],
            system_prompt=config_data['system_prompt'],
            api_key=api_key
        )
        # Why do we need to send db_path to clinent. Shouldn't access to the server path be enough?
        self.mcp_client = MCPClient(server_path, db_path)
        self.conversation_history = []
        # Initialize OpenAI client for DeepInfra
        self.openai_client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base_url
        )
        # Initialize SQL Coder Agent
        coder_config_path = os.path.join(Path(__file__).parent.parent, "coder", "config.yaml")
        self.sql_coder = SQLCoderAgent(str(coder_config_path))
        
        # Keywords for detecting flight-related queries
        self.travel_keywords = [
            'flight', 'fly', 'book', 'booking', 'travel', 'trip', 'journey',
            'departure', 'arrival', 'airport', 'airline', 'ticket', 'reservation',
            'schedule', 'itinerary', 'boarding', 'layover', 'direct', 'connecting',
            'round trip', 'one way', 'return', 'economy', 'business', 'first class'
        ]
    
    async def initialize(self):
        """Initialize the chat agent and test MCP connection"""
        logger.info("Testing MCP connection...")
        connection_ok = await self.mcp_client.test_connection()
        if not connection_ok:
            raise RuntimeError("Failed to connect to MCP server")
        tools = await self.mcp_client.get_available_tools()
        logger.info(f"Successfully connected to MCP server with {len(tools)} tools")
        return True
        

    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM API using OpenAI client with retry logic"""
        chat_completion = self.openai_client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=0.9
        )
        return chat_completion.choices[0].message.content

    async def _execute_database_query(self, query: str, params: Optional[List] = None) -> ToolResult:
        """Execute a database query using MCP client"""
        return await self.mcp_client.query_database(query, params)


    async def _process_user_message(self, user_message: str) -> str:
        """Process user message and generate response"""
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Check if this is a flight-related request
        if self._is_flight_related(user_message):
            return await self._handle_flight_request(user_message)
        else:
            # Regular conversation
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                *self.conversation_history[-10:]  # Keep last 10 messages for context
            ]
            response = await self._call_llm(messages)
            self.conversation_history.append({"role": "assistant", "content": response})
            return response

    def _is_flight_related(self, text: str) -> bool:
        """Check if the user message is flight-related"""
        text_lower = text.lower()
        has_travel_keyword = any(keyword in text_lower for keyword in self.travel_keywords)
        return has_travel_keyword

    async def _handle_flight_request(self, user_message: str) -> str:
        """Handle flight-related requests using SQL Coder Agent and MCP queries"""
        logger.info(f"Generating SQL for: {user_message}")
        sql_result = await self.sql_coder.generate_sql_query(user_message)
        
        if not sql_result.get("success"):
            logger.error(f"SQL generation failed: {sql_result.get('error')}")
            response = "I had trouble understanding your flight search request. Could you please rephrase it with more specific details like departure city, destination, and any dates?"
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        
        sql_query = sql_result.get("sql_query")
        logger.info(f"Generated SQL: {sql_query[:100]}...")
        result = await self._execute_database_query(sql_query)
        
        if not result.success:
            logger.error(f"Database query failed: {result.error_message}")
            response = "I encountered an issue while searching for flights. Please try again with different search criteria or check your request details."
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        
        # Parse the JSON result from MCP server
        try:
            result_text = result.result.strip()
            logger.info(f"Raw MCP result: {result_text[:200]}...")
            
            # The MCP server returns JSON with metadata, extract the results array
            if result_text.startswith('{'):
                # Parse the full JSON response from MCP server
                mcp_response = json.loads(result_text)
                # Extract just the results array from the response
                if isinstance(mcp_response, dict) and 'results' in mcp_response:
                    flights_data = mcp_response['results']
                elif isinstance(mcp_response, list):
                    flights_data = mcp_response
                else:
                    # Last resort: look for any list in the response
                    flights_data = []
                    for key, value in mcp_response.items():
                        if isinstance(value, list) and len(value) > 0:
                            flights_data = value
                            break       
            elif result_text.startswith('['):
                flights_data = json.loads(result_text)
            else:
                import re
                json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
                if json_match:
                    flights_data = json.loads(json_match.group(0))
                else:
                    flights_data = []
                
            # Ensure flights_data is a list
            if not isinstance(flights_data, list):
                flights_data = []
                
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            logger.error(f"Failed to parse database result: {e}")
            logger.error(f"Raw result was: {result.result}")
            flights_data = []  # Initialize flights_data to prevent UnboundLocalError
            response = "I encountered an issue while processing the flight data. Let me search for your Toronto to Dubai flight using a different approach."
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        
        # Check if we got results
        if not flights_data or len(flights_data) == 0:
            response = "I couldn't find any flights matching your criteria. Would you like to try with:"
            response += "\n• Different dates (perhaps a few days earlier or later)?"
            response += "\n• Nearby airports or different destinations?"
            response += "\n• Different cabin class (economy, business, or first)?"
            response += "\n• More flexible departure/arrival times?"
            response += "\n\nPlease let me know how you'd like to adjust your search!"
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        
        # Format the flight results for the user
        response = await self._format_flight_results(flights_data, user_message)
        self.conversation_history.append({"role": "assistant", "content": response})
        return response
    
    async def _format_flight_results(self, flights_data: List[Dict], user_query: str) -> str:
        """Format flight search results for user display"""
        try:
            # Prepare context for LLM to format the results
            flights_info = f"Flight search results for query: '{user_query}'\n\n"
            flights_info += f"Found {len(flights_data)} flight(s):\n\n"
            
            for i, flight in enumerate(flights_data, 1):
                flights_info += f"Flight {i}:\n"
                flights_info += f"  Flight Number: {flight.get('flight_number', 'N/A')}\n"
                flights_info += f"  Airline: {flight.get('airline_name', 'N/A')}\n"
                flights_info += f"  Route: {flight.get('departure_airport', 'N/A')} → {flight.get('arrival_airport', 'N/A')}\n"
                flights_info += f"  Departure: {flight.get('departure_time', 'N/A')}\n"
                flights_info += f"  Arrival: {flight.get('arrival_time', 'N/A')}\n"
                flights_info += f"  Duration: {flight.get('duration_minutes', 'N/A')} minutes\n"
                flights_info += f"  Class: {flight.get('cabin_class', 'N/A')}\n"
                flights_info += f"  Price: ${flight.get('price', 'N/A')} {flight.get('currency', 'USD')}\n"
                flights_info += f"  Available Seats: {flight.get('available_seats', 'N/A')}\n"
                flights_info += f"  Aircraft: {flight.get('aircraft_type', 'N/A')}\n"
                
                # Optional amenities
                amenities = []
                if flight.get('meal_service'):
                    amenities.append('Meal Service')
                if flight.get('wifi_available'):
                    amenities.append('WiFi')
                if flight.get('baggage_allowance'):
                    amenities.append(f"Baggage: {flight.get('baggage_allowance')}")
                
                if amenities:
                    flights_info += f"  Amenities: {', '.join(amenities)}\n"
                
                flights_info += "\n"
            
            # Use LLM to format this nicely for the user
            format_prompt = [
                {"role": "system", "content": self.config.system_prompt + "\n\nFormat the flight results in a clear, user-friendly way. Include booking instructions and ask if they'd like to book any specific flight."},
                {"role": "user", "content": f"Please format these flight search results nicely:\n\n{flights_info}"}
            ]
            
            formatted_response = await self._call_llm(format_prompt)
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting flight results: {e}")
            return f"Found {len(flights_data)} flights, but encountered an error formatting the results. Please try your search again."
    
    async def chat_loop(self):
        """Interactive chat loop for the flight booking agent"""
        print("\n🛫 Welcome to the Flight Booking Assistant!")
        print("I can help you search for flights and make bookings.")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n✈️ Thank you for using Flight Booking Assistant! Safe travels!")
                    break
                if not user_input:
                    continue
                print("\n🤖 Assistant: ", end="")
                response = await self._process_user_message(user_input)
                print(response)      
            except KeyboardInterrupt:
                print("\n\n✈️ Chat ended by user. Thank you for using Flight Booking Assistant!")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"\n❌ Sorry, I encountered an error: {e}")
                print("Please try again or type 'quit' to exit.")
