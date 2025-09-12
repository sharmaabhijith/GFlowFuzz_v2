#!/usr/bin/env python3

import asyncio
import json
import os
import sys
from pathlib import Path
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(os.path.join(Path(__file__).parent.parent.parent,"mcp-client"))
sys.path.append(os.path.join(Path(__file__).parent.parent,"coder"))
sys.path.append(os.path.join(Path(__file__).parent.parent,"verifier"))
from mcp_client import MCPClient, ToolResult
from coder.module import SQLCoderAgent
from verifier.module import BookingVerifierAgent
from datetime import datetime


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
        self.conversation_history = []  # Keep for backward compatibility
        
        # Initialize booking context that will be maintained by LLM
        self.booking_context = {
            "summary": "",
            "current_requirements": {},
            "search_history": [],
            "preferences": {}
        }
        # Initialize OpenAI client for DeepInfra
        self.openai_client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base_url
        )
        # Initialize SQL Coder Agent
        coder_config_path = os.path.join(Path(__file__).parent.parent, "coder", "config.yaml")
        self.sql_coder = SQLCoderAgent(str(coder_config_path))
        
        # Initialize Booking Verifier Agent
        verifier_config_path = os.path.join(Path(__file__).parent.parent, "verifier", "config.yaml")
        self.verifier = BookingVerifierAgent(str(verifier_config_path))
        
        # Keywords for detecting flight-related queries
        self.travel_keywords = [
            'flight', 'fly', 'book', 'booking', 'travel', 'trip', 'journey',
            'departure', 'arrival', 'airport', 'airline', 'ticket', 'reservation',
            'schedule', 'itinerary', 'boarding', 'layover', 'direct', 'connecting',
            'round trip', 'one way', 'return', 'economy', 'business', 'first class'
        ]
    
    async def initialize(self):
        """Initialize the chat agent and test MCP connection"""
        # Testing MCP connection...
        connection_ok = await self.mcp_client.test_connection()
        if not connection_ok:
            raise RuntimeError("Failed to connect to MCP server")
        tools = await self.mcp_client.get_available_tools()
        # Successfully connected to MCP server
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
    
    async def _update_booking_context(self, user_message: str) -> None:
        """Use LLM to automatically extract and maintain booking context"""
        # Create prompt for LLM to extract booking information
        extraction_prompt = f"""
        Based on the user's message and conversation history, extract and update the booking information.
        
        Current booking context:
        {json.dumps(self.booking_context, indent=2)}
        
        User's new message: {user_message}
        
        Please update the booking context with any new information from the user's message.
        Extract the following if present:
        - Departure city/airport
        - Arrival city/airport
        - Travel dates
        - Number of passengers
        - Class preference (economy/business/first)
        - Price constraints
        - Any specific requirements (direct flights, wifi, meals, etc.)
        
        Return the updated booking context as a JSON object with these keys:
        - summary: Brief summary of what the user wants
        - current_requirements: Dictionary of specific requirements
        - preferences: User preferences
        
        Respond ONLY with the JSON object, no other text.
        """
        
        messages = [
            {"role": "system", "content": "You are a booking information extractor. Extract and maintain booking context from user messages."},
            {"role": "user", "content": extraction_prompt}
        ]
        
        # Add recent conversation for context
        if len(self.conversation_history) > 0:
            recent_context = "Recent conversation:\n"
            for msg in self.conversation_history[-5:]:
                recent_context += f"{msg['role']}: {msg['content']}\n"
            messages[0]["content"] += "\n\n" + recent_context
        
        try:
            response = await self._call_llm(messages)
            # Try to parse the response as JSON
            updated_context = json.loads(response)
            
            # Merge with existing context
            if "summary" in updated_context:
                self.booking_context["summary"] = updated_context["summary"]
            if "current_requirements" in updated_context:
                self.booking_context["current_requirements"].update(updated_context["current_requirements"])
            if "preferences" in updated_context:
                self.booking_context["preferences"].update(updated_context["preferences"])
                
        except (json.JSONDecodeError, Exception) as e:
            # If extraction fails, just keep the existing context
            pass


    async def _process_user_message(self, user_message: str) -> str:
        """Process user message and generate response with LLM-managed state"""
        # Update booking context using LLM
        await self._update_booking_context(user_message)
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Check if this is a flight-related request
        if self._is_flight_related(user_message):
            response = await self._handle_flight_request(user_message)
        else:
            # Regular conversation with booking context
            messages = [
                {"role": "system", "content": self.config.system_prompt + "\n\nCurrent Booking Context: " + json.dumps(self.booking_context, indent=2)},
            ]
            
            # Add recent conversation history (last 10 messages)
            recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
            messages.extend(recent_history)
            
            response = await self._call_llm(messages)
        
        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response

    def _is_flight_related(self, text: str) -> bool:
        """Check if the user message is flight-related"""
        text_lower = text.lower()
        has_travel_keyword = any(keyword in text_lower for keyword in self.travel_keywords)
        return has_travel_keyword

    async def _handle_flight_request(self, user_message: str) -> str:
        """Handle flight-related requests using SQL Coder Agent and MCP queries"""
        # Generating SQL for user message
        
        # Create enriched context for SQL coder from booking context
        enriched_history = []
        
        # Add booking context as system message
        enriched_history.append({
            "role": "system", 
            "content": f"Current Booking Information:\n{json.dumps(self.booking_context, indent=2)}"
        })
        
        # Add recent conversation messages
        recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        enriched_history.extend(recent_history)
        
        # Pass enriched context to SQL coder
        sql_result = await self.sql_coder.generate_sql_query(user_message, enriched_history)
        
        if not sql_result.get("success"):
            response = "I had trouble understanding your flight search request. Could you please rephrase it with more specific details like departure city, destination, and any dates?"
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        
        sql_query = sql_result.get("sql_query")
        # Generated SQL query
        result = await self._execute_database_query(sql_query)
        
        if not result.success:
            response = "I encountered an issue while searching for flights. Please try again with different search criteria or check your request details."
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        
        # Parse the JSON result from MCP server
        try:
            result_text = result.result.strip()
            # Received MCP result
            
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
            flights_data = []  # Initialize flights_data to prevent UnboundLocalError
            response = "I encountered an issue while processing the flight data. Let me search for your Toronto to Dubai flight using a different approach."
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        
        # Update booking context with search results
        self.booking_context["search_history"].append({
            "query": user_message,
            "results_count": len(flights_data),
            "timestamp": datetime.now().isoformat()
        })
        
        # Check if we got results
        if not flights_data or len(flights_data) == 0:
            response = "I couldn't find any flights matching your criteria. Would you like to try with:"
            response += "\n• Different dates (perhaps a few days earlier or later)?"
            response += "\n• Nearby airports or different destinations?"
            response += "\n• Different cabin class (economy, business, or first)?"
            response += "\n• More flexible departure/arrival times?"
            response += "\n\nPlease let me know how you'd like to adjust your search!"
            # No need to add to history here as _process_user_message handles it
            return response
        
        # Format the flight results for the user
        response = await self._format_flight_results(flights_data, user_message)
        # No need to add to history here as _process_user_message handles it
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
            
            # Use LLM to format with booking context
            format_prompt = [
                {"role": "system", "content": self.config.system_prompt + "\n\nFormat the flight results in a clear, user-friendly way. Include booking instructions and ask if they'd like to book any specific flight.\n\nCurrent Booking Context: " + json.dumps(self.booking_context, indent=2)}
            ]
            
            # Add recent messages for context
            recent_history = self.conversation_history[-5:] if len(self.conversation_history) > 5 else self.conversation_history
            format_prompt.extend(recent_history)
            
            # Add the current request
            format_prompt.append({"role": "user", "content": f"Please format these flight search results nicely:\n\n{flights_info}"})
            
            formatted_response = await self._call_llm(format_prompt)
            return formatted_response
            
        except Exception as e:
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
                    print("\n✈️ Thank you for using Flight Booking Assistant!")
                    
                    # Generate booking summary and run verification if there's booking information
                    if len(self.conversation_history) > 0 and self._has_booking_claims():
                        # First, generate and display booking summary
                        print("\n📋 BOOKING SUMMARY")
                        print("=" * 60)
                        print("Generating summary of your flight booking conversation...")
                        
                        try:
                            booking_summary = await self._generate_booking_summary()
                            print("\n" + booking_summary)
                            print("=" * 60)
                            
                            # Save summary to conversation for verification
                            self.conversation_history.append({
                                "role": "system", 
                                "content": f"BOOKING SUMMARY: {booking_summary}"
                            })
                            
                        except Exception as e:
                            print("\n⚠️ Could not generate booking summary")
                        
                        # Now run verification
                        print("\n🔍 VERIFICATION PROCESS")
                        print("=" * 60)
                        print("Now verifying the booking information against our database...")
                        
                        try:
                            # Run verification
                            verification_report = await self.verifier.verify_bookings(
                                self.conversation_history, 
                                self.mcp_client
                            )
                            
                            # Format and display the report
                            formatted_report = self.verifier.format_verification_report(verification_report)
                            print(formatted_report)
                            
                            # Save both summary and verification report to file
                            report_path = os.path.join(Path(__file__).parent.parent.parent, "booking_verification_report.json")
                            full_report = {
                                "booking_summary": booking_summary if 'booking_summary' in locals() else "Not generated",
                                "verification_report": verification_report
                            }
                            with open(report_path, 'w') as f:
                                json.dump(full_report, f, indent=2)
                            print(f"📄 Full report saved to: {report_path}")
                            
                        except Exception as e:
                            print(f"\n⚠️ Could not complete verification: {e}")
                    
                    print("\n✈️ Safe travels!")
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
                print(f"\n❌ Sorry, I encountered an error: {e}")
                print("Please try again or type 'quit' to exit.")
    
    def _has_booking_claims(self) -> bool:
        """Check if conversation history contains any booking claims"""
        for message in self.conversation_history:
            if message["role"] == "assistant":
                content = message["content"].lower()
                # Check for indicators of booking information
                if any(indicator in content for indicator in [
                    'flight', 'price', '$', 'departure', 'arrival', 
                    'booking', 'seat', 'class', 'airline'
                ]):
                    return True
        return False
    
    async def _generate_booking_summary(self) -> str:
        """Generate a precise summary of final booking information only"""
        # Use the maintained booking context and conversation history
        summary_prompt = f"""
        Based on the booking context, provide ONLY the final booking information.
        
        Current Booking Context:
        {json.dumps(self.booking_context, indent=2)}
        
        Create a BRIEF summary with ONLY these details (if available):
        • Origin: [city]
        • Destination: [city]
        • Date: [date]
        • Passengers: [number]
        • Class: [class type]
        • Selected Flight: [flight number if chosen]
        • Price: [amount if available]
        
        ONLY include information that was actually discussed. Do NOT include:
        - Search history
        - User preferences discussion
        - Multiple options presented
        - Conversation details
        - Any explanations or narrative
        
        Keep it extremely concise - just the final booking facts.
        """
        
        messages = [
            {"role": "system", "content": "You are a booking summary assistant. Extract ONLY the final booking information. Be extremely concise and factual. No explanations, just booking details."},
            {"role": "user", "content": summary_prompt}
        ]
        
        try:
            summary = await self._call_llm(messages)
            return summary
        except Exception as e:
            return "Unable to generate booking summary due to an error."
