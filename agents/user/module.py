#!/usr/bin/env python3

import json
import os
import sys
import random
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from openai import OpenAI
import time
from dotenv import load_dotenv

# Rich imports for beautiful console output
from rich.console import Console
from rich.panel import Panel  
from rich.text import Text
from rich.table import Table
from rich import box

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(os.path.join(Path(__file__).parent.parent.parent, "mcp-client"))
sys.path.append(os.path.join(Path(__file__).parent.parent, "chat"))

from mcp_client import MCPClient
from chat.module import FlightBookingChatAgent

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

class FlightBookingUserAgent:
    """User agent that simulates natural human conversations for booking flights"""
    
    def __init__(self, config_path: str, console: Optional[Console] = None):
        """Initialize the user agent"""
        env_path = os.path.join(Path(__file__).parent.parent.parent, ".env")
        load_dotenv(env_path)
        
        # Load configuration
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        api_key = os.environ.get('DEEPINFRA_API_KEY')
        self.config = UserConfig(
            api_base_url=config_data['api_base_url'],
            model_name=config_data['model_name'],
            temperature=config_data.get('temperature', 0.7),
            max_tokens=config_data.get('max_tokens', 2048),
            timeout=config_data.get('timeout', 30),
            system_prompt=config_data.get('system_prompt', ''),
            api_key=api_key
        )
        
        # Initialize OpenAI client for the user's LLM
        self.openai_client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base_url
        )
        
        # Initialize console for rich output
        self.console = console if console else Console()
        
        # Conversation state
        self.conversation_history: List[Dict[str, str]] = []
        self.booking_objective: str = ""
        self.has_greeted = False
    
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
    
    def generate_booking_objective(self) -> str:
        """Generate a random booking objective for this conversation"""
        prompt = """Generate a simple, realistic flight booking request that a user might have.
        Include basic details like:
        - Where they want to go (any real cities)
        - Approximate time frame (use year 2026)
        - Number of passengers (if more than 1)
        - Any specific preference (optional)
        
        Keep it to 1-2 sentences max. Be natural and conversational.
        Examples:
        - "I need to fly from Boston to Seattle next month for a conference"
        - "Looking for 2 tickets to Miami in February 2026"
        - "I want to book a business class flight to London in March 2026"
        
        Generate ONE booking objective. Return only the objective, nothing else."""
        
        messages = [
            {"role": "system", "content": "You are generating a random flight booking request."},
            {"role": "user", "content": prompt}
        ]
        
        return self._call_llm(messages)
    
    def generate_user_message(self, context: Dict[str, Any]) -> str:
        """Generate a natural user message based on conversation context"""
        
        # Build context from conversation history
        recent_history = self.conversation_history[-4:] if len(self.conversation_history) > 4 else self.conversation_history
        
        prompt = f"""You are a user trying to book a flight with this objective:
        {self.booking_objective}
        
        Recent conversation:
        {json.dumps(recent_history, indent=2)}
        
        Based on the conversation:
        - If the agent asks for information, provide it briefly
        - If shown flight options, pick one or ask a short question
        - If ready to book, confirm briefly
        - If booking is complete, thank them and say "quit" or "exit"
        
        IMPORTANT RULES:
        1. Keep responses SHORT (1-2 sentences max)
        2. Be natural and conversational
        3. Don't repeat information already provided
        4. If you've achieved your booking goal, end with "quit" or "exit"
        5. Never be verbose - real users type short messages
        
        Generate a brief, natural response. Return only the message, nothing else."""
        
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        return self._call_llm(messages)
    
    def should_end_conversation(self) -> bool:
        """Determine if the conversation should end based on context"""
        if len(self.conversation_history) < 4:
            return False
        
        # Check if booking seems complete
        last_messages = " ".join([msg["content"].lower() for msg in self.conversation_history[-3:]])
        
        booking_complete_indicators = [
            "booking confirmed",
            "reservation complete",
            "thank you for booking",
            "booking reference",
            "confirmation number",
            "all set",
            "booking successful"
        ]
        
        if any(indicator in last_messages for indicator in booking_complete_indicators):
            return True
        
        # End if conversation is too long
        if len(self.conversation_history) > 16:  # 8 exchanges
            return True
        
        return False
    
    def simulate_booking_conversation(self, chat_agent: FlightBookingChatAgent) -> Dict[str, Any]:
        """Simulate a complete booking conversation with the chat agent"""
        
        # Generate random booking objective
        self.booking_objective = self.generate_booking_objective()
        self.conversation_history = []
        self.has_greeted = False
        
        # Display booking objective
        self.console.print()
        objective_panel = Panel(
            f"[bold cyan]Booking Objective:[/bold cyan] {self.booking_objective}",
            title="[bold yellow]🎯 USER OBJECTIVE[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED
        )
        self.console.print(objective_panel)
        
        max_turns = 10
        conversation_ended = False
        
        for turn in range(max_turns):
            
            # Check if we should end
            if turn > 0 and self.should_end_conversation():
                # Generate closing message
                user_message = "Great, thank you for your help! quit"
            else:
                # Generate user message based on context
                user_message = self.generate_user_message({
                    "turn": turn,
                    "history": self.conversation_history
                })
            
            # Display user message
            user_text = Text()
            user_text.append("👤 USER", style="bold cyan")
            user_panel = Panel(
                user_message,
                title=user_text,
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2)
            )
            self.console.print(user_panel)
            
            # Add to history
            self.conversation_history.append({"role": "user", "content": user_message})
            
            # Check if user wants to quit
            if "quit" in user_message.lower() or "exit" in user_message.lower():
                conversation_ended = True
                self.console.print(Panel(
                    "[bold green]✅ User ended conversation properly[/bold green]",
                    border_style="green",
                    box=box.ROUNDED
                ))
                break
            
            # Get response from chat agent
            try:
                response = chat_agent._process_user_message(user_message)
                
                # Display assistant response
                assistant_text = Text()
                assistant_text.append("🤖 ASSISTANT", style="bold green")
                
                # Truncate long responses for display
                display_response = response
                if len(response) > 500:
                    display_response = response[:500] + "\n\n[Response truncated...]"
                
                assistant_panel = Panel(
                    display_response,
                    title=assistant_text,
                    border_style="green",
                    box=box.ROUNDED,
                    padding=(1, 2)
                )
                self.console.print(assistant_panel)
                
                # Add to history
                self.conversation_history.append({"role": "assistant", "content": response})
                
                # Small delay to simulate thinking
                time.sleep(1)
                
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                break
        
        if not conversation_ended:
            # Force quit if max turns reached
            quit_message = "Thanks! quit"
            self.console.print(Panel(
                f"[yellow]Auto-ending conversation:[/yellow] {quit_message}",
                border_style="yellow"
            ))
            self.conversation_history.append({"role": "user", "content": quit_message})
            
            # Send quit message to chat agent
            try:
                chat_agent._process_user_message(quit_message)
            except:
                pass
        
        # Return conversation summary
        return {
            "objective": self.booking_objective,
            "conversation_length": len(self.conversation_history),
            "properly_ended": conversation_ended,
            "conversation_history": self.conversation_history
        }
    
    def run_user_simulation(self, chat_agent: FlightBookingChatAgent) -> Dict[str, Any]:
        """Run a single user simulation"""
        self.console.print()
        self.console.print(Panel(
            "[bold magenta]Starting Natural User Conversation Simulation[/bold magenta]",
            box=box.DOUBLE,
            border_style="magenta"
        ))
        
        result = self.simulate_booking_conversation(chat_agent)
        
        # Display summary
        self.console.print()
        summary_table = Table(
            title="[bold green]Conversation Summary[/bold green]",
            box=box.ROUNDED,
            border_style="green"
        )
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Objective", result["objective"][:50] + "..." if len(result["objective"]) > 50 else result["objective"])
        summary_table.add_row("Messages", str(result["conversation_length"]))
        summary_table.add_row("Properly Ended", "✅ Yes" if result["properly_ended"] else "❌ No")
        
        self.console.print(summary_table)
        
        return result
