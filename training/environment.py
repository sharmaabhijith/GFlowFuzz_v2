import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "agents"))

from agents.chat.module import FlightBookingChatAgent
from agents.verifier.module import BookingVerifierAgent
from agents.coder.module import SQLCoderAgent
from agents.judge.module import PolicyJudgeAgent


@dataclass
class ConversationState:
    """Represents the current state of a conversation"""
    booking_context: Dict[str, Any]
    turn_count: int
    is_terminated: bool = False
    booking_objective: str = ""


@dataclass
class EnvironmentStep:
    """Result of an environment step"""
    state: ConversationState
    done: bool
    info: Dict[str, Any]


class BookingConversationEnvironment:
    """
    Unified environment for training user agents in conversation flow.
    Combines functionality from both original environment files.
    """

    def __init__(self, config: Dict[str, Any], auditor_config=None):
        """
        Initialize the environment

        Args:
            config: Environment configuration dictionary
            auditor_config: Configuration for auditor agent (optional)
        """
        self.config = config
        self.auditor_config = auditor_config
        self.max_conversation_length = config.get('max_conversation_length', 8)
        self.booking_agent_config = config.get('booking_agent_config')
        self.verifier_config = config.get('verifier_config')
        self.coder_config = config.get('coder_config', {
            'config_path': 'agents/coder/config.yaml'
        })
        self.judge_config = config.get('judge', {
            'config_path': 'agents/judge/config.yaml',
            'policy_set_path': 'policies/booking_policies.yaml',
        })

        self.logger = logging.getLogger(__name__)
        self.booking_agent = None
        self.verifier_agent = None
        self.coder_agent = None
        self.judge_agent: Optional[PolicyJudgeAgent] = None
        self.current_state = None
        self.conversation_count = 0
        self.total_rewards = []
        self.current_objective = None
        self.is_initialized = False
        self.final_booking_summary = None

    def initialize(self):
        """Initialize the environment and agents"""
        # Initialize booking agent
        chat_config_path = self.booking_agent_config['config_path']
        db_path = self.booking_agent_config['db_path']
        server_path = self.booking_agent_config['server_path']
        self.booking_agent = FlightBookingChatAgent(chat_config_path, db_path, server_path)
        self.booking_agent.initialize()
        # Initialize verifier agent
        verifier_config_path = self.verifier_config['config_path']
        self.verifier_agent = BookingVerifierAgent(verifier_config_path)
        # Initialize coder agent
        coder_config_path = self.coder_config['config_path']
        self.coder_agent = SQLCoderAgent(coder_config_path)
        judge_config_path = self.judge_config.get('config_path')
        self.judge_agent = PolicyJudgeAgent(judge_config_path)
        self.is_initialized = True

    def reset(self, booking_objective: Optional[str] = None) -> ConversationState:
        """
        Reset the environment for a new conversation

        Args:
            booking_objective: Optional specific booking objective for this conversation

        Returns:
            Initial conversation state
        """
        if not self.is_initialized:
            raise RuntimeError("Environment not initialized. Call initialize() first.")
        # Generate or use provided objective
        self.current_objective = booking_objective
        # Reset final booking summary
        self.final_booking_summary = None
        # Reset booking agent
        self.booking_agent.conversation_history = []
        self.booking_agent.booking_context = {
            "summary": "",
            "current_requirements": {},
            "search_history": [],
            "preferences": {}
        }
        # Create initial conversation state
        self.current_state = ConversationState(
            booking_context={},
            turn_count=0,
            is_terminated=False,
            booking_objective=booking_objective
        )
        return self.current_state

    def step(self, user_action: str) -> EnvironmentStep:
        """
        Execute a step in the environment

        Args:
            user_action: The user's message/action

        Returns:
            EnvironmentStep containing new state, done flag, and info
        """
        if self.current_state.is_terminated:
            raise ValueError("Environment is terminated. Call reset() to start a new conversation.")

        # Get booking agent response (it handles conversation history internally)
        booking_response = self.booking_agent._process_user_message(user_action)
        # Update state
        self.current_state.turn_count += 1
        if self.booking_agent:
            self.current_state.booking_context = self.booking_agent.booking_context.copy()
        # Check if conversation should terminate
        done = self._should_terminate(user_action, booking_response)
        self.current_state.is_terminated = done

        # Generate booking summary when conversation ends
        if done and not self.final_booking_summary:
            try:
                # Generate the final booking summary
                self.final_booking_summary = self.booking_agent._generate_booking_summary()
            except Exception as e:
                self.logger.warning(f"Could not generate booking summary: {e}")
                self.final_booking_summary = None

        # Create info dict
        info = {
            "turn_count": self.current_state.turn_count,
            "booking_objective": self.current_state.booking_objective,
            "conversation_length": len(self.booking_agent.conversation_history),
            "booking_summary": self.final_booking_summary if done else None
        }

        return EnvironmentStep(
            state=self.current_state,
            done=done,
            info=info
        )

    def _format_state(self, conversation_state: ConversationState) -> str:
        """
        Format the conversation state into a concise, structured prompt for the policy.

        Args:
            conversation_state: Current state of the conversation

        Returns:
            Formatted string containing complete context for policy
        """
        state_parts: List[str] = []

        # Objective / goal
        goal = conversation_state.booking_objective or "<no objective>"
        state_parts.append(f"[GOAL] {goal}")

        # Conversation history (keep recent turns, truncate long utterances)
        history = self.booking_agent.conversation_history
        max_history_length = 30
        if len(history) > max_history_length:
            skipped = len(history) - max_history_length
            state_parts.append(f"[CONTEXT] Earlier conversation truncated ({skipped} messages omitted)")
            history = history[-max_history_length:]

        if history:
            state_parts.append("[DIALOG]")
            for msg in history:
                prefix = "U:" if msg["role"] == "user" else "A:"
                content = msg["content"]
                if len(content) > 250:
                    content = content[:247] + "..."
                state_parts.append(f"{prefix} {content}")
        else:
            state_parts.append("[DIALOG] <start>")

        # Booking context
        ctx = conversation_state.booking_context or {}
        summary = ctx.get("summary")
        if summary:
            trimmed_summary = summary if len(summary) <= 250 else summary[:247] + "..."
            state_parts.append(f"[SUMMARY] {trimmed_summary}")

        requirements = ctx.get("current_requirements") or {}
        if requirements:
            req_items = [f"{k}:{v}" for k, v in requirements.items() if v]
            if req_items:
                state_parts.append(f"[REQUIREMENTS] {', '.join(req_items)}")

        preferences = ctx.get("preferences") or {}
        if preferences:
            pref_items = [f"{k}:{v}" for k, v in preferences.items() if v]
            if pref_items:
                state_parts.append(f"[PREFERENCES] {', '.join(pref_items)}")

        # Metadata for next turn
        state_parts.append(f"[TURN] {conversation_state.turn_count + 1}")
        state_parts.append("[NEXT] User:")

        return "\n".join(state_parts)

    def _should_terminate(self, user_action: str, booking_response: str) -> bool:
        """Determine if the conversation should terminate"""
        # More precise termination conditions
        # Check for explicit quit commands (not just "thank you")
        user_wants_to_quit = any(phrase in user_action.lower() for phrase in [
            'quit', 'exit', 'bye', 'goodbye',
            'stop', 'end conversation', 'that\'s all',
            'no more', 'i\'m done', 'cancel'
        ])
        # Check if "thank you" is used as a closing statement (with context)
        thank_you_closing = (
            'thank you' in user_action.lower() and
            any(word in user_action.lower() for word in ['bye', 'all', 'done', 'help', 'booking'])
        )
        # Check for successful booking completion
        booking_complete = any(indicator in booking_response.lower() for indicator in [
            'booking confirmed', 'reservation complete', 'booking reference',
            'confirmation number', 'all set', 'booking successful',
            'transaction completed', 'payment processed'
        ])
        # Check max length
        max_length_reached = self.current_state.turn_count >= self.max_conversation_length

        return user_wants_to_quit or thank_you_closing or booking_complete or max_length_reached


    def _generate_booking_objective(self) -> str:
        """Generate a simple fallback booking objective if none is provided."""
        self.logger.warning("No booking objective provided; using fallback objective")
        return "I need to fly from New York to London"
