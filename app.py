#!/usr/bin/env python3
import os
import asyncio
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import time

# Rich imports for beautiful terminal interface
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import box
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.columns import Columns
from rich.rule import Rule
from rich.style import Style
from rich.theme import Theme

# Custom theme for professional look
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "highlight": "bold magenta",
    "user": "bold blue",
    "assistant": "bold orange1",
    "system": "dim cyan"
})

# Add agents directory to Python path
chat_path = os.path.join(str(Path(__file__).parent), "agents", "chat")
user_path = os.path.join(str(Path(__file__).parent), "agents", "user")

# Import chat agent
sys.path.insert(0, chat_path)
from module import FlightBookingChatAgent
sys.path.remove(chat_path)

# Import user agent
sys.path.insert(0, user_path)
import importlib
import module as user_module
importlib.reload(user_module)
FlightBookingUserAgent = user_module.FlightBookingUserAgent
sys.path.remove(user_path)

# Initialize console with custom theme
console = Console(theme=custom_theme)

class ProfessionalFlightBookingApp:
    def __init__(self):
        self.console = console
        self.project_root = Path(__file__).parent
        self.env_path = self.project_root / ".env"
        self.mode = None
        self.agent = None
        self.session_start = datetime.now()
        
        # Load environment variables immediately
        load_dotenv(self.env_path)
        
    def display_welcome(self):
        """Display professional welcome banner"""
        self.console.clear()
        
        # Simple, clean welcome message
        self.console.print()
        welcome_panel = Panel(
            Align.center(
                "[bold cyan]✈️  FLIGHT BOOKING SYSTEM  ✈️[/bold cyan]\n\n" +
                "[dim]AI-Powered Assistant & Testing Platform[/dim]"
            ),
            box=box.DOUBLE,
            style="cyan",
            padding=(1, 3)
        )
        self.console.print(welcome_panel)
        self.console.print()
        
        # Display system status
        self._display_system_status()
        
    def _display_system_status(self):
        """Display system status indicators"""
        # Check database status
        db_exists = False
        db_paths = [
            os.path.join(self.project_root, "database", "flights.db"),
            os.path.join(self.project_root, "flights.db"),
            os.path.join(self.project_root, "flight_bookings.db")
        ]
        for path in db_paths:
            if Path(path).exists():
                db_exists = True
                break
        
        db_status = "✓" if db_exists else "✗"
        db_color = "green" if db_exists else "yellow"
        
        # Check API key status
        api_status = "✓" if os.environ.get('DEEPINFRA_API_KEY') else "✗"
        api_color = "green" if os.environ.get('DEEPINFRA_API_KEY') else "red"
        
        # Simple status line
        status_text = f"[dim]System Status: Database [{db_color}]{db_status}[/{db_color}] | API Key [{api_color}]{api_status}[/{api_color}] | Config [green]✓[/green][/dim]"
        self.console.print(Align.center(status_text))
        self.console.print()
        
    def select_mode_simple(self) -> str:
        """Simple mode selection after welcome"""
        self.console.print(Rule(style="cyan"))
        self.console.print()
        self.console.print("[bold cyan]Choose your mode:[/bold cyan]")
        self.console.print()
        self.console.print("  [bold]1.[/bold] Chat Mode - Talk with the booking assistant")
        self.console.print("  [bold]2.[/bold] Simulation Mode - Run automated user tests")
        self.console.print()
        
        while True:
            choice = Prompt.ask(
                "[bold green]Enter your choice (1 or 2)[/bold green]",
                choices=["1", "2"],
                default="1"
            )
            if choice == "1":
                self.console.print("\n[success]✓ Starting Chat Mode...[/success]\n")
                return "chat"
            elif choice == "2":
                self.console.print("\n[success]✓ Starting Simulation Mode...[/success]\n")
                return "user"
                
    async def initialize_environment(self) -> bool:
        """Initialize environment with progress indicators"""
        self.console.print()
        self.console.print(Rule("System Initialization", style="cyan"))
        
        tasks = [
            ("Loading environment variables", self._load_env),
            ("Verifying API credentials", self._verify_api),
            ("Setting up paths", self._setup_paths),
            ("Checking database", self._check_database)
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            transient=True
        ) as progress:
            
            main_task = progress.add_task("[cyan]Initializing...", total=len(tasks))
            
            for task_name, task_func in tasks:
                progress.update(main_task, description=f"[cyan]{task_name}...")
                result = await task_func()
                if not result:
                    self.console.print(f"\n[error]✗ {task_name} failed[/error]")
                    return False
                progress.advance(main_task)
                await asyncio.sleep(0.3)  # Small delay for visual effect
        
        self.console.print("\n[success]✓ All systems initialized successfully![/success]")
        return True
    
    async def _load_env(self):
        """Load environment variables"""
        # Already loaded in __init__, just verify it's loaded
        return True
    
    async def _verify_api(self):
        """Verify API key exists"""
        return bool(os.environ.get('DEEPINFRA_API_KEY'))
    
    async def _setup_paths(self):
        """Set up required paths"""
        self.config_path_chat = os.path.join(self.project_root, "agents", "chat", "config.yaml")
        self.config_path_user = os.path.join(self.project_root, "agents", "user", "config.yaml")
        self.server_path = os.path.join(self.project_root, "mcp-server", "database_server.py")
        # Check for database in multiple locations
        possible_db_paths = [
            os.path.join(self.project_root, "database", "flights.db"),
            os.path.join(self.project_root, "flights.db"),
            os.path.join(self.project_root, "flight_bookings.db")
        ]
        for db_path in possible_db_paths:
            if Path(db_path).exists():
                self.database_path = db_path
                break
        else:
            # Use default path even if it doesn't exist yet
            self.database_path = os.path.join(self.project_root, "database", "flights.db")
        return True
    
    async def _check_database(self):
        """Check if database exists or can be created"""
        db_path = Path(self.database_path)
        if db_path.exists():
            return True
        # Try to create the database directory if it doesn't exist
        try:
            db_dir = db_path.parent
            if not db_dir.exists():
                db_dir.mkdir(parents=True, exist_ok=True)
            # Database will be created when first accessed
            return True
        except Exception as e:
            self.console.print(f"[warning]Database not found at {self.database_path}[/warning]")
            self.console.print(f"[dim]Will attempt to create it when needed[/dim]")
            return True  # Continue anyway, database might be created on first use
        
    def _has_booking_claims(self, agent) -> bool:
        """Check if conversation contains booking information"""
        for message in agent.conversation_history:
            if message["role"] == "assistant":
                content = message["content"].lower()
                if any(indicator in content for indicator in [
                    'flight', 'price', '$', 'departure', 'arrival', 
                    'booking', 'seat', 'class', 'airline'
                ]):
                    return True
        return False
    
    async def run_chat_mode(self):
        """Professional chat interface"""
        self.console.print()
        self.console.print(Rule("Interactive Chat Session", style="cyan"))
        
        # Session info header
        session_info = Table(show_header=False, box=None, padding=(0, 2))
        session_info.add_column("", style="dim")
        session_info.add_column("", style="bright_white")
        session_info.add_row("📅 Session Started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        session_info.add_row("🆔 Session ID", f"CHAT-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        self.console.print(session_info)
        self.console.print()
        
        # Initialize chat agent with loading animation
        with self.console.status("[bold cyan]Initializing AI assistant...[/bold cyan]", spinner="dots12"):
            agent = FlightBookingChatAgent(
                config_path=str(self.config_path_chat),
                db_path=str(self.database_path),
                server_path=str(self.server_path)
            )
            await agent.initialize()
            await asyncio.sleep(1)  # Visual effect
            
        # Welcome message
        self.console.print(Panel(
            "[bold cyan]Welcome! I'm your flight booking assistant.[/bold cyan]\n\n" +
            "How can I help you today? You can:\n" +
            "• Search for flights between cities\n" +
            "• Check prices and availability\n" +
            "• Make bookings\n\n" +
            "[dim]Type 'help' for commands | Say 'thanks, quit' to exit with booking summary[/dim]",
            box=box.ROUNDED,
            style="cyan"
        ))
        
        # Chat loop
        message_count = 0
        while True:
            try:
                # User input with styled prompt
                user_input = Prompt.ask("\n[user]You[/user]")
                message_count += 1
                
                # Check for exit commands - especially polite exits with thank you
                user_input_lower = user_input.lower()
                
                # Check if user is saying thank you with quit/exit in the same message
                has_thanks = any(word in user_input_lower for word in ['thank', 'thanks', 'thx', 'ty'])
                has_exit = any(word in user_input_lower for word in ['quit', 'exit', 'bye', 'goodbye'])
                
                # If user says thank you and quit/exit together, trigger verification
                if has_thanks and has_exit:
                    # Process this final message first
                    with self.console.status(
                        "[dim cyan]Processing final message...[/dim cyan]", 
                        spinner="dots12"
                    ):
                        response = await agent._process_user_message(user_input)
                        await asyncio.sleep(0.5)
                    
                    # Display response
                    response_panel = Panel(
                        response,
                        title=f"[assistant]✈️ Assistant[/assistant] [dim](Message #{message_count})[/dim]",
                        box=box.ROUNDED,
                        style="orange1",
                        padding=(1, 2)
                    )
                    self.console.print(response_panel)
                    
                    # Now handle exit with verification
                    await self._handle_chat_exit(agent, message_count)
                    break
                
                # Regular exit without thank you
                elif user_input_lower in ['exit', 'quit', 'bye', 'goodbye', 'q']:
                    await self._handle_chat_exit(agent, message_count)
                    break
                
                # Help command
                if user_input.lower() == 'help':
                    self._display_help()
                    continue
                
                # Process message with thinking indicator
                with self.console.status(
                    "[dim cyan]AI is processing your request...[/dim cyan]", 
                    spinner="dots12"
                ):
                    response = await agent._process_user_message(user_input)
                    await asyncio.sleep(0.5)  # Small delay for realism
                
                # Display response in professional panel
                response_panel = Panel(
                    response,
                    title=f"[assistant]✈️ Assistant[/assistant] [dim](Message #{message_count})[/dim]",
                    box=box.ROUNDED,
                    style="orange1",
                    padding=(1, 2)
                )
                self.console.print(response_panel)
                
            except KeyboardInterrupt:
                if Confirm.ask("\n[warning]Interrupt detected. Exit chat?[/warning]"):
                    await self._handle_chat_exit(agent, message_count)
                    break
                continue
            except Exception as e:
                self._display_error(str(e))
    
    def _display_help(self):
        """Display help information"""
        help_panel = Panel(
            """[bold cyan]Available Commands:[/bold cyan]

[yellow]Flight Search:[/yellow]
  • "Find flights from [city] to [city]"
  • "Show me flights on [date]"
  • "I need [n] tickets"

[yellow]Preferences:[/yellow]
  • "Business/Economy/First class"
  • "Direct flights only"
  • "Morning/Evening flights"

[yellow]Ending Your Session:[/yellow]
  • [cyan]"Thanks, quit"[/cyan] - Exit with booking summary & database verification
  • [cyan]"Thank you, exit"[/cyan] - Exit with booking summary & database verification
  • [cyan]exit/quit[/cyan] - Quick exit (no summary or verification)
  • [cyan]help[/cyan] - Show this help

[dim]💡 Tip: Say "thank you" with "quit" to get your booking summary![/dim]""",
            title="[bold]Help Information[/bold]",
            box=box.DOUBLE,
            style="cyan"
        )
        self.console.print(help_panel)
    
    def _display_error(self, error_msg: str):
        """Display error in professional format"""
        error_panel = Panel(
            f"[error]⚠️ {error_msg}[/error]\n\n[dim]Please try again or type 'help' for assistance[/dim]",
            title="[error]Error[/error]",
            box=box.HEAVY,
            style="red"
        )
        self.console.print(error_panel)
    
    async def _handle_chat_exit(self, agent, message_count):
        """Handle chat session exit with summary"""
        self.console.print()
        self.console.print(Rule("Session Complete", style="cyan"))
        
        # Session statistics
        session_duration = datetime.now() - self.session_start
        stats_table = Table(show_header=False, box=box.MINIMAL, padding=(0, 2))
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="yellow")
        
        stats_table.add_row("📊 Total Messages", str(message_count * 2))
        stats_table.add_row("⏱️ Session Duration", str(session_duration).split('.')[0])
        stats_table.add_row("🤖 AI Model", "Claude-4-Sonnet")
        
        self.console.print(stats_table)
        
        # Generate booking summary if applicable
        if len(agent.conversation_history) > 0 and self._has_booking_claims(agent):
            await self._generate_session_summary(agent)
        
        # Farewell message
        farewell = Panel(
            """[bold yellow]Thank you for using Flight Booking System![/bold yellow]

✈️ [dim]Safe travels and see you next time![/dim]""",
            box=box.DOUBLE_EDGE,
            style="cyan",
            padding=(1, 2)
        )
        self.console.print("\n", farewell)
    
    async def _generate_session_summary(self, agent):
        """Generate and display session summary with verification"""
        self.console.print("\n[highlight]📋 Generating Booking Summary...[/highlight]")
        
        try:
            with self.console.status("[dim]Processing conversation history...[/dim]", spinner="dots12"):
                booking_summary = await agent._generate_booking_summary()
                await asyncio.sleep(1)
            
            summary_panel = Panel(
                booking_summary,
                title="[bold]Booking Summary[/bold]",
                box=box.DOUBLE,
                style="green",
                padding=(1, 2)
            )
            self.console.print(summary_panel)
            
            # Run verification process
            self.console.print("\n[highlight]🔍 Verifying Booking Information...[/highlight]")
            
            try:
                with self.console.status("[dim]Checking booking details against database...[/dim]", spinner="dots12"):
                    # Run verification using the verifier agent from chat module
                    verification_report = await agent.verifier.verify_bookings(
                        agent.conversation_history, 
                        agent.mcp_client
                    )
                    await asyncio.sleep(1)
                
                # Format and display verification report
                formatted_report = agent.verifier.format_verification_report(verification_report)
                
                verification_panel = Panel(
                    formatted_report,
                    title="[bold]Verification Report[/bold]",
                    box=box.DOUBLE,
                    style="orange1",
                    padding=(1, 2)
                )
                self.console.print(verification_panel)
                
                # Save both summary and verification
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_file = f"booking_verification_{timestamp}.json"
                full_report = {
                    "booking_summary": booking_summary,
                    "verification_report": verification_report,
                    "timestamp": timestamp
                }
                with open(report_file, 'w') as f:
                    json.dump(full_report, f, indent=2)
                self.console.print(f"\n[success]✓ Full report saved to {report_file}[/success]")
                
            except Exception as e:
                self.console.print(f"[warning]Could not complete verification: {e}[/warning]")
                # Still save the summary even if verification fails
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                summary_file = f"booking_summary_{timestamp}.txt"
                with open(summary_file, 'w') as f:
                    f.write(booking_summary)
                self.console.print(f"[success]✓ Summary saved to {summary_file}[/success]")
            
        except Exception as e:
            self.console.print(f"[warning]Could not generate summary: {e}[/warning]")
    
    async def run_user_mode(self):
        """Run user simulation mode with professional UI"""
        self.console.print()
        self.console.print(Rule("Automated Simulation Mode", style="magenta"))
        self.console.print("\n[cyan]Starting automated user simulation...[/cyan]\n")
        
        # Initialize agents
        self.console.print("\n[highlight]Initializing simulation environment...[/highlight]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console,
            transient=True
        ) as progress:
            
            task = progress.add_task("[cyan]Setting up agents...", total=2)
            
            # Initialize user agent
            user_agent = FlightBookingUserAgent(
                config_path=str(self.config_path_user),
                console=self.console
            )
            progress.advance(task)
            
            # Initialize chat agent
            chat_agent = FlightBookingChatAgent(
                config_path=str(self.config_path_chat),
                db_path=str(self.database_path),
                server_path=str(self.server_path)
            )
            await chat_agent.initialize()
            progress.advance(task)
        
        self.console.print("[success]✓ Simulation environment ready![/success]\n")
        
        # Run single simulation
        results = await self._run_simulations(user_agent, chat_agent, 1)
        
        # Display results
        self._display_simulation_report(results)
        
        # Run verification if there are booking claims
        if self._has_booking_claims(chat_agent):
            await self._run_simulation_verification(chat_agent)
    
    async def _run_simulations(self, user_agent, chat_agent, iterations=1):
        """Run simulation with progress tracking"""
        results = []
        
        with Progress(
            TextColumn("[bold blue]Simulation Progress"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("• {task.description}"),
            console=self.console
        ) as progress:
            
            main_task = progress.add_task(
                "[cyan]Running simulation...", 
                total=1
            )
                
            # Run single simulation
            result = await user_agent.run_user_simulation(chat_agent)
            results.append(result)
            
            progress.advance(main_task)
        
        return results
    
    def _display_simulation_report(self, results):
        """Display professional simulation report"""
        self.console.print()
        self.console.print(Rule("Simulation Report", style="green"))
        
        # Calculate statistics
        total = len(results)
        successful = sum(1 for r in results if r.get('properly_ended', False))
        avg_messages = sum(r.get('conversation_length', 0) for r in results) / total if total > 0 else 0
        
        # Create report card
        report = f"""
[bold cyan]📊 SIMULATION RESULTS[/bold cyan]

[yellow]Performance Metrics:[/yellow]
  • Total Simulations: [bold]{total}[/bold]
  • Successful: [green]{successful}[/green]
  • Failed: [red]{total - successful}[/red]
  • Success Rate: [bold]{(successful/total*100):.1f}%[/bold]
  
[yellow]Conversation Analytics:[/yellow]
  • Avg. Messages: [bold]{avg_messages:.1f}[/bold]
  • Completion Rate: [bold]{(successful/total*100):.1f}%[/bold]

[dim]Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]"""
        
        report_panel = Panel(
            report,
            title="[bold]Final Report[/bold]",
            box=box.DOUBLE_EDGE,
            style="green",
            padding=(1, 2)
        )
        self.console.print(report_panel)
        
        # Save detailed report
        self._save_simulation_report(results)
    
    async def _run_simulation_verification(self, chat_agent):
        """Run verification after simulation completes"""
        self.console.print()
        self.console.print(Rule("Verification Process", style="orange1"))
        
        try:
            # Generate booking summary first
            self.console.print("\n[highlight]📋 Generating Booking Summary...[/highlight]")
            with self.console.status("[dim]Processing conversation...[/dim]", spinner="dots12"):
                booking_summary = await chat_agent._generate_booking_summary()
                await asyncio.sleep(0.5)
            
            summary_panel = Panel(
                booking_summary,
                title="[bold]Simulation Booking Summary[/bold]",
                box=box.DOUBLE,
                style="green",
                padding=(1, 2)
            )
            self.console.print(summary_panel)
            
            # Run verification
            self.console.print("\n[highlight]🔍 Verifying Against Database...[/highlight]")
            with self.console.status("[dim]Checking booking claims...[/dim]", spinner="dots12"):
                verification_report = await chat_agent.verifier.verify_bookings(
                    chat_agent.conversation_history,
                    chat_agent.mcp_client
                )
                await asyncio.sleep(0.5)
            
            # Display verification report
            formatted_report = chat_agent.verifier.format_verification_report(verification_report)
            verification_panel = Panel(
                formatted_report,
                title="[bold]Simulation Verification Report[/bold]",
                box=box.DOUBLE,
                style="orange1",
                padding=(1, 2)
            )
            self.console.print(verification_panel)
            
            # Save verification report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"simulation_verification_{timestamp}.json"
            full_report = {
                "booking_summary": booking_summary,
                "verification_report": verification_report,
                "timestamp": timestamp,
                "mode": "simulation"
            }
            with open(report_file, 'w') as f:
                json.dump(full_report, f, indent=2)
            self.console.print(f"\n[success]✓ Verification report saved to {report_file}[/success]")
            
        except Exception as e:
            self.console.print(f"[warning]Could not complete verification: {e}[/warning]")
    
    def _save_simulation_report(self, results):
        """Save detailed simulation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"simulation_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.console.print(f"\n[success]✓ Detailed report saved to {report_file}[/success]")
                
    async def run(self):
        """Main application runner"""
        try:
            self.display_welcome()
            
            if not await self.initialize_environment():
                self.console.print("\n[error]✗ Failed to initialize. Please check your configuration.[/error]")
                return
            
            # Check if we should start with auto mode (from command line)
            if hasattr(self, 'start_with_auto') and self.start_with_auto:
                # Skip mode selection and go directly to simulation
                await self.run_user_mode()
            else:
                # Ask user to select mode
                mode = self.select_mode_simple()
                if mode == "chat":
                    await self.run_chat_mode()
                else:
                    await self.run_user_mode()
                
        except KeyboardInterrupt:
            self.console.print("\n\n[warning]⚠️ Application terminated by user[/warning]")
        except Exception as e:
            self.console.print(f"\n[error]✗ Critical error: {e}[/error]")
            if self.console.is_terminal:
                import traceback
                self.console.print(traceback.format_exc(), style="dim red")

def main():
    """Entry point for the professional flight booking application"""
    parser = argparse.ArgumentParser(
        description="Professional Flight Booking System - AI-Powered Assistant"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Start with automated simulation"
    )
    
    args = parser.parse_args()
    
    app = ProfessionalFlightBookingApp()
    app.start_with_auto = args.auto if hasattr(args, 'auto') else False
    
    # Run the application
    asyncio.run(app.run())

if __name__ == "__main__":
    main()