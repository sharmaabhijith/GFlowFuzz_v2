#!/usr/bin/env python3
"""
PPO Training Application for Auto User Agent

This application replaces the manual user agent with a trainable auto user agent
while maintaining the exact same framework flow as the original app.py.

Key features:
- Uses auto user agent instead of manual user interaction
- Maintains chat agent and verifier agent integration
- Leverages verifier agent feedback as PPO rewards
- Follows the same conversation -> verification -> reward cycle
"""

import os
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Rich imports for UI (matching original app)
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.rule import Rule
from rich import box

# Training imports
from algorithms.ppo_trainer import PPOTrainer, PPOConfig
from environment.booking_environment import BookingConversationEnvironment

# Agent imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from agents.auto_user.module import AutoUserAgent, AutoUserConfig
from agents.chat.module import FlightBookingChatAgent
from agents.verifier.module import BookingVerifierAgent

class PPOTrainingApp:
    """
    PPO Training Application that maintains the original app.py framework
    but substitutes the auto user agent for PPO training.
    """

    def __init__(self):
        self.console = Console()
        self.project_root = Path(__file__).parent.parent
        self.env_path = self.project_root / ".env"

        # Load environment variables
        load_dotenv(self.env_path)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for training"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ppo_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def display_welcome(self):
        """Display training welcome banner"""
        self.console.clear()
        self.console.print()

        welcome_panel = Panel(
            "[bold cyan]🤖  PPO TRAINING SYSTEM  🤖[/bold cyan]\n\n" +
            "[dim]Training Auto User Agent with Verifier Feedback[/dim]",
            box=box.DOUBLE,
            style="cyan",
            padding=(1, 3)
        )
        self.console.print(welcome_panel)
        self.console.print()

    async def initialize_training_environment(self):
        """Initialize all training components"""
        self.console.print(Rule("Training Environment Initialization", style="cyan"))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console,
            transient=True
        ) as progress:

            task = progress.add_task("[cyan]Setting up training environment...", total=4)

            # 1. Setup paths and config
            progress.update(task, description="[cyan]Configuring paths...")
            self.config_path_chat = os.path.join(self.project_root, "agents", "chat", "config.yaml")
            self.config_path_verifier = os.path.join(self.project_root, "agents", "verifier", "config.yaml")
            self.server_path = os.path.join(self.project_root, "mcp-server", "database_server.py")

            # Find database
            possible_db_paths = [
                os.path.join(self.project_root, "database", "flights.db"),
                os.path.join(self.project_root, "flights.db"),
                os.path.join(self.project_root, "flight_bookings.db")
            ]
            self.database_path = None
            for db_path in possible_db_paths:
                if Path(db_path).exists():
                    self.database_path = db_path
                    break
            if not self.database_path:
                self.database_path = os.path.join(self.project_root, "database", "flights.db")

            progress.advance(task)

            # 2. Create PPO configuration
            progress.update(task, description="[cyan]Configuring PPO trainer...")
            self.ppo_config = PPOConfig(
                model_name="gpt2",
                learning_rate=1e-4,
                batch_size=4,
                max_epochs=10,
                conversations_per_epoch=20,
                eval_conversations=5,
                use_verifier_rewards=True,
                output_dir="./ppo_training_output"
            )
            progress.advance(task)

            # 3. Create environment configuration
            progress.update(task, description="[cyan]Configuring conversation environment...")
            self.environment_config = {
                'max_conversation_length': 8,
                'booking_agent_config': {
                    'config_path': str(self.config_path_chat),
                    'db_path': str(self.database_path),
                    'server_path': str(self.server_path)
                },
                'verifier_config': {
                    'config_path': str(self.config_path_verifier)
                }
            }
            progress.advance(task)

            # 4. Initialize PPO trainer
            progress.update(task, description="[cyan]Initializing PPO trainer...")
            self.ppo_trainer = PPOTrainer(
                config=self.ppo_config,
                environment_config=self.environment_config
            )
            await self.ppo_trainer.initialize()
            progress.advance(task)

        self.console.print("\n[success]✓ Training environment initialized successfully![/success]")

    async def run_ppo_training(self):
        """Run the PPO training process"""
        self.console.print()
        self.console.print(Rule("PPO Training Process", style="green"))

        # Display training configuration
        config_panel = Panel(
            f"""[bold cyan]Training Configuration:[/bold cyan]
                [yellow]Model:[/yellow] {self.ppo_config.model_name}
                [yellow]Epochs:[/yellow] {self.ppo_config.max_epochs}
                [yellow]Conversations per Epoch:[/yellow] {self.ppo_config.conversations_per_epoch}
                [yellow]Learning Rate:[/yellow] {self.ppo_config.learning_rate}
                [yellow]Batch Size:[/yellow] {self.ppo_config.batch_size}
                [yellow]Verifier Rewards:[/yellow] {'Enabled' if self.ppo_config.use_verifier_rewards else 'Disabled'}

                [dim]Output Directory: {self.ppo_config.output_dir}[/dim]""",
                            title="[bold]Configuration[/bold]",
                            box=box.ROUNDED,
                            style="green"
        )
        self.console.print(config_panel)
        self.console.print()

        # Start training
        try:
            await self.ppo_trainer.train()

            # Display final results
            stats = self.ppo_trainer.get_training_statistics()
            self._display_training_results(stats)

        except KeyboardInterrupt:
            self.console.print("\n[warning]Training interrupted by user[/warning]")
            self._save_interrupted_training()
        except Exception as e:
            self.console.print(f"\n[error]Training failed: {e}[/error]")
            self.logger.error(f"Training failed: {e}", exc_info=True)

    def _display_training_results(self, stats):
        """Display final training results"""
        self.console.print()
        self.console.print(Rule("Training Complete", style="green"))

        epoch_rewards = stats['training_stats']['epoch_rewards']
        eval_rewards = stats['training_stats']['eval_rewards']

        if epoch_rewards:
            final_train_reward = epoch_rewards[-1]
            best_eval_reward = max(eval_rewards) if eval_rewards else 0.0

            results_panel = Panel(
                f"""[bold green]🎉 Training Completed Successfully![/bold green]

[yellow]Final Training Results:[/yellow]
  • Total Epochs: [bold]{stats['current_epoch'] + 1}[/bold]
  • Total Training Steps: [bold]{stats['global_step']}[/bold]
  • Final Training Reward: [bold]{final_train_reward:.3f}[/bold]
  • Best Evaluation Reward: [bold]{best_eval_reward:.3f}[/bold]

[yellow]Model Saved To:[/yellow]
  • Best Model: [cyan]{self.ppo_config.output_dir}/best_model[/cyan]
  • Final Model: [cyan]{self.ppo_config.output_dir}/final_model[/cyan]
  • Checkpoints: [cyan]{self.ppo_config.output_dir}/checkpoints[/cyan]

[dim]Training logs available in: {self.ppo_config.output_dir}/training.log[/dim]""",
                title="[bold]Training Results[/bold]",
                box=box.DOUBLE_EDGE,
                style="green",
                padding=(1, 2)
            )
            self.console.print(results_panel)
        else:
            self.console.print("[warning]No training results available[/warning]")

    def _save_interrupted_training(self):
        """Save training state when interrupted"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(
                self.ppo_config.output_dir,
                "checkpoints",
                f"interrupted_{timestamp}"
            )

            # Save current training state
            if hasattr(self.ppo_trainer, 'model') and self.ppo_trainer.model:
                os.makedirs(checkpoint_path, exist_ok=True)
                self.ppo_trainer.model.save_pretrained(checkpoint_path)
                self.ppo_trainer.tokenizer.save_pretrained(checkpoint_path)

                self.console.print(f"[success]Training state saved to: {checkpoint_path}[/success]")

        except Exception as e:
            self.console.print(f"[error]Failed to save interrupted training: {e}[/error]")

    async def run_demo_conversation(self):
        """Run a demo conversation with the trained model"""
        self.console.print()
        self.console.print(Rule("Demo Conversation", style="magenta"))

        # Load best model if available
        best_model_path = os.path.join(self.ppo_config.output_dir, "best_model")
        if not Path(best_model_path).exists():
            best_model_path = os.path.join(self.ppo_config.output_dir, "final_model")

        if Path(best_model_path).exists():
            self.console.print(f"[cyan]Loading trained model from: {best_model_path}[/cyan]")

            # Initialize auto user with trained model
            auto_user_config = AutoUserConfig(
                model_name="gpt2",
                max_length=30,
                temperature=0.7
            )
            auto_user = AutoUserAgent(auto_user_config)
            auto_user.load_model(best_model_path, use_value_head=False)

            # Initialize chat agent (same as in original app)
            chat_agent = FlightBookingChatAgent(
                config_path=str(self.config_path_chat),
                db_path=str(self.database_path),
                server_path=str(self.server_path)
            )
            await chat_agent.initialize()

            # Initialize verifier agent
            verifier_agent = BookingVerifierAgent(str(self.config_path_verifier))

            # Run demo conversation (following original app pattern)
            self.console.print("\n[highlight]Running demo conversation...[/highlight]")

            booking_objective = "I need to fly from New York to London in May 2025"
            auto_user.reset_conversation(booking_objective)

            self.console.print(f"\n[bold cyan]Booking Objective:[/bold cyan] {booking_objective}")
            self.console.print()

            # Conversation loop (same pattern as original app simulation mode)
            conversation_history = []
            for turn in range(8):  # Max 8 turns
                if turn == 0:
                    user_message = booking_objective
                else:
                    # Get last assistant message for context
                    last_assistant_msg = conversation_history[-1]["content"] if conversation_history else None
                    user_message = await auto_user.generate_response(last_assistant_msg)

                self.console.print(f"[user]User:[/user] {user_message}")
                conversation_history.append({"role": "user", "content": user_message})

                # Get assistant response
                assistant_response = await chat_agent._process_user_message(user_message)
                self.console.print(f"[assistant]Assistant:[/assistant] {assistant_response}")
                conversation_history.append({"role": "assistant", "content": assistant_response})

                # Check for conversation end
                if any(word in user_message.lower() for word in ['quit', 'exit', 'thank', 'done']):
                    break

            # Verification (same as original app)
            self.console.print("\n[highlight]🔍 Running verification...[/highlight]")
            verification_report = await verifier_agent.verify_bookings(
                conversation_history,
                chat_agent.mcp_client
            )

            # Display verification results (same format as original app)
            formatted_report = verifier_agent.format_verification_report(verification_report)
            verification_panel = Panel(
                formatted_report,
                title="[bold]Demo Verification Report[/bold]",
                box=box.DOUBLE,
                style="orange1",
                padding=(1, 2)
            )
            self.console.print(verification_panel)

        else:
            self.console.print("[warning]No trained model found. Run training first.[/warning]")

    async def run(self):
        """Main application runner"""
        try:
            self.display_welcome()
            await self.initialize_training_environment()
            await self.run_ppo_training()

            # Optionally run demo
            if self.console.input("\n[cyan]Run demo conversation? (y/n): [/cyan]").lower() == 'y':
                await self.run_demo_conversation()

        except KeyboardInterrupt:
            self.console.print("\n\n[warning]⚠️ Application terminated by user[/warning]")
        except Exception as e:
            self.console.print(f"\n[error]✗ Critical error: {e}[/error]")
            self.logger.error(f"Critical error: {e}", exc_info=True)

def main():
    """Entry point for PPO training application"""
    parser = argparse.ArgumentParser(
        description="PPO Training System for Auto User Agent"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Run demo conversation only (skip training)"
    )

    args = parser.parse_args()

    app = PPOTrainingApp()

    # Handle demo-only mode
    if args.demo_only:
        asyncio.run(app.run_demo_conversation())
    else:
        asyncio.run(app.run())

if __name__ == "__main__":
    main()