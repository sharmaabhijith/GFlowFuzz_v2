import argparse
import copy
import logging
import json
import os
import sys
from pathlib import Path
from rich.panel import Panel
from dotenv import load_dotenv
from rich.console import Console
from dataclasses import asdict
from typing import Dict, Any, List, Optional
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)

from training.setup import (
    TrainingSetup,
    Trajectory,
    EpisodeMetrics,
    save_metrics,
    load_config,
)
from training.utils import BookingObjectiveGenerator, ConsoleReporter
from training.oracle import Oracle


project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def run_training(config: Dict[str, Any]) -> None:
    console = Console(record=True)

    algorithm_name = config.get("algorithm", "grpo").lower()

    setup = TrainingSetup(config, console)
    setup.configure_hf_caches()
    reporter = ConsoleReporter(console)
    reporter.show_config(algorithm_name, config)
    algo_config = config[algorithm_name]
    output_dir = setup.prepare_output()
    reporter.configure_logging(output_dir)
    logging.info("Training output directory: %s", output_dir)

    console.print(Panel("[bold cyan]Initializing Training Components[/bold cyan]", expand=False))
    environment = setup.environment()
    policy = setup.policy()
    trainer = setup.trainer()
    policy_cfg = config.get("environment", {}).get("judge", {})
    policy_set_path = policy_cfg.get("policy_set_path")
    policy_path = Path(policy_set_path)
    with open(policy_path, "r", encoding="utf-8") as policy_file:
        policy_bundle = yaml.safe_load(policy_file) or {}
        logging.info("Loaded policy bundle from %s", policy_path)

    # process_reward_cfg = copy.deepcopy(
    #     config.get("environment", {}).get("process_reward", {})
    # )
    # armo_cfg = process_reward_cfg.get("armo") if process_reward_cfg else None
    # if armo_cfg and armo_cfg.get("model_path"):
    #     model_path = Path(armo_cfg["model_path"])
    #     if not model_path.is_absolute():
    #         armo_cfg["model_path"] = str(project_root / model_path)

    reward = Oracle(
        booking_agent=environment.booking_agent,
        verifier_agent=environment.verifier_agent,
        coder_agent=environment.coder_agent,
        judge_agent=getattr(environment, "judge_agent", None),
        policy_bundle=policy_bundle,
        process_reward_config=process_reward_cfg,
    )

    objective_generator_cfg = config.get("objective_generator", {})
    env_cfg = config.get("environment", {})
    db_path = env_cfg.get("booking_agent_config", {}).get("db_path")
    objective_generator = BookingObjectiveGenerator(
        db_path=db_path,
        config=objective_generator_cfg,
        logger=logging.getLogger(__name__),
    )

    gamma = algo_config.get("gamma", 0.99)
    save_every = algo_config.get("save_freq", 0)
    total_episodes = algo_config.get("episodes")

    history: List[EpisodeMetrics] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[bold cyan]Collecting episodes...[/bold cyan]", total=total_episodes)

        for episode_idx in range(1, total_episodes + 1):
            progress.update(task, description=f"[bold cyan]Episode {episode_idx}/{total_episodes}[/bold cyan]")

            console.print(
                Panel(
                    f"[bold]Episode {episode_idx}[/bold]\nGenerating trajectory...",
                    title="[cyan]Training Status[/cyan]",
                    expand=False,
                )
            )
            episode_objective = (
                objective_generator.generate() if objective_generator else None
            )
            if episode_objective:
                console.print(
                    Panel(
                        f"[bold cyan]Objective[/bold cyan]\n{episode_objective}",
                        title="[yellow]Episode Goal[/yellow]",
                        expand=False,
                    )
                )
            state_obj = environment.reset(booking_objective=episode_objective)
            if episode_objective and getattr(state_obj, "booking_objective", None) != episode_objective:
                state_obj.booking_objective = episode_objective
            state = environment._format_state(state_obj)
            states: List[str] = []
            actions: List[str] = []
            conversation_history: List[Dict[str, Any]] = []
            process_rewards: List[float] = []
            done = False
            while not done:
                states.append(state)
                action = policy.get_action(state)
                actions.append(action)
                conversation_history.append(
                    {
                        "turn": len(states) - 1,
                        "state_preview": state[:200] + "..." if len(state) > 200 else state,
                        "action": action,
                    }
                )
                step_result = environment.step(action)
                process_reward_value = reward.compute_process(
                    previous_state,
                    action,
                    step_result.state,
                    history_before=history_before_action,
                    history_after=history_after_action,
                )
                process_rewards.append(float(process_reward_value))

                assistant_reply = environment.booking_agent.conversation_history[-1].get("content")
                reporter.show_turn(episode_idx, len(conversation_history), action, assistant_reply)

                done = step_result.done
                state_obj = step_result.state
                if not done:
                    state = environment._format_state(step_result.state)
            num_turns = len(states)
            terminal_reward = reward.compute_terminal(
                state=environment.current_state,
                final_booking_summary=environment.final_booking_summary,
            )
            shaped_rewards = reward.compute_shaped(
                terminal_reward,
                num_turns,
                gamma=gamma,
            )
            combined_rewards: List[float] = []
            for idx in range(num_turns):
                shaped_val = shaped_rewards[idx] if idx < len(shaped_rewards) else 0.0
                process_val = process_rewards[idx] if idx < len(process_rewards) else 0.0
                combined_rewards.append(float(shaped_val + process_val))
            policy_result = reward.latest_policy_result
            reward_components = dict(reward.latest_reward_components or {})

            trajectory = Trajectory(
                episode_index=episode_idx,
                states=states,
                actions=actions,
                rewards=combined_rewards,
                terminal_reward=terminal_reward,
                episode_reward=terminal_reward,
                num_turns=num_turns,
                conversation_history=conversation_history,
                process_rewards=process_rewards,
                booking_summary=getattr(environment, "final_booking_summary", None),
                objective=episode_objective or getattr(state_obj, "booking_objective", None),
                reward_components=reward_components,
            )
            trajectory_path = output_dir / "trajectory_logs" / f"episode_{episode_idx:04d}.json"
            with open(trajectory_path, "w", encoding="utf-8") as trajectory_file:
                json.dump(trajectory.to_dict(), trajectory_file, indent=2)

            train_metrics = trainer.train_step(
                trajectories=[trajectory.to_dict()],
                epoch=episode_idx - 1,
            )
            metrics = EpisodeMetrics.from_trajectory(episode_idx, trajectory, train_metrics)
            history.append(metrics)
            reporter.show_episode(episode_idx, metrics)
            if save_every and (episode_idx % save_every == 0):
                trainer.save_checkpoint(episode_idx, str(output_dir))
            progress.advance(task, 1)

    console.print(Panel("[bold green]Training Complete![/bold green]", title="Status", expand=False))

    trainer.save_final_model(str(output_dir))
    save_metrics([rec.to_dict() for rec in history], str(output_dir))

    reporter.show_final(history, output_dir)
    console.print("[cyan]Review the logs to understand conversation patterns and success cases.[/cyan]")
    console.save_text(output_dir / "console.txt")
    console.export_html(output_dir / "console.html")
    logging.info("Saved console transcript to %s and %s", output_dir / "console.txt", output_dir / "console.html")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Fine Tuning for Conversation Policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/training_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    config = load_config(args.config)

    run_training(config)

if __name__ == "__main__":
    main()
