# 🚀 PPO Training Framework for Conversational AI Agents

> A state-of-the-art Reinforcement Learning framework implementing Proximal Policy Optimization (PPO) for training conversational agents using **proper RL architecture with stateless policies**.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Architecture Principles](#-key-architecture-principles)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Components](#-components)
- [Configuration](#-configuration)
- [Training Pipeline](#-training-pipeline)
- [Monitoring](#-monitoring)
- [Advanced Usage](#-advanced-usage)
- [Migration from Stateful to Stateless](#-migration-from-stateful-to-stateless)
- [Troubleshooting](#-troubleshooting)
- [API Reference](#-api-reference)

## 🎯 Overview

This framework implements a **proper RL architecture** for training conversational agents using Proximal Policy Optimization (PPO). The system follows standard reinforcement learning principles with a **stateless policy** that acts as a pure function mapping states to actions.

### What's New in This Version

✨ **Stateless Architecture**: The policy is now a pure function `π(s) → a` with no internal state
🎯 **Proper RL Design**: Environment manages all state, policy just maps states to actions
🔄 **Clean Separation**: Clear boundaries between environment and policy responsibilities
⚡ **Better Performance**: Stateless design enables parallelization and easier debugging

### Key Features

- 🧠 **Stateless Policy**: Pure function architecture following RL best practices
- 🎮 **Environment-Managed State**: All conversation state handled by the environment
- 🔄 **Experience Replay Buffer**: Efficient trajectory storage and sampling
- 🎯 **Sparse Reward System**: Episode-level rewards with verification
- 📊 **Real-time Monitoring**: Comprehensive metrics and Tensorboard integration
- ⚡ **Memory Efficient**: Support for quantization and LoRA fine-tuning

## 🏛️ Key Architecture Principles

### ❌ What NOT to Do (Old Approach)

```python
# WRONG: Policy maintains internal state
class StatefulAgent:
    def __init__(self):
        self.conversation_history = []  # ❌ Policy stores state
        self.objective = None           # ❌ Policy tracks goal

    def reset_conversation(self, objective):  # ❌ Policy has reset
        self.conversation_history = []
        self.objective = objective

    def get_action(self, partial_state):
        # Uses internal state + partial state
        return self.generate_with_memory(partial_state)
```

### ✅ Correct Approach (Current Implementation)

```python
# CORRECT: Policy is stateless
class AutoUserAgent:
    def __init__(self, config):
        self.model = None       # Only model, no state
        self.tokenizer = None   # Only tokenizer, no memory

    async def get_action(self, complete_state: str) -> str:
        # Pure function: complete_state → action
        # No internal memory needed
        return await self.generate(complete_state)

# Environment manages ALL state
environment.reset()  # Only environment resets
state = environment._format_state(state_obj)  # Complete state with context
action = policy.get_action(state)  # Pure function call
```

## 🏗️ System Architecture

```
+---------------------------------------------------------------+
|                    PPO Training Framework                     |
|                    (Stateless Architecture)                  |
+---------------------------------------------------------------+
|                                                               |
|  +========================+  +======================+        |
|  | AutoModelForCausalLM   |  | Environment         |        |
|  | WithValueHead          |  | (State Manager)     |        |
|  +========================+  +======================+        |
|  | +------------------+   |  | - Objective         |        |
|  | | Policy Head      |   |  | - Conversation      |        |
|  | | π(s) → a         |   |  |   History           |        |
|  | | (stateless)      |   |  | - Booking Context   |        |
|  | +------------------+   |  | - Turn Count        |        |
|  |          |             |  +----------+-----------+        |
|  | +------------------+   |             |                    |
|  | | Value Head       |   |             v                    |
|  | | V(s) → R         |   |    Complete State String         |
|  | +------------------+   |    "Objective: Book flight...    |
|  +----------+-------------+     History: User: I need...     |
|             |                   Turn: 3"                     |
|             v                            |                    |
|  +========================================v=========+         |
|  |              PPO Trainer (TRL Library)          |         |
|  | - Receives complete states from environment     |         |
|  | - Gets actions from stateless policy            |         |
|  | - Updates both policy and value heads           |         |
|  +==================================================+         |
|                                                               |
|  +--------------------------------------------------+         |
|  | Reference Model (Frozen)                        |         |
|  | - KL divergence constraint                      |         |
|  | - Prevents policy drift                         |         |
|  +--------------------------------------------------+         |
+---------------------------------------------------------------+
```

### Architecture Flow

```
1. Environment Reset
   └─> Generates objective
   └─> Initializes empty conversation
   └─> Returns initial state

2. State Formatting (Environment)
   └─> Includes objective
   └─> Includes full conversation history
   └─> Includes booking context
   └─> Creates complete Markovian state

3. Policy Action (Stateless)
   └─> Receives complete state string
   └─> No access to internal memory
   └─> Returns action (user message)

4. Environment Step
   └─> Processes action
   └─> Updates conversation history
   └─> Returns next complete state

5. PPO Training
   └─> Uses complete states as queries
   └─> Uses actions as responses
   └─> Updates policy and value heads
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/gflowfuzz_v2.git
cd gflowfuzz_v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your configuration
```

## 🚀 Quick Start

### Basic Training with Stateless Architecture

```bash
# Run training with stateless policy
python training/train.py --config configs/ppo_training_config.yaml
```

The training script will:
1. Initialize environment (state manager)
2. Create stateless policy agent
3. Collect trajectories without policy state
4. Train using PPO with proper RL architecture

### Resume Training

```bash
python training/train.py \
    --config configs/ppo_training_config.yaml \
    --resume ./ppo_training_output/checkpoint-5
```

### Debug Mode

```bash
python training/train.py \
    --config configs/ppo_training_config.yaml \
    --debug
```

## 📁 Components

### 1. **train.py** - Main Training Orchestrator

The central training script implementing proper RL architecture with stateless policy.

#### Key Features:
- ✅ No `policy.reset_conversation()` calls
- ✅ Environment manages all state
- ✅ Policy is pure function: `state → action`
- ✅ Clean separation of concerns

#### Core Functions:

```python
async def collect_trajectories(
    policy: AutoUserAgent,
    environment: BookingConversationEnvironment,
    num_episodes: int
) -> List[Dict]:
    """
    Collect trajectories with STATELESS policy

    Key: No policy reset needed - only environment resets!
    """
    trajectories = []

    for episode in range(num_episodes):
        # ONLY environment resets (no policy.reset_conversation!)
        state_obj = environment.reset()
        state = environment._format_state(state_obj)  # Complete state

        trajectory = {"states": [], "actions": [], ...}
        done = False

        while not done:
            # Policy is pure function: complete_state → action
            action = await policy.get_action(state)  # No internal state!

            # Environment manages state transition
            step_result = await environment.step(action)
            state = environment._format_state(step_result.state)
            done = step_result.done

        # ... rest of trajectory collection
```

### 2. **agents/auto_user/module.py** - Stateless Policy Agent

The AutoUserAgent is now a **pure function** with no conversation state.

```python
class AutoUserAgent:
    """
    Stateless Policy - Pure Function π(s) → a

    NO conversation history
    NO objective tracking
    NO reset method
    ONLY model and tokenizer
    """

    def __init__(self, config: AutoUserConfig):
        self.config = config
        self.model = None      # Just the model
        self.tokenizer = None  # Just the tokenizer
        # NO conversation_history or objective!

    async def get_action(self, state: str) -> str:
        """
        Pure function mapping state to action

        Args:
            state: COMPLETE state from environment including:
                   - Objective
                   - Full conversation history
                   - Current context

        Returns:
            action: Generated user message
        """
        # Tokenize complete state
        inputs = self.tokenizer.encode(
            state,  # Complete context from environment
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        # Generate action
        outputs = self.model.generate(inputs, ...)
        return self.tokenizer.decode(outputs)
```

### 3. **environment.py** - Conversation Environment (State Manager)

The environment now manages **ALL** conversation state.

```python
class BookingConversationEnvironment:
    """
    Environment manages ALL state - proper RL design
    """

    def reset(self, objective: Optional[str] = None) -> ConversationState:
        """
        Environment reset - manages ALL state
        """
        self.current_objective = objective or self._generate_objective()
        self.conversation_history = []  # Environment tracks this!
        self.booking_context = {}

        return ConversationState(
            conversation_history=[],
            booking_context={},
            turn_count=0,
            booking_objective=self.current_objective
        )

    def _format_state(self, conversation_state: ConversationState) -> str:
        """
        Creates COMPLETE Markovian state for policy
        """
        parts = []

        # Include objective (goal)
        parts.append(f"Objective: {conversation_state.booking_objective}")

        # Include full conversation history
        parts.append("Conversation History:")
        for msg in conversation_state.conversation_history[-20:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            parts.append(f"{role}: {msg['content']}")

        # Include booking context
        if conversation_state.booking_context:
            parts.append(f"Current Status: {conversation_state.booking_context}")

        # Include metadata
        parts.append(f"Turn: {conversation_state.turn_count + 1}")

        # Prompt for next action
        parts.append("User:")

        return "\n".join(parts)
```

### 4. **utils.py** - Training Utilities

Supporting utilities remain largely unchanged but work with stateless architecture.

## ⚙️ Configuration

### Configuration Structure (`ppo_training_config.yaml`)

```yaml
# Environment Configuration
environment:
  max_conversation_length: 10       # Max turns per episode
  booking_agent_config:
    config_path: "agents/chat/config.yaml"
    db_path: "database/flights.db"
    server_path: "mcp-server/database_server.py"
  verifier_config:
    config_path: "agents/verifier/config.yaml"

# Model Configuration
model:
  base_model: "microsoft/DialoGPT-small"  # Pre-trained LM
  max_response_length: 30                 # Token limit
  device: "auto"                          # GPU/CPU selection

  # Memory Optimization
  use_quantization: true                  # 4-bit quantization
  use_lora: true                          # LoRA fine-tuning

# PPO Hyperparameters
ppo:
  learning_rate: 1.0e-5                   # Adam LR
  batch_size: 8                           # Training batch
  max_epochs: 10                          # Total epochs
  rollout_episodes: 5                     # Episodes/epoch

  # PPO Specific
  ppo_epochs: 4                           # PPO iterations
  gamma: 0.99                             # Discount factor
  lam: 0.95                               # GAE lambda
  cliprange: 0.2                          # Policy clip
  vf_coef: 0.5                            # Value loss weight
```

## 📊 Training Pipeline

### Phase 1: Initialization
1. Load configuration
2. Initialize environment (state manager)
3. Setup stateless policy agent
4. Create PPO trainer with value head
5. Initialize experience buffer

### Phase 2: Trajectory Collection (Stateless)
```python
for episode in episodes:
    # Environment manages reset
    state = environment.reset()

    while not done:
        # Policy: pure function call
        action = policy.get_action(complete_state)

        # Environment: state transition
        next_state = environment.step(action)
```

### Phase 3: PPO Training
1. Prepare batches from trajectories
2. Compute advantages using value head
3. Update policy with clipped objective
4. Update value function
5. Apply KL constraint

### Phase 4: Evaluation & Checkpointing
1. Calculate metrics
2. Save checkpoints
3. Log to Tensorboard

## 📈 Monitoring

### Key Metrics

| Metric | Description | Target Range |
|--------|-------------|--------------|
| `episode_reward` | Average reward per episode | 0.7 - 1.0 |
| `ppo/loss/total` | Combined PPO loss | < 0.5 |
| `architecture` | Training architecture | "stateless" |
| `state_completeness` | Markovian state check | 1.0 |

### Tensorboard

```bash
tensorboard --logdir ./tensorboard_logs --port 6006
```

## 🔄 Migration from Stateful to Stateless

### If you're upgrading from the old stateful version:

#### Old Code (Stateful - Wrong)
```python
# ❌ OLD: Policy maintains state
policy.reset_conversation(objective)
policy.update_conversation_history(role, message)
action = await policy.generate_response(assistant_message)
```

#### New Code (Stateless - Correct)
```python
# ✅ NEW: Environment manages state
state = environment.reset()
complete_state = environment._format_state(state)
action = await policy.get_action(complete_state)
```

### Migration Checklist

- [ ] Remove `reset_conversation()` calls from training loop
- [ ] Remove `update_conversation_history()` from policy
- [ ] Remove `conversation_history` attribute from policy
- [ ] Remove `booking_objective` attribute from policy
- [ ] Update environment's `_format_state()` to provide complete context
- [ ] Ensure policy's `get_action()` uses only provided state
- [ ] Test that policy has no internal state dependencies

## 🔧 Advanced Usage

### Parallel Training with Stateless Policy

Since the policy is stateless, you can easily parallelize:

```python
import asyncio

async def parallel_collection(policies, environments, episodes_per_env):
    """Collect trajectories in parallel with stateless policies"""
    tasks = []
    for policy, env in zip(policies, environments):
        task = collect_trajectories(policy, env, episodes_per_env)
        tasks.append(task)

    # All policies can run in parallel - no state conflicts!
    all_trajectories = await asyncio.gather(*tasks)
    return [traj for sublist in all_trajectories for traj in sublist]
```

### Custom State Formatting

Extend the environment's state formatting for your needs:

```python
def _format_state(self, conversation_state: ConversationState) -> str:
    """Custom state formatting with additional context"""

    # Base state
    state = super()._format_state(conversation_state)

    # Add custom features
    state += f"\nTime: {conversation_state.timestamp}"
    state += f"\nUser Profile: {self.user_profile}"
    state += f"\nSystem Status: {self.system_status}"

    return state
```

## 🐛 Troubleshooting

### Common Issues with Stateless Architecture

#### Issue: "Policy doesn't remember context"
**Solution**: Check that environment's `_format_state()` includes full conversation history

#### Issue: "Training is unstable"
**Solution**: Ensure state is truly Markovian (contains all needed information)

#### Issue: "Old code breaks with new architecture"
**Solution**: Remove all state management from policy, move to environment

## 📚 API Reference

### Core Classes

#### `AutoUserAgent` (Stateless Policy)
```python
class AutoUserAgent:
    """Stateless policy agent - pure function"""

    async def get_action(state: str) -> str:
        """Maps complete state to action"""
```

#### `BookingConversationEnvironment` (State Manager)
```python
class BookingConversationEnvironment:
    """Environment managing all conversation state"""

    def reset(objective: Optional[str] = None) -> ConversationState
    async def step(action: str) -> EnvironmentStep
    def _format_state(state: ConversationState) -> str
```

#### `PPOTrainer`
```python
class PPOTrainer:
    """PPO trainer working with stateless policy"""

    def step(queries: List[Tensor],
             responses: List[Tensor],
             rewards: Tensor) -> Dict
```

## 🎓 Why Stateless Architecture?

### Benefits

1. **True Markov Property**: State contains all information needed
2. **Parallelizable**: No state conflicts between instances
3. **Debuggable**: Pure functions are predictable
4. **Standard RL**: Compatible with any RL library
5. **Clean Code**: Clear separation of concerns

### Comparison

| Aspect | Stateful (Old) | Stateless (New) |
|--------|---------------|-----------------|
| Policy Memory | Maintains history | No memory |
| Reset Location | Policy + Environment | Environment only |
| State Completeness | Partial | Complete (Markovian) |
| Parallelization | Difficult | Easy |
| Debugging | Complex | Simple |
| RL Compatibility | Limited | Full |

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions welcome! Please ensure:
- Policy remains stateless
- Environment manages all state
- State is Markovian (complete)
- Clear separation of concerns

## 📬 Contact

- GitHub Issues: [Create an issue](https://github.com/yourusername/gflowfuzz_v2/issues)
- Email: support@gflowfuzz.ai

## 🙏 Acknowledgments

- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) - Schulman et al.
- [TRL Library](https://github.com/huggingface/trl) - HuggingFace
- Reinforcement Learning Community for best practices

---

**Built with ❤️ following proper RL principles by the GFlowFuzz Team**