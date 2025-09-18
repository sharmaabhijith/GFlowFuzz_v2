#!/usr/bin/env python3

"""
Test script to verify PPO training setup
"""

import os
import sys
import torch
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")

    try:
        # Test core libraries
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")

        import trl
        print(f"✓ TRL installed")

        import peft
        print(f"✓ PEFT installed")

        # Test custom modules
        from training.algorithms.ppo_trainer import PPOAutoUserTrainer
        print("✓ PPO Trainer module")

        from training.environment.ppo_environment_wrapper import PPOEnvironmentWrapper
        print("✓ Environment wrapper")

        from agents.auto_user.module import AutoUserAgent
        print("✓ Auto user agent")

        from training.environment.booking_environment import BookingConversationEnvironment
        print("✓ Booking environment")

        print("\n✅ All imports successful!")
        return True

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")

    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("✓ CUDA not available, will use CPU")

    return True


def test_huggingface_cache():
    """Test HuggingFace cache setup"""
    print("\nTesting HuggingFace cache...")

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        print(f"✓ HF_HOME set to: {hf_home}")

        if os.path.exists(hf_home):
            print(f"✓ Cache directory exists")
        else:
            os.makedirs(hf_home, exist_ok=True)
            print(f"✓ Created cache directory")
    else:
        print("⚠ HF_HOME not set, using default")

    return True


def test_config_files():
    """Test if configuration files exist"""
    print("\nTesting configuration files...")

    configs = [
        "configs/ppo_training_config.yaml",
        "agents/chat/config.yaml",
        "agents/auto_user/module.py",
        "training/configs/training_config.yaml"
    ]

    all_exist = True
    for config in configs:
        path = project_root / config
        if path.exists():
            print(f"✓ {config}")
        else:
            print(f"✗ {config} not found")
            all_exist = False

    return all_exist


def test_model_loading():
    """Test if we can load a small model"""
    print("\nTesting model loading...")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Try loading a tiny model for testing
        model_name = "microsoft/DialoGPT-small"
        cache_dir = os.path.join(os.environ.get("HF_HOME", "~/hf_cache"), "hub")

        print(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=True
        )
        print("✓ Tokenizer loaded")

        # Don't load full model in test to save time/memory
        print("✓ Model loading test passed (skipped full load)")

        return True

    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("PPO Training Setup Test")
    print("=" * 50)

    tests = [
        test_imports,
        test_cuda,
        test_huggingface_cache,
        test_config_files,
        test_model_loading
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test failed with error: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    if all(results):
        print("✅ All tests passed! Setup is ready for training.")
        print("\nYou can now run:")
        print("  python training/train_ppo.py --config configs/ppo_training_config.yaml")
        print("\nOr use the convenience script:")
        print("  ./training/run_training.sh")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        failed_count = len([r for r in results if not r])
        print(f"\nFailed tests: {failed_count}/{len(results)}")
    print("=" * 50)


if __name__ == "__main__":
    main()