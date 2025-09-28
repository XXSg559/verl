# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

verl (Volcano Engine Reinforcement Learning) is a flexible, efficient and production-ready RL training library for large language models (LLMs). It's the open-source version of the HybridFlow paper and enables flexible representation and efficient execution of complex post-training dataflows.

## Development Commands

### Installation and Setup
```bash
# Basic development installation
pip install -e .[test,vllm]
# or with SGLang
pip install -e .[test,sglang]

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Linting and Formatting
```bash
# Run pre-commit on staged changes
pre-commit run

# Run pre-commit on all files
pre-commit run --all-files

# Run specific hooks
pre-commit run --all-files ruff
pre-commit run --all-files autogen-trainer-cfg
```

### Testing
```bash
# Run specific test categories (check .github/workflows/ for available test suites)
# - GPU unit tests
# - CPU unit tests
# - vLLM tests
# - SGLang tests

# Run single test file
python -m pytest tests/path/to/test_file.py

# Run tests with specific markers (check test files for available markers)
python -m pytest -m "marker_name"
```

### Documentation
```bash
# Install documentation dependencies
pip install -r requirements-docs.txt

# Build documentation
make clean
make html

# Preview locally
python -m http.server -d _build/html/
```

## Architecture Overview

### Core Components

1. **Protocol System** (`verl/protocol.py`): Core data transfer protocol using TensorDict for communication between modules. The `DataProto` class handles batch data transfer across the distributed system.

2. **Hybrid Controller Programming Model**: Enables flexible representation of RL dataflows by decoupling computation and data dependencies. Supports various placement of models onto different GPU sets.

3. **Workers** (`verl/workers/`):
   - **FSDP Workers** (`fsdp_workers.py`): PyTorch FSDP backend for training
   - **Megatron Workers** (`megatron_workers.py`): Megatron-LM backend for training
   - **Rollout Workers** (`rollout/`): Handle generation using vLLM, SGLang, or HF Transformers
   - **Reward Model Workers** (`reward_model/`): Process reward computations

4. **Trainers** (`verl/trainer/`):
   - **PPO Trainer** (`ppo/`): Proximal Policy Optimization implementation
   - **GRPO, GSPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO** and other RL algorithms
   - **SFT Trainer** (`sft_trainer.py`): Supervised fine-tuning

5. **Configuration System** (`verl/trainer/config/`): YAML-based configuration with hydra integration. Auto-generated configs in `_generated_*.yaml` files.

### Key Design Patterns

- **3D-HybridEngine**: Eliminates memory redundancy and reduces communication overhead during transitions between training and generation phases
- **Ray-based Distribution**: Uses Ray for distributed coordination and resource management
- **Modular Backend Support**: Seamless integration with FSDP, FSDP2, Megatron-LM, vLLM, SGLang
- **Device Mapping Flexibility**: Supports various placement strategies for efficient resource utilization

### Training Backends
- **FSDP/FSDP2**: Recommended PyTorch distributed training (set `strategy=fsdp2`)
- **Megatron-LM**: For large-scale models with expert parallelism (supports up to 671B models)

### Inference Backends
- **vLLM**: High-performance inference (>=0.8.2 supported, avoid 0.7.x)
- **SGLang**: Full support with multi-turn agentic RL and VLM RLHF features
- **HF Transformers**: Basic inference support

## Important Notes

### Model Support
- Compatible with HuggingFace and Modelscope models: Qwen-3, Qwen-2.5, Llama3.1, Gemma2, DeepSeek-LLM
- Vision-language models (VLMs) supported: Qwen2.5-vl, Kimi-VL
- Multi-modal RL and multi-turn tool calling support

### Performance Features
- Flash Attention 2, sequence packing, sequence parallelism
- LoRA training support for memory efficiency
- Expert parallelism for MoE models
- FSDP2 CPU offloading with gradient accumulation

### Environment Variables
- `VERL_USE_MODELSCOPE=true`: Use Modelscope hub instead of HuggingFace

### NPU Support
- ASCEND NPU support available with transformers>=4.52.4
- Includes special patches for tensordict synchronization