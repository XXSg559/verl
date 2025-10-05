# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Async Batch Processing for Agent Tree GRPO Training

This module provides comprehensive async batch processing capabilities for both
agent-level and step-level batch generation in Tree GRPO training scenarios.

Key Components:
- AsyncBatchVLLMModel: Async vLLM batch generation
- StepSyncBarrier: Agent synchronization for step-level batching
- AsyncCodeAgent: Async wrapper for smolagents CodeAgent
- StepBatchCoordinator: Step-level batch orchestration
- AsyncTreeOrchestrator: Unified agent + step level batch modes
"""

# Core async batch components
from .async_batch_vllm_model import AsyncBatchVLLMModel
from .step_sync_barrier import (
    StepSyncBarrier,
    SyncBarrierConfig,
    SyncBarrierManager,
    SyncState,
    AgentSyncStatus
)
from .async_agent import (
    AsyncAgentInterface,
    AsyncAgentPool,
    AsyncExecutionConfig,
    ExecutionMode,
    StepExecutionResult
)
from .step_batch_coordinator import (
    StepBatchCoordinator,
    BatchCoordinationConfig,
    CoordinationStrategy,
    BatchExecutionMetrics,
    CoordinationSession
)
from .async_tree_orchestrator import (
    AsyncTreeOrchestrator,
    AsyncTreeConfig,
    BatchMode,
    OrchestrationStrategy,
    TreeExecutionResult,
    TreeSession
)

# Import high-level API
from .easy_api import (
    coordinate_async_agents,
    run_demo
)

# Version and metadata
__version__ = "1.0.0"
__author__ = "Bytedance VERL Team"

# Export all public components
__all__ = [
    # Core Models
    "AsyncBatchVLLMModel",

    # Synchronization
    "StepSyncBarrier",
    "SyncBarrierConfig",
    "SyncBarrierManager",
    "SyncState",
    "AgentSyncStatus",

    # Async Agents
    "AsyncAgentInterface",
    "AsyncAgentPool",
    "AsyncExecutionConfig",
    "ExecutionMode",
    "StepExecutionResult",

    # Coordination
    "StepBatchCoordinator",
    "BatchCoordinationConfig",
    "CoordinationStrategy",
    "BatchExecutionMetrics",
    "CoordinationSession",

    # Orchestration
    "AsyncTreeOrchestrator",
    "AsyncTreeConfig",
    "BatchMode",
    "OrchestrationStrategy",
    "TreeExecutionResult",
    "TreeSession",

    # High-level API
    "coordinate_async_agents",
    "run_demo",
]

# Convenience imports for common use cases
def create_async_orchestrator(
    model_id: str,
    batch_mode: str = "hybrid",
    max_agents: int = 5,
    **kwargs
):
    """
    Convenience function to create an async agent orchestrator.

    Args:
        model_id: vLLM model identifier
        batch_mode: "agent_level", "step_level", or "hybrid"
        max_agents: Maximum number of agents for coordination
        **kwargs: Additional configuration options

    Returns:
        AsyncTreeOrchestrator configured for multi-agent coordination
    """
    # Create async model
    model = AsyncBatchVLLMModel(
        model_id=model_id,
        batch_size=kwargs.get('batch_size', 4),
        max_concurrent_batches=kwargs.get('max_concurrent_batches', 2),
        **kwargs.get('model_kwargs', {})
    )

    # Map batch mode strings to enums
    mode_mapping = {
        "agent_level": BatchMode.AGENT_LEVEL,
        "step_level": BatchMode.STEP_LEVEL,
        "hybrid": BatchMode.HYBRID,
        "adaptive": BatchMode.ADAPTIVE
    }

    # Create configuration
    config = AsyncTreeConfig(
        batch_mode=mode_mapping.get(batch_mode, BatchMode.HYBRID),
        orchestration_strategy=OrchestrationStrategy.HIERARCHICAL,
        agent_level_proposals=kwargs.get('agent_level_proposals', 3),
        max_agents_per_step_batch=max_agents,
        step_sync_timeout=kwargs.get('step_sync_timeout', 30.0),
        max_tree_depth=kwargs.get('max_tree_depth', 5)
    )

    # Create and return orchestrator
    return AsyncTreeOrchestrator(model, config)


# Helper function for quick testing
def create_mock_async_solver(batch_mode: str = "hybrid", max_agents: int = 3):
    """
    Create a mock async solver for testing without requiring real models.

    Args:
        batch_mode: "agent_level", "step_level", or "hybrid"
        max_agents: Maximum number of agents

    Returns:
        AsyncTreeOrchestrator with mock model
    """
    from .async_batch_vllm_model import AsyncBatchVLLMModel

    # Create mock model that doesn't require actual vLLM
    class MockAsyncModel(AsyncBatchVLLMModel):
        def __init__(self):
            self.model_id = "mock://test-model"
            self.batch_size = 3
            self.generation_count = 0
            # Skip parent __init__ to avoid vLLM requirements

        async def generate_async(self, messages, **kwargs):
            import asyncio
            await asyncio.sleep(0.1)

            class MockResponse:
                def __init__(self, content):
                    self.content = content

            self.generation_count += 1
            return MockResponse(f"Mock solution step {self.generation_count}")

        async def generate_batch_async(self, batch_messages, **kwargs):
            import asyncio
            await asyncio.sleep(0.2)

            class MockResponse:
                def __init__(self, content):
                    self.content = content

            responses = []
            for i, _ in enumerate(batch_messages):
                self.generation_count += 1
                responses.append(MockResponse(f"Mock batch response {self.generation_count}"))
            return responses

    # Create mock orchestrator
    return create_async_orchestrator(
        model_id="mock://test",
        batch_mode=batch_mode,
        max_agents=max_agents
    )


# Documentation and usage information
USAGE_EXAMPLES = """
# Quick Start Examples

## 1. Create async orchestrator with real vLLM model
from verl.experimental.agent_tree.async_batch import create_async_orchestrator

orchestrator = create_async_orchestrator(
    model_id="deepseek/deepseek-chat",
    batch_mode="hybrid",
    max_agents=5
)

## 2. Create mock orchestrator for testing
from verl.experimental.agent_tree.async_batch import create_mock_async_solver

mock_orchestrator = create_mock_async_solver(batch_mode="step_level")

## 3. Manual configuration
from verl.experimental.agent_tree.async_batch import *

model = AsyncBatchVLLMModel(model_id="your-model")
config = AsyncTreeConfig(batch_mode=BatchMode.HYBRID)
orchestrator = AsyncTreeOrchestrator(model, config)

## 4. Using the solver
import asyncio

async def solve_problem():
    session = await solver.create_tree_session(
        "math_session",
        "Solve: 2x² - 7x + 3 = 0"
    )

    # Agent-level batch (multiple solution approaches)
    agent_result = await solver.execute_agent_level_batch(
        "math_session",
        num_branches=3
    )

    # Step-level batch (coordinated agents)
    step_configs = [
        {"agent_id": "analyzer"},
        {"agent_id": "solver"},
        {"agent_id": "verifier"}
    ]
    step_result = await solver.execute_step_level_batch(
        "math_session",
        step_configs,
        max_steps=3
    )

asyncio.run(solve_problem())
"""

def print_usage():
    """Print usage examples for the async batch module."""
    print("=" * 80)
    print("Async Batch Processing for Agent Tree GRPO")
    print("=" * 80)
    print(USAGE_EXAMPLES)
    print("=" * 80)