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
AsyncAgent provides async execution interface for step-level batch processing.

This module defines the async agent interface and base implementation that enables
both agent-level and step-level batch generation through async/await patterns
and synchronization barriers.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum

from .async_batch_vllm_model import AsyncBatchVLLMModel
from .step_sync_barrier import StepSyncBarrier, SyncBarrierConfig, AgentSyncStatus

logger = logging.getLogger(__name__)

try:
    from smolagents import CodeAgent
    from smolagents.models import ChatMessage, MessageRole
    from smolagents.memory import AgentMemory, ActionStep, TaskStep, PlanningStep
    SMOLAGENTS_AVAILABLE = True
    Memory = AgentMemory  # For compatibility
except ImportError:
    SMOLAGENTS_AVAILABLE = False
    CodeAgent = None
    ChatMessage = None
    MessageRole = None
    Memory = None
    AgentMemory = None
    ActionStep = None
    TaskStep = None
    PlanningStep = None


class ExecutionMode(Enum):
    """Agent execution modes."""
    SEQUENTIAL = "sequential"
    AGENT_BATCH = "agent_batch"
    STEP_BATCH = "step_batch"
    MIXED_BATCH = "mixed_batch"


@dataclass
class AsyncExecutionConfig:
    """Configuration for async agent execution."""
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    enable_step_sync: bool = True
    step_sync_timeout: float = 30.0
    max_concurrent_agents: int = 5
    enable_memory_persistence: bool = True
    retry_failed_steps: bool = True
    max_step_retries: int = 3


@dataclass
class StepExecutionResult:
    """Result of a single step execution."""
    step_number: int
    agent_id: str
    execution_id: str
    success: bool
    response: Optional[Any] = None
    memory_state: Optional[Any] = None
    batch_info: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None


class AsyncAgentInterface(ABC):
    """
    Abstract interface for async agent execution.

    This interface defines the contract for agents that can participate in
    step-level batch processing and agent-level batch generation.
    """

    @abstractmethod
    async def execute_step_async(
        self,
        step_input: Any,
        step_number: int,
        sync_barrier: Optional[StepSyncBarrier] = None,
        **kwargs
    ) -> StepExecutionResult:
        """Execute a single step asynchronously."""
        pass

    @abstractmethod
    async def execute_sequence_async(
        self,
        initial_input: Any,
        max_steps: int = 10,
        sync_barrier: Optional[StepSyncBarrier] = None,
        **kwargs
    ) -> List[StepExecutionResult]:
        """Execute a complete sequence of steps asynchronously."""
        pass

    @abstractmethod
    def get_current_memory_state(self) -> Optional[Any]:
        """Get the current memory state for synchronization."""
        pass

    @abstractmethod
    async def set_memory_state(self, memory_state: Any) -> bool:
        """Set the memory state from synchronization."""
        pass


class AsyncCodeAgent(AsyncAgentInterface):
    """
    Async wrapper for smolagents CodeAgent enabling step-level batch processing.

    This class wraps a standard CodeAgent and provides async execution capabilities
    with support for step-level synchronization and batch generation.
    """

    def __init__(
        self,
        # === CodeAgent 的原生参数，完全一致 ===
        tools: list,
        model,
        prompt_templates: Optional[Any] = None,
        additional_authorized_imports: Optional[list] = None,
        planning_interval: Optional[int] = None,
        executor_type: str = "local",
        executor_kwargs: Optional[dict] = None,
        max_print_outputs_length: Optional[int] = None,
        stream_outputs: bool = False,
        use_structured_outputs_internally: bool = False,
        code_block_tags: Optional[Union[str, tuple]] = None,

        # === 异步新增参数 ===
        agent_id: Optional[str] = None,
        async_config: Optional[AsyncExecutionConfig] = None,
        **kwargs
    ):
        if not SMOLAGENTS_AVAILABLE:
            raise ImportError("smolagents is required for AsyncCodeAgent")

        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        self.async_config = async_config or AsyncExecutionConfig()

        # Handle async model - extract base model for CodeAgent compatibility
        if hasattr(model, 'base_model'):
            # This is an AsyncBatchVLLMModel, use its sync interface
            code_agent_model = model.base_model
            self.async_model = model
        else:
            # This is a regular Model, use directly
            code_agent_model = model
            self.async_model = None

        # Create CodeAgent with all parameters transparently passed
        self.base_agent = CodeAgent(
            tools=tools,
            model=code_agent_model,
            prompt_templates=prompt_templates,
            additional_authorized_imports=additional_authorized_imports,
            planning_interval=planning_interval,
            executor_type=executor_type,
            executor_kwargs=executor_kwargs,
            max_print_outputs_length=max_print_outputs_length,
            stream_outputs=stream_outputs,
            use_structured_outputs_internally=use_structured_outputs_internally,
            code_block_tags=code_block_tags,
            **kwargs
        )

        # Execution tracking
        self.execution_history: List[StepExecutionResult] = []
        self.current_step = 0
        self.is_executing = False

        # Memory state management
        self.memory_snapshots: Dict[int, Any] = {}
        self.execution_lock = asyncio.Lock()

        logger.info(f"AsyncCodeAgent created: {self.agent_id}")

    async def execute_step_async(
        self,
        step_input: Any,
        step_number: int,
        sync_barrier: Optional[StepSyncBarrier] = None,
        **kwargs
    ) -> StepExecutionResult:
        """
        Execute a single step asynchronously with optional synchronization.

        This method coordinates with other agents through the sync barrier for
        step-level batch processing.
        """
        execution_id = str(uuid.uuid4())
        start_time = time.time()

        logger.debug(f"Agent {self.agent_id} executing step {step_number} (exec: {execution_id[:8]})")

        async with self.execution_lock:
            try:
                # Prepare step input
                if isinstance(step_input, str):
                    step_messages = [{"role": "user", "content": step_input}]
                elif isinstance(step_input, list):
                    step_messages = step_input
                else:
                    step_messages = [{"role": "user", "content": str(step_input)}]

                # Get current memory state
                memory_state = self.get_current_memory_state()

                # Step-level synchronization
                if sync_barrier and self.async_config.enable_step_sync:
                    sync_result = await sync_barrier.wait_for_step_sync(
                        agent_id=self.agent_id,
                        step_number=step_number,
                        agent_prompt=step_messages,
                        memory_state=memory_state,
                        timeout=self.async_config.step_sync_timeout
                    )

                    # Execute as part of batch if sync succeeded
                    if sync_result.get('sync_completed'):
                        response = await self._execute_batch_step(sync_result, step_messages)
                        batch_info = sync_result.get('batch_info')
                    else:
                        response = await self._execute_individual_step(step_messages)
                        batch_info = None
                else:
                    # Execute individually
                    response = await self._execute_individual_step(step_messages)
                    batch_info = None

                # Create result
                execution_time = time.time() - start_time
                result = StepExecutionResult(
                    step_number=step_number,
                    agent_id=self.agent_id,
                    execution_id=execution_id,
                    success=True,
                    response=response,
                    memory_state=self.get_current_memory_state(),
                    batch_info=batch_info,
                    execution_time=execution_time
                )

                # Update tracking
                self.execution_history.append(result)
                self.current_step = max(self.current_step, step_number + 1)

                logger.debug(f"Agent {self.agent_id} completed step {step_number} in {execution_time:.2f}s")
                return result

            except Exception as e:
                execution_time = time.time() - start_time
                error_result = StepExecutionResult(
                    step_number=step_number,
                    agent_id=self.agent_id,
                    execution_id=execution_id,
                    success=False,
                    error_message=str(e),
                    execution_time=execution_time
                )

                self.execution_history.append(error_result)
                logger.error(f"Agent {self.agent_id} step {step_number} failed: {e}")

                # Retry if configured
                if self.async_config.retry_failed_steps and kwargs.get('retry_count', 0) < self.async_config.max_step_retries:
                    logger.info(f"Retrying step {step_number} for agent {self.agent_id}")
                    await asyncio.sleep(1.0)  # Brief delay before retry
                    kwargs['retry_count'] = kwargs.get('retry_count', 0) + 1
                    return await self.execute_step_async(step_input, step_number, sync_barrier, **kwargs)

                raise

    async def _execute_batch_step(self, sync_result: Dict[str, Any], step_messages: List[Any]) -> Any:
        """Execute step as part of a synchronized batch."""
        batch_info = sync_result.get('batch_info', {})
        batch_prompts = sync_result.get('batch_prompts', [])
        agent_prompt_map = sync_result.get('agent_prompt_map', {})

        logger.debug(f"Agent {self.agent_id} executing in batch {batch_info.get('batch_id')} with {len(batch_prompts)} agents")

        if self.async_model and hasattr(self.async_model, 'generate_batch_async'):
            # Use async batch generation
            if self.async_model:
                batch_responses = await self.async_model.generate_batch_async(batch_prompts)
            else:
                # Fall back to individual generation
                return await self._execute_individual_step(step_messages)

            # Find this agent's response in the batch
            agent_index = None
            for idx, agent_id in agent_prompt_map.items():
                if agent_id == self.agent_id:
                    agent_index = idx
                    break

            if agent_index is not None and agent_index < len(batch_responses):
                return batch_responses[agent_index]
            else:
                logger.warning(f"Agent {self.agent_id} not found in batch response, falling back to individual")
                return await self._execute_individual_step(step_messages)
        else:
            # Fallback to individual execution
            return await self._execute_individual_step(step_messages)

    async def _execute_individual_step(self, step_messages: List[Any]) -> Any:
        """Execute step individually (non-batched)."""
        if self.async_model and hasattr(self.async_model, 'generate_async'):
            # Use async model
            if self.async_model:
                return await self.async_model.generate_async(step_messages)
            else:
                # Fall back to sync generation
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    self.base_agent.model.generate,
                    step_messages
                )
        else:
            # Use sync model in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.base_agent.model.generate,
                step_messages
            )

    async def execute_sequence_async(
        self,
        initial_input: Any,
        max_steps: int = 10,
        sync_barrier: Optional[StepSyncBarrier] = None,
        **kwargs
    ) -> List[StepExecutionResult]:
        """
        Execute a complete sequence of steps asynchronously.

        This method runs the agent through multiple steps, optionally synchronizing
        with other agents at each step for step-level batch processing.
        """
        logger.info(f"Agent {self.agent_id} starting sequence execution (max_steps: {max_steps})")

        self.is_executing = True
        sequence_results = []

        try:
            current_input = initial_input

            for step_num in range(max_steps):
                # Execute current step
                step_result = await self.execute_step_async(
                    step_input=current_input,
                    step_number=step_num,
                    sync_barrier=sync_barrier,
                    **kwargs
                )

                sequence_results.append(step_result)

                # Check if we should continue
                if not step_result.success:
                    logger.warning(f"Agent {self.agent_id} stopping sequence due to step failure")
                    break

                # Prepare input for next step (this would need agent-specific logic)
                current_input = self._prepare_next_step_input(step_result)

                # Check for termination conditions
                if self._should_terminate_sequence(step_result, step_num):
                    logger.info(f"Agent {self.agent_id} sequence terminated at step {step_num}")
                    break

        finally:
            self.is_executing = False

        logger.info(f"Agent {self.agent_id} completed sequence: {len(sequence_results)} steps")
        return sequence_results

    def _prepare_next_step_input(self, previous_result: StepExecutionResult) -> Any:
        """Prepare input for the next step based on previous result."""
        # This is a simple default implementation
        # In practice, this would be agent-specific logic
        if previous_result.response and hasattr(previous_result.response, 'content'):
            return previous_result.response.content
        else:
            return str(previous_result.response)

    def _should_terminate_sequence(self, step_result: StepExecutionResult, step_number: int) -> bool:
        """Determine if the sequence should terminate."""
        # Simple default termination logic
        # In practice, this would be agent-specific
        if step_result.response and hasattr(step_result.response, 'content'):
            content = step_result.response.content.lower()
            if any(term in content for term in ['final answer', 'conclusion', 'done', 'complete']):
                return True
        return False

    def get_current_memory_state(self) -> Optional[Any]:
        """Get the current memory state for synchronization."""
        if hasattr(self.base_agent, 'memory') and self.base_agent.memory:
            # Create a snapshot of current memory
            memory_snapshot = {
                'steps': list(self.base_agent.memory.steps),
                'system_prompt': getattr(self.base_agent.memory, 'system_prompt', ''),
                'step_count': len(self.base_agent.memory.steps)
            }
            return memory_snapshot
        return None

    async def set_memory_state(self, memory_state: Any) -> bool:
        """Set the memory state from synchronization."""
        try:
            if memory_state and hasattr(self.base_agent, 'memory') and self.base_agent.memory:
                # Restore memory state
                if isinstance(memory_state, dict) and 'steps' in memory_state:
                    self.base_agent.memory.steps = list(memory_state['steps'])
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to set memory state for agent {self.agent_id}: {e}")
            return False

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution performance metrics."""
        if not self.execution_history:
            return {'total_steps': 0}

        successful_steps = [r for r in self.execution_history if r.success]
        failed_steps = [r for r in self.execution_history if not r.success]

        total_time = sum(r.execution_time for r in self.execution_history)
        avg_time = total_time / len(self.execution_history) if self.execution_history else 0

        batch_steps = [r for r in self.execution_history if r.batch_info]
        individual_steps = [r for r in self.execution_history if not r.batch_info]

        return {
            'agent_id': self.agent_id,
            'total_steps': len(self.execution_history),
            'successful_steps': len(successful_steps),
            'failed_steps': len(failed_steps),
            'success_rate': len(successful_steps) / len(self.execution_history) if self.execution_history else 0,
            'total_execution_time': total_time,
            'average_step_time': avg_time,
            'batch_steps': len(batch_steps),
            'individual_steps': len(individual_steps),
            'batch_ratio': len(batch_steps) / len(self.execution_history) if self.execution_history else 0,
            'current_step': self.current_step,
            'is_executing': self.is_executing,
            'config': {
                'execution_mode': self.config.execution_mode.value,
                'step_sync_enabled': self.config.enable_step_sync,
                'max_concurrent': self.config.max_concurrent_agents
            }
        }


class AsyncAgentPool:
    """
    Pool for managing multiple async agents with coordinated execution.

    This class provides utilities for managing multiple agents that can execute
    both independently and in coordinated step-level batch processing.
    """

    def __init__(self, max_concurrent_agents: int = 10):
        self.agents: Dict[str, AsyncCodeAgent] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent_agents)
        self.execution_lock = asyncio.Lock()

    def add_agent(self, agent: AsyncCodeAgent) -> bool:
        """Add an agent to the pool."""
        if agent.agent_id in self.agents:
            logger.warning(f"Agent {agent.agent_id} already in pool")
            return False

        self.agents[agent.agent_id] = agent
        logger.info(f"Added agent {agent.agent_id} to pool (total: {len(self.agents)})")
        return True

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the pool."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Removed agent {agent_id} from pool")
            return True
        return False

    async def execute_agents_parallel(
        self,
        agent_inputs: Dict[str, Any],
        max_steps: int = 10,
        sync_barrier: Optional[StepSyncBarrier] = None
    ) -> Dict[str, List[StepExecutionResult]]:
        """
        Execute multiple agents in parallel with optional step synchronization.

        Args:
            agent_inputs: Dict mapping agent_id to their initial input
            max_steps: Maximum steps per agent
            sync_barrier: Optional sync barrier for step-level coordination

        Returns:
            Dict mapping agent_id to their execution results
        """
        logger.info(f"Executing {len(agent_inputs)} agents in parallel")

        async def execute_single_agent(agent_id: str, initial_input: Any):
            async with self.semaphore:
                if agent_id not in self.agents:
                    logger.error(f"Agent {agent_id} not found in pool")
                    return []

                agent = self.agents[agent_id]
                return await agent.execute_sequence_async(
                    initial_input=initial_input,
                    max_steps=max_steps,
                    sync_barrier=sync_barrier
                )

        # Create tasks for all agents
        tasks = [
            execute_single_agent(agent_id, input_data)
            for agent_id, input_data in agent_inputs.items()
        ]

        # Execute all agents concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        execution_results = {}
        for i, (agent_id, result) in enumerate(zip(agent_inputs.keys(), results)):
            if isinstance(result, Exception):
                logger.error(f"Agent {agent_id} execution failed: {result}")
                execution_results[agent_id] = []
            else:
                execution_results[agent_id] = result

        logger.info(f"Parallel execution completed: {len(execution_results)} agents")
        return execution_results

    def get_pool_metrics(self) -> Dict[str, Any]:
        """Get metrics for the entire agent pool."""
        if not self.agents:
            return {'total_agents': 0}

        agent_metrics = {
            agent_id: agent.get_execution_metrics()
            for agent_id, agent in self.agents.items()
        }

        total_steps = sum(m['total_steps'] for m in agent_metrics.values())
        total_time = sum(m['total_execution_time'] for m in agent_metrics.values())
        total_batch_steps = sum(m['batch_steps'] for m in agent_metrics.values())

        return {
            'total_agents': len(self.agents),
            'total_steps_all_agents': total_steps,
            'total_execution_time': total_time,
            'total_batch_steps': total_batch_steps,
            'overall_batch_ratio': total_batch_steps / total_steps if total_steps > 0 else 0,
            'average_steps_per_agent': total_steps / len(self.agents) if self.agents else 0,
            'agent_metrics': agent_metrics
        }