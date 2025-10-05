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
StepBatchCoordinator orchestrates step-level batch processing across multiple agents.

This module provides high-level coordination for step-level batch generation,
managing sync barriers, batch execution, and agent coordination for efficient
parallel processing of multiple agent trajectories.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .async_batch_vllm_model import AsyncBatchVLLMModel
from .step_sync_barrier import StepSyncBarrier, SyncBarrierConfig, SyncBarrierManager
from .async_agent import AsyncCodeAgent, AsyncAgentPool, StepExecutionResult, ExecutionMode

logger = logging.getLogger(__name__)


class CoordinationStrategy(Enum):
    """Strategies for coordinating step-level batch processing."""
    STRICT_SYNC = "strict_sync"  # All agents must sync at every step
    FLEXIBLE_SYNC = "flexible_sync"  # Agents can proceed with partial sync
    ADAPTIVE_BATCH = "adaptive_batch"  # Dynamically adjust batch sizes
    MIXED_MODE = "mixed_mode"  # Combine sync and async execution


@dataclass
class BatchCoordinationConfig:
    """Configuration for step-level batch coordination."""
    coordination_strategy: CoordinationStrategy = CoordinationStrategy.FLEXIBLE_SYNC
    max_agents_per_batch: int = 8
    step_sync_timeout: float = 30.0
    enable_partial_sync: bool = True
    min_agents_for_batch: int = 2
    max_wait_time_per_step: float = 45.0
    adaptive_batch_sizing: bool = True
    fallback_to_individual: bool = True
    enable_step_caching: bool = True
    max_concurrent_batches: int = 3


@dataclass
class BatchExecutionMetrics:
    """Metrics for batch execution performance."""
    total_steps_coordinated: int = 0
    total_agents_coordinated: int = 0
    successful_batch_steps: int = 0
    individual_fallback_steps: int = 0
    average_batch_size: float = 0.0
    average_step_time: float = 0.0
    total_coordination_time: float = 0.0
    sync_timeout_count: int = 0
    batch_efficiency: float = 0.0  # Ratio of batched vs individual steps


class StepBatchCoordinator:
    """
    Coordinates step-level batch processing across multiple agents.

    This class manages the complex orchestration of multiple agents executing
    in parallel with step-level synchronization and batch generation. It handles
    sync barriers, batch coordination, fallback mechanisms, and performance optimization.

    Key features:
    - Flexible coordination strategies (strict, flexible, adaptive)
    - Dynamic batch sizing based on agent readiness
    - Automatic fallback to individual execution
    - Performance monitoring and optimization
    - Support for mixed execution modes
    """

    def __init__(
        self,
        async_model: AsyncBatchVLLMModel,
        config: Optional[BatchCoordinationConfig] = None
    ):
        self.async_model = async_model
        self.config = config or BatchCoordinationConfig()

        # Core coordination components
        self.barrier_manager = SyncBarrierManager()
        self.agent_pool = AsyncAgentPool(max_concurrent_agents=self.config.max_agents_per_batch * 2)

        # Execution state
        self.active_sessions: Dict[str, 'CoordinationSession'] = {}
        self.execution_lock = asyncio.Lock()

        # Performance tracking
        self.metrics = BatchExecutionMetrics()
        self.step_cache: Dict[str, Any] = {}

        # Coordination control
        self.is_active = True
        self.coordination_tasks: Set[asyncio.Task] = set()

        logger.info(f"StepBatchCoordinator initialized with strategy: {self.config.coordination_strategy.value}")

    async def create_coordination_session(
        self,
        session_id: str,
        agent_configs: List[Dict[str, Any]],
        coordination_config: Optional[Dict[str, Any]] = None
    ) -> 'CoordinationSession':
        """
        Create a new coordination session for multiple agents.

        Args:
            session_id: Unique identifier for the coordination session
            agent_configs: List of agent configuration dictionaries
            coordination_config: Optional session-specific coordination config

        Returns:
            CoordinationSession object for managing the agents
        """
        logger.info(f"Creating coordination session: {session_id} with {len(agent_configs)} agents")

        # Create session-specific sync barrier
        barrier_config = SyncBarrierConfig(
            timeout_seconds=self.config.step_sync_timeout,
            max_wait_agents=self.config.max_agents_per_batch,
            enable_partial_sync=self.config.enable_partial_sync,
            min_agents_for_partial=self.config.min_agents_for_batch
        )

        sync_barrier = self.barrier_manager.create_barrier(f"{session_id}_barrier", barrier_config)

        # Create and configure agents
        session_agents = []
        for i, agent_config in enumerate(agent_configs):
            agent_id = agent_config.get('agent_id', f"{session_id}_agent_{i}")

            # Remove agent_id from config to avoid duplication
            filtered_config = {k: v for k, v in agent_config.items() if k != 'agent_id'}

            # Ensure tools parameter is provided (required by CodeAgent)
            filtered_config.setdefault('tools', [])

            # Create async agent with new interface
            async_agent = AsyncCodeAgent(
                # Required CodeAgent parameters
                tools=filtered_config.pop('tools', []),
                model=self.async_model,

                # Other CodeAgent parameters (optional)
                prompt_templates=filtered_config.pop('prompt_templates', None),
                additional_authorized_imports=filtered_config.pop('additional_authorized_imports', None),
                planning_interval=filtered_config.pop('planning_interval', None),
                executor_type=filtered_config.pop('executor_type', 'local'),
                executor_kwargs=filtered_config.pop('executor_kwargs', None),
                max_print_outputs_length=filtered_config.pop('max_print_outputs_length', None),
                stream_outputs=filtered_config.pop('stream_outputs', False),
                use_structured_outputs_internally=filtered_config.pop('use_structured_outputs_internally', False),
                code_block_tags=filtered_config.pop('code_block_tags', None),

                # Async-specific parameters
                agent_id=agent_id,
                async_config=filtered_config.pop('async_config', None),

                # Any remaining parameters
                **filtered_config
            )

            # Register with sync barrier
            await sync_barrier.register_agent(agent_id)

            # Add to agent pool
            self.agent_pool.add_agent(async_agent)
            session_agents.append(async_agent)

        # Create coordination session
        session = CoordinationSession(
            session_id=session_id,
            agents=session_agents,
            sync_barrier=sync_barrier,
            coordinator=self,
            config=coordination_config or {}
        )

        self.active_sessions[session_id] = session

        logger.info(f"Coordination session created: {session_id} with {len(session_agents)} agents")
        return session

    async def execute_coordinated_steps(
        self,
        session_id: str,
        agent_inputs: Dict[str, Any],
        max_steps: int = 10,
        step_configs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, List[StepExecutionResult]]:
        """
        Execute coordinated step-level batch processing for a session.

        This is the main method for running step-level batch coordination across
        multiple agents with synchronization and batch generation.

        Args:
            session_id: ID of the coordination session
            agent_inputs: Dict mapping agent_id to their initial inputs
            max_steps: Maximum number of steps to execute
            step_configs: Optional step-specific configurations

        Returns:
            Dict mapping agent_id to their execution results
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Coordination session {session_id} not found")

        session = self.active_sessions[session_id]
        logger.info(f"Starting coordinated execution for session {session_id}: {len(agent_inputs)} agents, {max_steps} max steps")

        start_time = time.time()
        all_results: Dict[str, List[StepExecutionResult]] = {
            agent_id: [] for agent_id in agent_inputs.keys()
        }

        try:
            # Execute steps with coordination
            for step_num in range(max_steps):
                logger.debug(f"Session {session_id} executing step {step_num}")

                step_start_time = time.time()

                # Execute coordinated step
                step_results = await self._execute_coordinated_step(
                    session=session,
                    step_number=step_num,
                    agent_inputs=agent_inputs,
                    step_config=step_configs[step_num] if step_configs and step_num < len(step_configs) else {}
                )

                # Process results and update inputs for next step
                active_agents = set()
                for agent_id, result in step_results.items():
                    all_results[agent_id].append(result)

                    if result.success:
                        # Prepare input for next step
                        agent_inputs[agent_id] = self._prepare_next_step_input(result)
                        active_agents.add(agent_id)
                    else:
                        logger.warning(f"Agent {agent_id} failed at step {step_num}, removing from coordination")

                # Update metrics
                step_time = time.time() - step_start_time
                self._update_step_metrics(step_results, step_time)

                # Check if any agents are still active
                if not active_agents:
                    logger.info(f"Session {session_id} completed: no active agents remaining")
                    break

                # Filter agent_inputs to only active agents
                agent_inputs = {agent_id: agent_inputs[agent_id] for agent_id in active_agents}

                logger.debug(f"Session {session_id} step {step_num} completed in {step_time:.2f}s, {len(active_agents)} agents continuing")

        except Exception as e:
            logger.error(f"Coordinated execution failed for session {session_id}: {e}")
            raise

        finally:
            execution_time = time.time() - start_time
            self.metrics.total_coordination_time += execution_time

            logger.info(f"Session {session_id} coordination completed in {execution_time:.2f}s")

        return all_results

    async def _execute_coordinated_step(
        self,
        session: 'CoordinationSession',
        step_number: int,
        agent_inputs: Dict[str, Any],
        step_config: Dict[str, Any]
    ) -> Dict[str, StepExecutionResult]:
        """Execute a single coordinated step across multiple agents."""

        coordination_strategy = step_config.get('strategy', self.config.coordination_strategy)
        logger.debug(f"Session {session.session_id} step {step_number}: {coordination_strategy.value} with {len(agent_inputs)} agents")

        if coordination_strategy == CoordinationStrategy.STRICT_SYNC:
            return await self._execute_strict_sync_step(session, step_number, agent_inputs, step_config)
        elif coordination_strategy == CoordinationStrategy.FLEXIBLE_SYNC:
            return await self._execute_flexible_sync_step(session, step_number, agent_inputs, step_config)
        elif coordination_strategy == CoordinationStrategy.ADAPTIVE_BATCH:
            return await self._execute_adaptive_batch_step(session, step_number, agent_inputs, step_config)
        elif coordination_strategy == CoordinationStrategy.MIXED_MODE:
            return await self._execute_mixed_mode_step(session, step_number, agent_inputs, step_config)
        else:
            # Default to flexible sync
            return await self._execute_flexible_sync_step(session, step_number, agent_inputs, step_config)

    async def _execute_strict_sync_step(
        self,
        session: 'CoordinationSession',
        step_number: int,
        agent_inputs: Dict[str, Any],
        step_config: Dict[str, Any]
    ) -> Dict[str, StepExecutionResult]:
        """Execute step with strict synchronization - all agents must sync."""

        agent_tasks = []
        for agent_id, step_input in agent_inputs.items():
            agent = next((a for a in session.agents if a.agent_id == agent_id), None)
            if agent:
                task = agent.execute_step_async(
                    step_input=step_input,
                    step_number=step_number,
                    sync_barrier=session.sync_barrier
                )
                agent_tasks.append((agent_id, task))

        # Wait for all agents to complete
        results = {}
        completed_tasks = await asyncio.gather(
            *[task for _, task in agent_tasks],
            return_exceptions=True
        )

        for (agent_id, _), result in zip(agent_tasks, completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"Agent {agent_id} failed in strict sync: {result}")
                # Create failure result
                results[agent_id] = StepExecutionResult(
                    step_number=step_number,
                    agent_id=agent_id,
                    execution_id=str(uuid.uuid4()),
                    success=False,
                    error_message=str(result)
                )
            else:
                results[agent_id] = result

        return results

    async def _execute_flexible_sync_step(
        self,
        session: 'CoordinationSession',
        step_number: int,
        agent_inputs: Dict[str, Any],
        step_config: Dict[str, Any]
    ) -> Dict[str, StepExecutionResult]:
        """Execute step with flexible synchronization - allow partial sync."""

        # Start all agent tasks
        agent_tasks = {}
        for agent_id, step_input in agent_inputs.items():
            agent = next((a for a in session.agents if a.agent_id == agent_id), None)
            if agent:
                task = asyncio.create_task(agent.execute_step_async(
                    step_input=step_input,
                    step_number=step_number,
                    sync_barrier=session.sync_barrier
                ))
                agent_tasks[agent_id] = task

        # Wait for tasks with timeout
        timeout = step_config.get('timeout', self.config.max_wait_time_per_step)

        try:
            results = {}
            done, pending = await asyncio.wait(
                agent_tasks.values(),
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )

            # Process completed tasks
            for task in done:
                try:
                    result = await task
                    results[result.agent_id] = result
                except Exception as e:
                    # Find agent_id for this task
                    agent_id = next((aid for aid, t in agent_tasks.items() if t == task), "unknown")
                    logger.error(f"Agent {agent_id} failed in flexible sync: {e}")
                    results[agent_id] = StepExecutionResult(
                        step_number=step_number,
                        agent_id=agent_id,
                        execution_id=str(uuid.uuid4()),
                        success=False,
                        error_message=str(e)
                    )

            # Handle pending (timed out) tasks
            for task in pending:
                task.cancel()
                agent_id = next((aid for aid, t in agent_tasks.items() if t == task), "unknown")
                logger.warning(f"Agent {agent_id} timed out in flexible sync")
                results[agent_id] = StepExecutionResult(
                    step_number=step_number,
                    agent_id=agent_id,
                    execution_id=str(uuid.uuid4()),
                    success=False,
                    error_message="Execution timeout"
                )

            return results

        except Exception as e:
            logger.error(f"Flexible sync step failed: {e}")
            # Cancel all pending tasks
            for task in agent_tasks.values():
                if not task.done():
                    task.cancel()
            raise

    async def _execute_adaptive_batch_step(
        self,
        session: 'CoordinationSession',
        step_number: int,
        agent_inputs: Dict[str, Any],
        step_config: Dict[str, Any]
    ) -> Dict[str, StepExecutionResult]:
        """Execute step with adaptive batch sizing based on agent readiness."""

        # Implement adaptive batching logic
        # This would dynamically adjust batch sizes based on agent performance
        # For now, fall back to flexible sync
        logger.debug(f"Adaptive batch step {step_number} falling back to flexible sync")
        return await self._execute_flexible_sync_step(session, step_number, agent_inputs, step_config)

    async def _execute_mixed_mode_step(
        self,
        session: 'CoordinationSession',
        step_number: int,
        agent_inputs: Dict[str, Any],
        step_config: Dict[str, Any]
    ) -> Dict[str, StepExecutionResult]:
        """Execute step with mixed mode - combine sync and async execution."""

        # Implement mixed mode logic
        # This would intelligently choose between sync and async based on conditions
        # For now, fall back to flexible sync
        logger.debug(f"Mixed mode step {step_number} falling back to flexible sync")
        return await self._execute_flexible_sync_step(session, step_number, agent_inputs, step_config)

    def _prepare_next_step_input(self, result: StepExecutionResult) -> Any:
        """Prepare input for the next step based on previous result."""
        if result.response and hasattr(result.response, 'content'):
            return result.response.content
        else:
            return str(result.response) if result.response else ""

    def _update_step_metrics(self, step_results: Dict[str, StepExecutionResult], step_time: float):
        """Update coordination metrics based on step results."""
        self.metrics.total_steps_coordinated += 1
        self.metrics.total_agents_coordinated += len(step_results)
        self.metrics.average_step_time = (
            (self.metrics.average_step_time * (self.metrics.total_steps_coordinated - 1) + step_time) /
            self.metrics.total_steps_coordinated
        )

        # Count successful steps and batch vs individual execution
        successful_steps = sum(1 for r in step_results.values() if r.success)
        batch_steps = sum(1 for r in step_results.values() if r.batch_info)

        self.metrics.successful_batch_steps += successful_steps
        if batch_steps > 0:
            # Update batch size tracking
            total_batch_ops = self.metrics.successful_batch_steps + self.metrics.individual_fallback_steps
            if total_batch_ops > 0:
                self.metrics.average_batch_size = (
                    (self.metrics.average_batch_size * (total_batch_ops - successful_steps) + batch_steps) /
                    total_batch_ops
                )
        else:
            self.metrics.individual_fallback_steps += successful_steps

        # Update batch efficiency
        total_steps = self.metrics.successful_batch_steps + self.metrics.individual_fallback_steps
        if total_steps > 0:
            self.metrics.batch_efficiency = self.metrics.successful_batch_steps / total_steps

    async def shutdown_session(self, session_id: str) -> bool:
        """Shutdown a coordination session and clean up resources."""
        if session_id not in self.active_sessions:
            return False

        logger.info(f"Shutting down coordination session: {session_id}")

        session = self.active_sessions[session_id]

        # Remove agents from pool
        for agent in session.agents:
            self.agent_pool.remove_agent(agent.agent_id)

        # Shutdown sync barrier
        await self.barrier_manager.shutdown_barrier(f"{session_id}_barrier")

        # Remove session
        del self.active_sessions[session_id]

        logger.info(f"Coordination session shutdown complete: {session_id}")
        return True

    async def shutdown_all(self):
        """Shutdown all coordination sessions and clean up resources."""
        logger.info(f"Shutting down StepBatchCoordinator: {len(self.active_sessions)} sessions")

        # Shutdown all sessions
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            await self.shutdown_session(session_id)

        # Cancel coordination tasks
        for task in self.coordination_tasks:
            if not task.done():
                task.cancel()

        # Shutdown barrier manager
        await self.barrier_manager.shutdown_all()

        self.is_active = False
        logger.info("StepBatchCoordinator shutdown complete")

    def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get comprehensive coordination performance metrics."""
        agent_pool_metrics = self.agent_pool.get_pool_metrics()

        return {
            'coordinator_metrics': {
                'total_steps_coordinated': self.metrics.total_steps_coordinated,
                'total_agents_coordinated': self.metrics.total_agents_coordinated,
                'successful_batch_steps': self.metrics.successful_batch_steps,
                'individual_fallback_steps': self.metrics.individual_fallback_steps,
                'average_batch_size': self.metrics.average_batch_size,
                'average_step_time': self.metrics.average_step_time,
                'total_coordination_time': self.metrics.total_coordination_time,
                'batch_efficiency': self.metrics.batch_efficiency,
                'sync_timeout_count': self.metrics.sync_timeout_count
            },
            'agent_pool_metrics': agent_pool_metrics,
            'active_sessions': len(self.active_sessions),
            'coordination_strategy': self.config.coordination_strategy.value,
            'config': {
                'max_agents_per_batch': self.config.max_agents_per_batch,
                'step_sync_timeout': self.config.step_sync_timeout,
                'enable_partial_sync': self.config.enable_partial_sync,
                'min_agents_for_batch': self.config.min_agents_for_batch
            }
        }


@dataclass
class CoordinationSession:
    """Represents a coordination session for multiple agents."""
    session_id: str
    agents: List[AsyncCodeAgent]
    sync_barrier: StepSyncBarrier
    coordinator: StepBatchCoordinator
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status."""
        return {
            'session_id': self.session_id,
            'agent_count': len(self.agents),
            'agent_ids': [agent.agent_id for agent in self.agents],
            'barrier_status': self.sync_barrier.get_sync_status(),
            'created_at': self.created_at,
            'config': self.config
        }