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
StepSyncBarrier provides synchronization mechanisms for step-level batch processing.

This module enables multiple agents to synchronize at specific steps, allowing for
efficient step-level batch generation where multiple agents can proceed together
through their execution steps.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Set, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from smolagents.memory import AgentMemory, ActionStep, TaskStep
    SMOLAGENTS_AVAILABLE = True
    Memory = AgentMemory  # For compatibility
except ImportError:
    SMOLAGENTS_AVAILABLE = False
    Memory = None
    AgentMemory = None
    ActionStep = None
    TaskStep = None


class SyncState(Enum):
    """Agent synchronization states."""
    WAITING = "waiting"
    READY = "ready"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentSyncStatus:
    """Status tracking for an agent at a sync barrier."""
    agent_id: str
    sync_state: SyncState = SyncState.WAITING
    step_number: int = 0
    last_update: float = field(default_factory=time.time)
    memory_state: Optional[Any] = None
    error_message: Optional[str] = None

    def update_state(self, new_state: SyncState, step_number: Optional[int] = None, error: Optional[str] = None):
        """Update agent sync status."""
        self.sync_state = new_state
        if step_number is not None:
            self.step_number = step_number
        if error is not None:
            self.error_message = error
        self.last_update = time.time()


@dataclass
class SyncBarrierConfig:
    """Configuration for sync barrier behavior."""
    timeout_seconds: float = 30.0
    max_wait_agents: int = 10
    enable_partial_sync: bool = False  # Allow proceeding with subset of agents
    min_agents_for_partial: int = 2
    retry_failed_agents: bool = True
    max_retries: int = 3


class StepSyncBarrier:
    """
    Synchronization barrier for coordinating multiple agents at specific steps.

    This class enables step-level batch processing by ensuring multiple agents
    reach synchronization points together, allowing their prompts to be batched
    for efficient parallel generation.

    Key features:
    - Async/await based synchronization
    - Configurable timeout and retry mechanisms
    - Support for partial synchronization (proceeding with subset)
    - Memory state preservation during sync
    - Error handling and agent failure recovery
    """

    def __init__(self, barrier_id: str, config: Optional[SyncBarrierConfig] = None):
        self.barrier_id = barrier_id
        self.config = config or SyncBarrierConfig()

        # Agent tracking
        self.agent_statuses: Dict[str, AgentSyncStatus] = {}
        self.registered_agents: Set[str] = set()

        # Synchronization primitives
        self.sync_event = asyncio.Event()
        self.sync_lock = asyncio.Lock()
        self.step_condition = asyncio.Condition()

        # Batch coordination
        self.current_step = 0
        self.batch_ready_agents: Set[str] = set()
        self.sync_results: Dict[str, Any] = {}

        # State management
        self.is_active = True
        self.sync_history: List[Dict[str, Any]] = []

        logger.info(f"StepSyncBarrier created: {barrier_id}")

    async def register_agent(self, agent_id: str, initial_memory: Optional[Any] = None) -> bool:
        """
        Register an agent with the sync barrier.

        Args:
            agent_id: Unique identifier for the agent
            initial_memory: Initial memory state for the agent

        Returns:
            True if registration successful, False otherwise
        """
        async with self.sync_lock:
            if len(self.registered_agents) >= self.config.max_wait_agents:
                logger.warning(f"Cannot register agent {agent_id}: max agents reached")
                return False

            self.registered_agents.add(agent_id)
            self.agent_statuses[agent_id] = AgentSyncStatus(
                agent_id=agent_id,
                memory_state=initial_memory
            )

            logger.debug(f"Agent registered: {agent_id} (total: {len(self.registered_agents)})")
            return True

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the sync barrier."""
        async with self.sync_lock:
            if agent_id in self.registered_agents:
                self.registered_agents.remove(agent_id)
                self.agent_statuses.pop(agent_id, None)
                self.batch_ready_agents.discard(agent_id)

                logger.debug(f"Agent unregistered: {agent_id}")

                # Check if we should proceed with remaining agents
                await self._check_sync_conditions()
                return True
            return False

    async def wait_for_step_sync(
        self,
        agent_id: str,
        step_number: int,
        agent_prompt: List[Any],
        memory_state: Optional[Any] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Wait for multiple agents to synchronize at a specific step.

        This is the core method for step-level batch processing. Agents call this
        when they're ready to generate their next response, and the barrier waits
        until enough agents are ready to proceed with batch generation.

        Args:
            agent_id: Identifier for the calling agent
            step_number: The step number the agent is ready to process
            agent_prompt: The prompt/messages for generation
            memory_state: Current memory state of the agent
            timeout: Optional timeout override

        Returns:
            Dict containing sync results and batch information
        """
        if agent_id not in self.registered_agents:
            raise ValueError(f"Agent {agent_id} not registered with barrier")

        timeout = timeout or self.config.timeout_seconds

        # Update agent status
        status = self.agent_statuses[agent_id]
        status.update_state(SyncState.READY, step_number)
        status.memory_state = memory_state

        # Store agent's prompt for batching
        sync_data = {
            'agent_id': agent_id,
            'step_number': step_number,
            'prompt': agent_prompt,
            'memory_state': memory_state,
            'timestamp': time.time()
        }

        logger.debug(f"Agent {agent_id} waiting for step {step_number} sync")

        try:
            # Wait for synchronization with timeout
            result = await asyncio.wait_for(
                self._wait_for_sync_completion(agent_id, sync_data),
                timeout=timeout
            )

            status.update_state(SyncState.COMPLETED)
            return result

        except asyncio.TimeoutError:
            status.update_state(SyncState.FAILED, error="Sync timeout")
            logger.warning(f"Sync timeout for agent {agent_id} at step {step_number}")

            # Try partial sync if enabled
            if self.config.enable_partial_sync:
                return await self._attempt_partial_sync(agent_id, sync_data)
            else:
                raise

        except Exception as e:
            status.update_state(SyncState.FAILED, error=str(e))
            logger.error(f"Sync failed for agent {agent_id}: {e}")
            raise

    async def _wait_for_sync_completion(self, agent_id: str, sync_data: Dict[str, Any]) -> Dict[str, Any]:
        """Wait for synchronization conditions to be met."""
        async with self.step_condition:
            # Add this agent to the ready set
            self.batch_ready_agents.add(agent_id)
            self.sync_results[agent_id] = sync_data

            # Check if we can proceed
            await self._check_sync_conditions()

            # Wait for sync completion
            while agent_id in self.batch_ready_agents and self.is_active:
                await self.step_condition.wait()

            # Return the sync result
            return self.sync_results.get(agent_id, {})

    async def _check_sync_conditions(self):
        """Check if synchronization conditions are met and proceed if so."""
        ready_count = len(self.batch_ready_agents)
        total_registered = len(self.registered_agents)

        # Determine if we should proceed
        should_proceed = False

        if ready_count == total_registered and ready_count > 0:
            # All agents ready
            should_proceed = True
            sync_type = "full"
        elif (self.config.enable_partial_sync and
              ready_count >= self.config.min_agents_for_partial):
            # Partial sync conditions met
            should_proceed = True
            sync_type = "partial"

        if should_proceed:
            await self._proceed_with_sync(sync_type)

    async def _proceed_with_sync(self, sync_type: str):
        """Proceed with synchronization and batch processing."""
        sync_batch_id = str(uuid.uuid4())[:8]
        ready_agents = list(self.batch_ready_agents)

        logger.info(f"Proceeding with {sync_type} sync: {len(ready_agents)} agents (batch: {sync_batch_id})")

        # Prepare batch data
        batch_prompts = []
        agent_prompt_map = {}

        for agent_id in ready_agents:
            sync_data = self.sync_results[agent_id]
            batch_prompts.append(sync_data['prompt'])
            agent_prompt_map[len(batch_prompts) - 1] = agent_id

        # Update sync results with batch information
        batch_info = {
            'sync_type': sync_type,
            'batch_id': sync_batch_id,
            'batch_size': len(batch_prompts),
            'agent_ids': ready_agents,
            'step_number': self.current_step,
            'timestamp': time.time()
        }

        # Add batch info to each agent's result
        for agent_id in ready_agents:
            self.sync_results[agent_id].update({
                'batch_info': batch_info,
                'sync_completed': True,
                'batch_prompts': batch_prompts,
                'agent_prompt_map': agent_prompt_map
            })

        # Record sync history
        self.sync_history.append({
            'step': self.current_step,
            'sync_type': sync_type,
            'batch_id': sync_batch_id,
            'agents': ready_agents,
            'timestamp': time.time()
        })

        # Clear ready agents and advance step
        self.batch_ready_agents.clear()
        self.current_step += 1

        # Notify all waiting agents
        self.step_condition.notify_all()

    async def _attempt_partial_sync(self, agent_id: str, sync_data: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt partial synchronization when full sync times out."""
        if len(self.batch_ready_agents) >= self.config.min_agents_for_partial:
            logger.info(f"Attempting partial sync with {len(self.batch_ready_agents)} agents")
            await self._proceed_with_sync("partial_timeout")
            return self.sync_results.get(agent_id, {})
        else:
            # Fallback to single agent processing
            logger.warning(f"Insufficient agents for partial sync, processing {agent_id} individually")
            return {
                'sync_type': 'individual',
                'batch_info': {
                    'batch_id': str(uuid.uuid4())[:8],
                    'batch_size': 1,
                    'agent_ids': [agent_id],
                    'step_number': self.current_step,
                    'timestamp': time.time()
                },
                'batch_prompts': [sync_data['prompt']],
                'agent_prompt_map': {0: agent_id},
                'sync_completed': True
            }

    async def force_sync_advance(self, reason: str = "manual") -> Dict[str, Any]:
        """Force synchronization to advance regardless of waiting agents."""
        async with self.step_condition:
            if self.batch_ready_agents:
                logger.info(f"Force advancing sync: {reason} (agents: {len(self.batch_ready_agents)})")
                await self._proceed_with_sync(f"forced_{reason}")
                return {
                    'forced': True,
                    'reason': reason,
                    'advanced_agents': len(self.batch_ready_agents)
                }
            return {'forced': False, 'reason': 'no_waiting_agents'}

    def get_sync_status(self) -> Dict[str, Any]:
        """Get current synchronization status."""
        return {
            'barrier_id': self.barrier_id,
            'current_step': self.current_step,
            'registered_agents': list(self.registered_agents),
            'ready_agents': list(self.batch_ready_agents),
            'agent_statuses': {
                agent_id: {
                    'state': status.sync_state.value,
                    'step': status.step_number,
                    'last_update': status.last_update,
                    'error': status.error_message
                }
                for agent_id, status in self.agent_statuses.items()
            },
            'is_active': self.is_active,
            'sync_history_count': len(self.sync_history)
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the sync barrier."""
        if not self.sync_history:
            return {'total_syncs': 0}

        total_syncs = len(self.sync_history)
        sync_types = {}
        total_agents_synced = 0

        for sync in self.sync_history:
            sync_type = sync['sync_type']
            sync_types[sync_type] = sync_types.get(sync_type, 0) + 1
            total_agents_synced += len(sync['agents'])

        avg_agents_per_sync = total_agents_synced / total_syncs if total_syncs > 0 else 0

        return {
            'total_syncs': total_syncs,
            'sync_types': sync_types,
            'total_agents_synced': total_agents_synced,
            'average_agents_per_sync': avg_agents_per_sync,
            'current_step': self.current_step,
            'registered_agents_count': len(self.registered_agents),
            'config': {
                'timeout': self.config.timeout_seconds,
                'max_agents': self.config.max_wait_agents,
                'partial_sync_enabled': self.config.enable_partial_sync
            }
        }

    async def shutdown(self):
        """Shutdown the sync barrier and release all waiting agents."""
        logger.info(f"Shutting down sync barrier: {self.barrier_id}")

        async with self.step_condition:
            self.is_active = False

            # Update all agent statuses
            for status in self.agent_statuses.values():
                if status.sync_state in [SyncState.WAITING, SyncState.READY]:
                    status.update_state(SyncState.FAILED, error="Barrier shutdown")

            # Clear waiting agents
            self.batch_ready_agents.clear()

            # Notify all waiting agents
            self.step_condition.notify_all()

        logger.info(f"Sync barrier shutdown complete: {self.barrier_id}")


class SyncBarrierManager:
    """
    Manager for multiple sync barriers to support complex multi-agent coordination.

    This class allows for hierarchical synchronization where different groups of
    agents can have their own sync barriers, or agents can participate in multiple
    barriers for different types of coordination.
    """

    def __init__(self):
        self.barriers: Dict[str, StepSyncBarrier] = {}
        self.barrier_configs: Dict[str, SyncBarrierConfig] = {}

    def create_barrier(
        self,
        barrier_id: str,
        config: Optional[SyncBarrierConfig] = None
    ) -> StepSyncBarrier:
        """Create a new sync barrier."""
        if barrier_id in self.barriers:
            raise ValueError(f"Barrier {barrier_id} already exists")

        config = config or SyncBarrierConfig()
        barrier = StepSyncBarrier(barrier_id, config)

        self.barriers[barrier_id] = barrier
        self.barrier_configs[barrier_id] = config

        logger.info(f"Created sync barrier: {barrier_id}")
        return barrier

    def get_barrier(self, barrier_id: str) -> Optional[StepSyncBarrier]:
        """Get an existing sync barrier."""
        return self.barriers.get(barrier_id)

    async def shutdown_barrier(self, barrier_id: str) -> bool:
        """Shutdown and remove a specific barrier."""
        if barrier_id in self.barriers:
            await self.barriers[barrier_id].shutdown()
            del self.barriers[barrier_id]
            del self.barrier_configs[barrier_id]
            logger.info(f"Shutdown barrier: {barrier_id}")
            return True
        return False

    async def shutdown_all(self):
        """Shutdown all sync barriers."""
        logger.info(f"Shutting down all sync barriers: {len(self.barriers)}")

        shutdown_tasks = [
            barrier.shutdown() for barrier in self.barriers.values()
        ]

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        self.barriers.clear()
        self.barrier_configs.clear()

        logger.info("All sync barriers shutdown complete")

    def get_global_status(self) -> Dict[str, Any]:
        """Get status of all managed barriers."""
        return {
            'total_barriers': len(self.barriers),
            'barriers': {
                barrier_id: barrier.get_sync_status()
                for barrier_id, barrier in self.barriers.items()
            }
        }