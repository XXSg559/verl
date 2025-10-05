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
AsyncTreeOrchestrator provides unified orchestration for both agent-level and step-level batch processing.

This module integrates all async components to provide a comprehensive orchestration system
that supports both types of batch generation for Tree GRPO training, with intelligent
mode selection and performance optimization.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .async_batch_vllm_model import AsyncBatchVLLMModel
from .step_batch_coordinator import StepBatchCoordinator, BatchCoordinationConfig, CoordinationStrategy
from .async_agent import AsyncCodeAgent, AsyncAgentPool, StepExecutionResult, ExecutionMode
from .step_sync_barrier import StepSyncBarrier, SyncBarrierConfig, SyncBarrierManager
from ..tree_memory import TreeMemory

logger = logging.getLogger(__name__)

try:
    from smolagents import CodeAgent
    from smolagents.models import ChatMessage, MessageRole
    from smolagents.memory import AgentMemory, TaskStep
    SMOLAGENTS_AVAILABLE = True
    Memory = AgentMemory  # For compatibility
except ImportError:
    SMOLAGENTS_AVAILABLE = False
    CodeAgent = None
    ChatMessage = None
    MessageRole = None
    Memory = None
    AgentMemory = None
    TaskStep = None


class BatchMode(Enum):
    """Batch processing modes."""
    AGENT_LEVEL = "agent_level"  # Multiple trajectories from same starting point
    STEP_LEVEL = "step_level"    # Multiple agents synchronized at each step
    HYBRID = "hybrid"            # Combination of both modes
    ADAPTIVE = "adaptive"        # Dynamically choose based on conditions


class OrchestrationStrategy(Enum):
    """Overall orchestration strategies."""
    PURE_AGENT_BATCH = "pure_agent_batch"
    PURE_STEP_BATCH = "pure_step_batch"
    HIERARCHICAL = "hierarchical"  # Agent-level batches with step-level coordination
    DYNAMIC = "dynamic"            # Switch modes based on performance/conditions


@dataclass
class AsyncTreeConfig:
    """Configuration for async tree orchestration."""
    batch_mode: BatchMode = BatchMode.HYBRID
    orchestration_strategy: OrchestrationStrategy = OrchestrationStrategy.HIERARCHICAL

    # Agent-level batch config
    agent_level_proposals: int = 3
    enable_agent_level_batch: bool = True
    agent_batch_timeout: float = 60.0

    # Step-level batch config
    step_level_coordination: bool = True
    max_agents_per_step_batch: int = 8
    step_sync_timeout: float = 30.0
    enable_step_partial_sync: bool = True

    # Tree GRPO specific
    max_tree_depth: int = 5
    enable_tree_pruning: bool = True
    quality_threshold: float = 0.6

    # Performance optimization
    enable_caching: bool = True
    max_concurrent_operations: int = 10
    adaptive_mode_switching: bool = True
    performance_monitoring: bool = True


@dataclass
class TreeExecutionResult:
    """Result of tree execution with both agent and step level information."""
    session_id: str
    execution_id: str
    success: bool

    # Agent-level results
    agent_branches: Dict[str, List[Any]] = field(default_factory=dict)
    agent_level_metrics: Dict[str, Any] = field(default_factory=dict)

    # Step-level results
    step_coordination_results: Dict[str, List[StepExecutionResult]] = field(default_factory=dict)
    step_level_metrics: Dict[str, Any] = field(default_factory=dict)

    # Tree structure
    tree_memory: Optional[TreeMemory] = None
    tree_branches: List[TreeMemory] = field(default_factory=list)

    # Performance
    total_execution_time: float = 0.0
    batch_efficiency: float = 0.0

    # Errors
    error_message: Optional[str] = None


class AsyncTreeOrchestrator:
    """
    Unified orchestrator for async agent-level and step-level batch processing.

    This class provides the highest-level interface for Tree GRPO training with
    comprehensive support for both batch generation modes, intelligent coordination,
    and performance optimization.

    Key features:
    - Unified agent-level and step-level batch processing
    - Intelligent mode selection and switching
    - Tree memory management for GRPO training
    - Performance monitoring and optimization
    - Flexible coordination strategies
    - Comprehensive error handling and recovery
    """

    def __init__(
        self,
        model: AsyncBatchVLLMModel,
        config: Optional[AsyncTreeConfig] = None
    ):
        self.model = model
        self.config = config or AsyncTreeConfig()

        # Core components
        self.step_coordinator = StepBatchCoordinator(
            async_model=model,
            config=BatchCoordinationConfig(
                max_agents_per_batch=self.config.max_agents_per_step_batch,
                step_sync_timeout=self.config.step_sync_timeout,
                enable_partial_sync=self.config.enable_step_partial_sync
            )
        )

        self.agent_pool = AsyncAgentPool(max_concurrent_agents=self.config.max_concurrent_operations)
        self.barrier_manager = SyncBarrierManager()

        # Tree management
        self.tree_memories: Dict[str, TreeMemory] = {}
        self.active_sessions: Dict[str, 'TreeSession'] = {}

        # Execution control
        self.execution_lock = asyncio.Lock()
        self.is_active = True

        # Performance tracking
        self.global_metrics = {
            'total_sessions': 0,
            'agent_level_executions': 0,
            'step_level_executions': 0,
            'hybrid_executions': 0,
            'total_execution_time': 0.0,
            'average_batch_efficiency': 0.0
        }

        logger.info(f"AsyncTreeOrchestrator initialized: {self.config.batch_mode.value} mode, {self.config.orchestration_strategy.value} strategy")

    async def create_tree_session(
        self,
        session_id: str,
        initial_prompt: Union[str, List[Any]],
        coordinator_agent: Optional[CodeAgent] = None,
        session_config: Optional[Dict[str, Any]] = None
    ) -> 'TreeSession':
        """
        Create a new tree execution session.

        Args:
            session_id: Unique identifier for the session
            initial_prompt: Initial prompt or message list
            coordinator_agent: Optional coordinator agent for managing sub-agents
            session_config: Optional session-specific configuration

        Returns:
            TreeSession object for managing the execution
        """
        logger.info(f"Creating tree session: {session_id}")

        # Create tree memory
        system_prompt = "You are coordinating tree-based problem solving with multiple agents."
        if coordinator_agent and hasattr(coordinator_agent, 'system_prompt'):
            system_prompt = coordinator_agent.system_prompt

        tree_memory = TreeMemory(system_prompt)

        # Add initial task
        if isinstance(initial_prompt, str):
            initial_messages = [{"role": "user", "content": initial_prompt}]
        else:
            initial_messages = initial_prompt

        # Convert to memory format if needed
        from smolagents.memory import TaskStep
        if initial_messages:
            task_content = initial_messages[0].get('content', str(initial_messages[0]))
            task_step = TaskStep(task=task_content)
            tree_memory.steps.append(task_step)

        self.tree_memories[session_id] = tree_memory

        # Create session
        session = TreeSession(
            session_id=session_id,
            tree_memory=tree_memory,
            coordinator_agent=coordinator_agent,
            orchestrator=self,
            config=session_config or {}
        )

        self.active_sessions[session_id] = session
        self.global_metrics['total_sessions'] += 1

        logger.info(f"Tree session created: {session_id}")
        return session

    async def execute_agent_level_batch(
        self,
        session_id: str,
        num_branches: Optional[int] = None,
        branch_configs: Optional[List[Dict[str, Any]]] = None
    ) -> TreeExecutionResult:
        """
        Execute agent-level batch generation for multiple parallel trajectories.

        This creates multiple independent agent trajectories from the same starting
        point, suitable for agent-level batch processing in Tree GRPO.

        Args:
            session_id: ID of the tree session
            num_branches: Number of parallel trajectories to generate
            branch_configs: Optional configuration for each branch

        Returns:
            TreeExecutionResult with agent-level batch results
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Tree session {session_id} not found")

        session = self.active_sessions[session_id]
        num_branches = num_branches or self.config.agent_level_proposals

        logger.info(f"Executing agent-level batch for session {session_id}: {num_branches} branches")

        start_time = time.time()
        execution_id = str(uuid.uuid4())

        try:
            # Create agent-level branches from tree memory
            agent_branches = session.tree_memory.create_agent_level_branches(num_branches)

            # Prepare branch execution
            branch_tasks = []
            for i, branch_memory in enumerate(agent_branches):
                branch_config = branch_configs[i] if branch_configs and i < len(branch_configs) else {}

                # Create async agent for this branch
                async_agent = AsyncCodeAgent(
                    model=self.model,
                    agent_id=f"{session_id}_agent_branch_{i}",
                    config=branch_config.get('agent_config')
                )

                # Set branch memory
                await async_agent.set_memory_state({
                    'steps': branch_memory.steps,
                    'system_prompt': branch_memory.system_prompt,
                    'branch_metadata': branch_memory.branch_metadata
                })

                # Create execution task
                branch_task = self._execute_agent_branch(
                    agent=async_agent,
                    branch_memory=branch_memory,
                    branch_config=branch_config
                )
                branch_tasks.append((f"branch_{i}", branch_task))

            # Execute all branches concurrently with timeout
            timeout = branch_configs[0].get('timeout', self.config.agent_batch_timeout) if branch_configs else self.config.agent_batch_timeout

            logger.debug(f"Starting {len(branch_tasks)} agent branches with {timeout}s timeout")

            branch_results = {}
            completed_tasks = await asyncio.wait_for(
                asyncio.gather(*[task for _, task in branch_tasks], return_exceptions=True),
                timeout=timeout
            )

            # Process results
            for (branch_id, _), result in zip(branch_tasks, completed_tasks):
                if isinstance(result, Exception):
                    logger.error(f"Agent branch {branch_id} failed: {result}")
                    branch_results[branch_id] = {'success': False, 'error': str(result)}
                else:
                    branch_results[branch_id] = result

            # Create execution result
            execution_time = time.time() - start_time

            result = TreeExecutionResult(
                session_id=session_id,
                execution_id=execution_id,
                success=True,
                agent_branches=branch_results,
                agent_level_metrics=self._calculate_agent_level_metrics(branch_results),
                tree_memory=session.tree_memory,
                tree_branches=agent_branches,
                total_execution_time=execution_time,
                batch_efficiency=self._calculate_batch_efficiency(branch_results)
            )

            # Update global metrics
            self.global_metrics['agent_level_executions'] += 1
            self.global_metrics['total_execution_time'] += execution_time

            logger.info(f"Agent-level batch completed for session {session_id}: {len(branch_results)} branches in {execution_time:.2f}s")
            return result

        except asyncio.TimeoutError:
            error_msg = f"Agent-level batch timed out after {timeout}s"
            logger.error(error_msg)
            return TreeExecutionResult(
                session_id=session_id,
                execution_id=execution_id,
                success=False,
                error_message=error_msg,
                total_execution_time=time.time() - start_time
            )
        except Exception as e:
            error_msg = f"Agent-level batch failed: {e}"
            logger.error(error_msg)
            return TreeExecutionResult(
                session_id=session_id,
                execution_id=execution_id,
                success=False,
                error_message=error_msg,
                total_execution_time=time.time() - start_time
            )

    async def execute_step_level_batch(
        self,
        session_id: str,
        agent_configs: List[Dict[str, Any]],
        max_steps: int = 10,
        coordination_strategy: Optional[CoordinationStrategy] = None
    ) -> TreeExecutionResult:
        """
        Execute step-level batch coordination for synchronized multi-agent execution.

        This coordinates multiple agents to execute steps synchronously, enabling
        step-level batch processing for Tree GRPO training.

        Args:
            session_id: ID of the tree session
            agent_configs: Configuration for each agent in the coordination
            max_steps: Maximum number of steps to execute
            coordination_strategy: Strategy for step-level coordination

        Returns:
            TreeExecutionResult with step-level coordination results
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Tree session {session_id} not found")

        session = self.active_sessions[session_id]

        logger.info(f"Executing step-level batch for session {session_id}: {len(agent_configs)} agents, {max_steps} max steps")

        start_time = time.time()
        execution_id = str(uuid.uuid4())

        try:
            # Create coordination session
            coord_session = await self.step_coordinator.create_coordination_session(
                session_id=f"{session_id}_step_coord",
                agent_configs=agent_configs,
                coordination_config={
                    'strategy': coordination_strategy or CoordinationStrategy.FLEXIBLE_SYNC,
                    'max_steps': max_steps
                }
            )

            # Prepare initial inputs for each agent
            agent_inputs = {}
            for i, agent_config in enumerate(agent_configs):
                agent_id = agent_config.get('agent_id', f"{session_id}_step_agent_{i}")

                # Use tree memory content as initial input
                if session.tree_memory.steps:
                    last_step = session.tree_memory.steps[-1]
                    if hasattr(last_step, 'task'):
                        agent_inputs[agent_id] = last_step.task
                    else:
                        agent_inputs[agent_id] = str(last_step)
                else:
                    agent_inputs[agent_id] = "Continue the problem solving process."

            # Execute coordinated steps
            step_results = await self.step_coordinator.execute_coordinated_steps(
                session_id=f"{session_id}_step_coord",
                agent_inputs=agent_inputs,
                max_steps=max_steps
            )

            # Create execution result
            execution_time = time.time() - start_time

            result = TreeExecutionResult(
                session_id=session_id,
                execution_id=execution_id,
                success=True,
                step_coordination_results=step_results,
                step_level_metrics=self._calculate_step_level_metrics(step_results),
                tree_memory=session.tree_memory,
                total_execution_time=execution_time,
                batch_efficiency=self._calculate_step_batch_efficiency(step_results)
            )

            # Update global metrics
            self.global_metrics['step_level_executions'] += 1
            self.global_metrics['total_execution_time'] += execution_time

            logger.info(f"Step-level batch completed for session {session_id}: {len(step_results)} agents in {execution_time:.2f}s")

            # Cleanup coordination session
            await self.step_coordinator.shutdown_session(f"{session_id}_step_coord")

            return result

        except Exception as e:
            error_msg = f"Step-level batch failed: {e}"
            logger.error(error_msg)
            return TreeExecutionResult(
                session_id=session_id,
                execution_id=execution_id,
                success=False,
                error_message=error_msg,
                total_execution_time=time.time() - start_time
            )

    async def execute_hybrid_batch(
        self,
        session_id: str,
        agent_level_branches: int = 3,
        step_level_agents: int = 5,
        max_steps: int = 10,
        hybrid_config: Optional[Dict[str, Any]] = None
    ) -> TreeExecutionResult:
        """
        Execute hybrid batch processing combining agent-level and step-level modes.

        This creates multiple agent-level branches, each coordinated with step-level
        batch processing for maximum efficiency in Tree GRPO training.

        Args:
            session_id: ID of the tree session
            agent_level_branches: Number of agent-level branches
            step_level_agents: Number of agents per step-level coordination
            max_steps: Maximum steps per coordination
            hybrid_config: Configuration for hybrid execution

        Returns:
            TreeExecutionResult with both agent and step level results
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Tree session {session_id} not found")

        logger.info(f"Executing hybrid batch for session {session_id}: {agent_level_branches} agent branches, {step_level_agents} step agents")

        start_time = time.time()
        execution_id = str(uuid.uuid4())

        try:
            # Execute agent-level branches first
            agent_result = await self.execute_agent_level_batch(
                session_id=session_id,
                num_branches=agent_level_branches
            )

            if not agent_result.success:
                return agent_result  # Return agent-level failure

            # For each successful agent branch, execute step-level coordination
            step_results = {}
            for branch_id, branch_result in agent_result.agent_branches.items():
                if branch_result.get('success', False):
                    # Create step-level agent configs based on branch result
                    step_agent_configs = []
                    for i in range(step_level_agents):
                        step_agent_configs.append({
                            'agent_id': f"{session_id}_{branch_id}_step_{i}",
                            'base_branch': branch_id,
                            'config': hybrid_config.get('step_config', {}) if hybrid_config else {}
                        })

                    # Execute step-level coordination for this branch
                    step_result = await self.execute_step_level_batch(
                        session_id=f"{session_id}_{branch_id}",
                        agent_configs=step_agent_configs,
                        max_steps=max_steps
                    )

                    step_results[branch_id] = step_result

            # Combine results
            execution_time = time.time() - start_time

            combined_result = TreeExecutionResult(
                session_id=session_id,
                execution_id=execution_id,
                success=True,
                agent_branches=agent_result.agent_branches,
                agent_level_metrics=agent_result.agent_level_metrics,
                step_coordination_results={},  # Will be populated from step_results
                step_level_metrics={},  # Will be populated from step_results
                tree_memory=agent_result.tree_memory,
                tree_branches=agent_result.tree_branches,
                total_execution_time=execution_time,
                batch_efficiency=self._calculate_hybrid_efficiency(agent_result, step_results)
            )

            # Merge step results
            for branch_id, step_result in step_results.items():
                if step_result.success:
                    combined_result.step_coordination_results.update(step_result.step_coordination_results)
                    combined_result.step_level_metrics[branch_id] = step_result.step_level_metrics

            # Update global metrics
            self.global_metrics['hybrid_executions'] += 1
            self.global_metrics['total_execution_time'] += execution_time

            logger.info(f"Hybrid batch completed for session {session_id}: {len(step_results)} branch coordinations in {execution_time:.2f}s")
            return combined_result

        except Exception as e:
            error_msg = f"Hybrid batch failed: {e}"
            logger.error(error_msg)
            return TreeExecutionResult(
                session_id=session_id,
                execution_id=execution_id,
                success=False,
                error_message=error_msg,
                total_execution_time=time.time() - start_time
            )

    async def _execute_agent_branch(
        self,
        agent: AsyncCodeAgent,
        branch_memory: TreeMemory,
        branch_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single agent branch."""
        try:
            # Extract task from memory
            if branch_memory.steps:
                last_step = branch_memory.steps[-1]
                if hasattr(last_step, 'task'):
                    initial_input = last_step.task
                else:
                    initial_input = str(last_step)
            else:
                initial_input = "Continue the problem solving process."

            # Execute agent sequence
            max_steps = branch_config.get('max_steps', 5)
            results = await agent.execute_sequence_async(
                initial_input=initial_input,
                max_steps=max_steps
            )

            return {
                'success': True,
                'agent_id': agent.agent_id,
                'results': results,
                'branch_metadata': branch_memory.branch_metadata,
                'final_response': results[-1].response if results else None
            }

        except Exception as e:
            logger.error(f"Agent branch execution failed for {agent.agent_id}: {e}")
            return {
                'success': False,
                'agent_id': agent.agent_id,
                'error': str(e),
                'branch_metadata': branch_memory.branch_metadata
            }

    def _calculate_agent_level_metrics(self, branch_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for agent-level batch execution."""
        total_branches = len(branch_results)
        successful_branches = sum(1 for r in branch_results.values() if r.get('success', False))

        return {
            'total_branches': total_branches,
            'successful_branches': successful_branches,
            'success_rate': successful_branches / total_branches if total_branches > 0 else 0,
            'parallel_efficiency': 1.0  # All branches executed in parallel
        }

    def _calculate_step_level_metrics(self, step_results: Dict[str, List[StepExecutionResult]]) -> Dict[str, Any]:
        """Calculate metrics for step-level coordination."""
        total_agents = len(step_results)
        total_steps = sum(len(results) for results in step_results.values())
        successful_steps = sum(
            sum(1 for r in results if r.success) for results in step_results.values()
        )
        batch_steps = sum(
            sum(1 for r in results if r.batch_info) for results in step_results.values()
        )

        return {
            'total_agents': total_agents,
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'success_rate': successful_steps / total_steps if total_steps > 0 else 0,
            'batch_steps': batch_steps,
            'batch_ratio': batch_steps / total_steps if total_steps > 0 else 0
        }

    def _calculate_batch_efficiency(self, branch_results: Dict[str, Any]) -> float:
        """Calculate batch efficiency for agent-level execution."""
        # Agent-level batching is inherently efficient (all parallel)
        successful = sum(1 for r in branch_results.values() if r.get('success', False))
        total = len(branch_results)
        return successful / total if total > 0 else 0.0

    def _calculate_step_batch_efficiency(self, step_results: Dict[str, List[StepExecutionResult]]) -> float:
        """Calculate batch efficiency for step-level coordination."""
        total_steps = sum(len(results) for results in step_results.values())
        batch_steps = sum(
            sum(1 for r in results if r.batch_info) for results in step_results.values()
        )
        return batch_steps / total_steps if total_steps > 0 else 0.0

    def _calculate_hybrid_efficiency(
        self,
        agent_result: TreeExecutionResult,
        step_results: Dict[str, TreeExecutionResult]
    ) -> float:
        """Calculate overall efficiency for hybrid execution."""
        agent_efficiency = agent_result.batch_efficiency

        step_efficiencies = [
            result.batch_efficiency for result in step_results.values() if result.success
        ]
        avg_step_efficiency = sum(step_efficiencies) / len(step_efficiencies) if step_efficiencies else 0.0

        # Weighted average of agent and step efficiencies
        return (agent_efficiency + avg_step_efficiency) / 2.0

    async def shutdown_session(self, session_id: str) -> bool:
        """Shutdown a tree session and clean up resources."""
        if session_id not in self.active_sessions:
            return False

        logger.info(f"Shutting down tree session: {session_id}")

        # Remove from active sessions
        del self.active_sessions[session_id]

        # Clean up tree memory
        if session_id in self.tree_memories:
            del self.tree_memories[session_id]

        logger.info(f"Tree session shutdown complete: {session_id}")
        return True

    async def shutdown_all(self):
        """Shutdown all tree sessions and clean up resources."""
        logger.info(f"Shutting down AsyncTreeOrchestrator: {len(self.active_sessions)} sessions")

        # Shutdown all sessions
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            await self.shutdown_session(session_id)

        # Shutdown coordinators
        await self.step_coordinator.shutdown_all()
        await self.barrier_manager.shutdown_all()

        self.is_active = False
        logger.info("AsyncTreeOrchestrator shutdown complete")

    def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive orchestration performance metrics."""
        coordinator_metrics = self.step_coordinator.get_coordination_metrics()

        # Calculate overall efficiency
        total_executions = (
            self.global_metrics['agent_level_executions'] +
            self.global_metrics['step_level_executions'] +
            self.global_metrics['hybrid_executions']
        )

        avg_execution_time = (
            self.global_metrics['total_execution_time'] / total_executions
            if total_executions > 0 else 0.0
        )

        return {
            'global_metrics': self.global_metrics,
            'coordinator_metrics': coordinator_metrics,
            'active_sessions': len(self.active_sessions),
            'total_tree_memories': len(self.tree_memories),
            'average_execution_time': avg_execution_time,
            'config': {
                'batch_mode': self.config.batch_mode.value,
                'orchestration_strategy': self.config.orchestration_strategy.value,
                'agent_level_proposals': self.config.agent_level_proposals,
                'max_agents_per_step_batch': self.config.max_agents_per_step_batch,
                'max_tree_depth': self.config.max_tree_depth
            }
        }


@dataclass
class TreeSession:
    """Represents a tree execution session with both batch modes."""
    session_id: str
    tree_memory: TreeMemory
    coordinator_agent: Optional[CodeAgent]
    orchestrator: AsyncTreeOrchestrator
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status."""
        tree_summary = self.tree_memory.get_tree_summary()

        return {
            'session_id': self.session_id,
            'coordinator_agent': self.coordinator_agent.name if self.coordinator_agent else None,
            'tree_structure': tree_summary,
            'memory_steps': len(self.tree_memory.steps),
            'created_at': self.created_at,
            'config': self.config
        }