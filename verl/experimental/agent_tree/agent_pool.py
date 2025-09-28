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

import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .base_agent import BaseTreeAgent, AgentRole
from .tree_structures import TreeNode, SearchContext

logger = logging.getLogger(__name__)


class AgentPool:
    """
    Manages a collection of tree search agents and handles agent selection,
    load balancing, and coordination.
    """

    def __init__(self):
        """Initialize an empty agent pool."""
        self._agents: Dict[str, BaseTreeAgent] = {}
        self._agents_by_role: Dict[AgentRole, List[BaseTreeAgent]] = defaultdict(list)
        self._agents_by_specialization: Dict[str, List[BaseTreeAgent]] = defaultdict(list)

        # Load balancing
        self._agent_workload: Dict[str, int] = defaultdict(int)
        self._agent_performance: Dict[str, float] = defaultdict(float)

        # Dynamic selection weights
        self._role_weights: Dict[AgentRole, float] = {
            AgentRole.EXPLORER: 1.0,
            AgentRole.EVALUATOR: 1.0,
            AgentRole.COORDINATOR: 0.5,  # Fewer coordinators needed
            AgentRole.SYNTHESIZER: 0.3,
            AgentRole.VALIDATOR: 0.7,
            AgentRole.CRITIC: 0.6,
            AgentRole.CREATIVE: 0.8
        }

    def register_agent(self, agent: BaseTreeAgent, weight: float = 1.0) -> None:
        """
        Register a new agent in the pool.

        Args:
            agent: The agent to register
            weight: Weight for selection probability (higher = more likely to be selected)
        """
        if agent.name in self._agents:
            logger.warning(f"Agent {agent.name} is already registered. Overwriting.")

        self._agents[agent.name] = agent
        self._agents_by_role[agent.role].append(agent)

        if agent.specialization:
            self._agents_by_specialization[agent.specialization].append(agent)

        self._agent_performance[agent.name] = weight

        logger.info(f"Registered agent: {agent.name} (role: {agent.role.value}, specialization: {agent.specialization})")

    def unregister_agent(self, agent_name: str) -> bool:
        """
        Remove an agent from the pool.

        Args:
            agent_name: Name of the agent to remove

        Returns:
            True if agent was found and removed, False otherwise
        """
        if agent_name not in self._agents:
            return False

        agent = self._agents[agent_name]

        # Remove from all indexes
        del self._agents[agent_name]
        self._agents_by_role[agent.role].remove(agent)

        if agent.specialization:
            self._agents_by_specialization[agent.specialization].remove(agent)

        if agent_name in self._agent_workload:
            del self._agent_workload[agent_name]
        if agent_name in self._agent_performance:
            del self._agent_performance[agent_name]

        logger.info(f"Unregistered agent: {agent_name}")
        return True

    def get_agents_by_role(self, role: AgentRole) -> List[BaseTreeAgent]:
        """Get all agents with the specified role."""
        return self._agents_by_role[role].copy()

    def get_agents_by_specialization(self, specialization: str) -> List[BaseTreeAgent]:
        """Get all agents with the specified specialization."""
        return self._agents_by_specialization[specialization].copy()

    def get_agent_by_name(self, name: str) -> Optional[BaseTreeAgent]:
        """Get a specific agent by name."""
        return self._agents.get(name)

    def select_agents_for_exploration(
        self,
        nodes: List[TreeNode],
        context: SearchContext,
        max_agents: int = 3
    ) -> Dict[TreeNode, BaseTreeAgent]:
        """
        Select the best agents for exploring the given nodes.

        Args:
            nodes: Nodes that need exploration
            context: Current search context
            max_agents: Maximum number of agents to assign

        Returns:
            Mapping from nodes to assigned agents
        """
        explorer_agents = self.get_agents_by_role(AgentRole.EXPLORER)
        if not explorer_agents:
            logger.warning("No explorer agents available")
            return {}

        # Sort agents by performance and inverse workload
        available_agents = [
            agent for agent in explorer_agents
            if self._agent_workload[agent.name] < 5  # Avoid overloaded agents
        ]

        if not available_agents:
            available_agents = explorer_agents  # Use all if none are available

        # Score agents based on performance, workload, and specialization match
        def score_agent_for_node(agent: BaseTreeAgent, node: TreeNode) -> float:
            base_score = self._agent_performance[agent.name]
            workload_penalty = self._agent_workload[agent.name] * 0.1

            # Bonus for specialization match
            specialization_bonus = 0.0
            if agent.specialization and context.problem_type:
                if agent.specialization.lower() in context.problem_type.lower():
                    specialization_bonus = 0.3

            return base_score - workload_penalty + specialization_bonus

        # Assign agents to nodes
        assignments = {}
        used_agents = set()

        for node in nodes:
            # Find best available agent for this node
            candidates = [
                (agent, score_agent_for_node(agent, node))
                for agent in available_agents
                if agent.name not in used_agents
            ]

            if not candidates:
                # Reuse agents if necessary
                candidates = [
                    (agent, score_agent_for_node(agent, node))
                    for agent in available_agents
                ]

            if candidates:
                best_agent = max(candidates, key=lambda x: x[1])[0]
                assignments[node] = best_agent
                used_agents.add(best_agent.name)
                self._agent_workload[best_agent.name] += 1

            if len(assignments) >= max_agents:
                break

        return assignments

    def select_agents_for_evaluation(
        self,
        nodes: List[TreeNode],
        context: SearchContext
    ) -> List[BaseTreeAgent]:
        """
        Select evaluator agents for the given nodes.

        Args:
            nodes: Nodes that need evaluation
            context: Current search context

        Returns:
            List of selected evaluator agents
        """
        evaluator_agents = self.get_agents_by_role(AgentRole.EVALUATOR)

        if not evaluator_agents:
            logger.warning("No evaluator agents available")
            return []

        # Use all evaluators for comprehensive evaluation
        return evaluator_agents

    def select_coordinator_agent(self, context: SearchContext) -> Optional[BaseTreeAgent]:
        """
        Select a coordinator agent for the current search state.

        Args:
            context: Current search context

        Returns:
            Selected coordinator agent, or None if none available
        """
        coordinator_agents = self.get_agents_by_role(AgentRole.COORDINATOR)

        if not coordinator_agents:
            return None

        # Select coordinator with lowest workload
        return min(
            coordinator_agents,
            key=lambda agent: self._agent_workload[agent.name]
        )

    def select_synthesizer_agent(self, context: SearchContext) -> Optional[BaseTreeAgent]:
        """
        Select a synthesizer agent for result aggregation.

        Args:
            context: Current search context

        Returns:
            Selected synthesizer agent, or None if none available
        """
        synthesizer_agents = self.get_agents_by_role(AgentRole.SYNTHESIZER)

        if not synthesizer_agents:
            return None

        # Select synthesizer with best performance
        return max(
            synthesizer_agents,
            key=lambda agent: self._agent_performance[agent.name]
        )

    def get_diverse_agents(self, num_agents: int = 3) -> List[BaseTreeAgent]:
        """
        Get a diverse set of agents across different roles and specializations.

        Args:
            num_agents: Number of agents to select

        Returns:
            List of diverse agents
        """
        selected_agents = []
        used_roles = set()
        used_specializations = set()

        # Prioritize different roles first
        all_agents = list(self._agents.values())
        random.shuffle(all_agents)

        for agent in all_agents:
            if len(selected_agents) >= num_agents:
                break

            # Prefer agents with unused roles or specializations
            role_diversity = agent.role not in used_roles
            spec_diversity = agent.specialization not in used_specializations

            if role_diversity or spec_diversity or len(selected_agents) < num_agents // 2:
                selected_agents.append(agent)
                used_roles.add(agent.role)
                if agent.specialization:
                    used_specializations.add(agent.specialization)

        return selected_agents

    def update_agent_performance(self, agent_name: str, success: bool, execution_time: float = 0.0):
        """
        Update an agent's performance metrics.

        Args:
            agent_name: Name of the agent
            success: Whether the agent's task was successful
            execution_time: Time taken to complete the task
        """
        if agent_name not in self._agents:
            return

        agent = self._agents[agent_name]
        agent.update_performance_metrics(success)

        # Update pool-level performance tracking
        current_perf = self._agent_performance[agent_name]
        alpha = 0.1  # Learning rate for exponential moving average

        new_score = 1.0 if success else 0.0
        # Penalize very slow execution
        if execution_time > 30.0:  # 30 seconds threshold
            new_score *= 0.8

        self._agent_performance[agent_name] = alpha * new_score + (1 - alpha) * current_perf

        # Reduce workload (task completed)
        if self._agent_workload[agent_name] > 0:
            self._agent_workload[agent_name] -= 1

    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get statistics about the agent pool."""
        stats = {
            "total_agents": len(self._agents),
            "agents_by_role": {role.value: len(agents) for role, agents in self._agents_by_role.items()},
            "agents_by_specialization": {spec: len(agents) for spec, agents in self._agents_by_specialization.items()},
            "average_performance": sum(self._agent_performance.values()) / len(self._agent_performance) if self._agent_performance else 0.0,
            "total_workload": sum(self._agent_workload.values()),
            "top_performers": sorted(
                [(name, perf) for name, perf in self._agent_performance.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
        return stats

    def reset_workload(self):
        """Reset all agent workload counters."""
        self._agent_workload.clear()

    def __len__(self) -> int:
        """Return the number of registered agents."""
        return len(self._agents)

    def __contains__(self, agent_name: str) -> bool:
        """Check if an agent is registered in the pool."""
        return agent_name in self._agents

    def __repr__(self) -> str:
        role_counts = {role.value: len(agents) for role, agents in self._agents_by_role.items() if agents}
        return f"AgentPool(total={len(self._agents)}, roles={role_counts})"