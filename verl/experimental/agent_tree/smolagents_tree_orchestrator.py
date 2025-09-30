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
Main orchestrator for smolagents + tree GRPO integration.

This module provides the main orchestrator class that coordinates between
smolagents managed agents and tree GRPO training, handling the complete
workflow from tree expansion to reward decomposition.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .tree_structures import TreeGRPONode, SearchContext
from .smolagents_integration import (
    SmolagentsTreeCoordinator,
    MultiProposalGenerator,
    ComboNodeBuilder,
    AgentOutput,
    SMOLAGENTS_AVAILABLE
)

if SMOLAGENTS_AVAILABLE:
    from smolagents import CodeAgent, ToolCallingAgent
    from smolagents.monitoring import LogLevel

logger = logging.getLogger(__name__)


@dataclass
class SmolagentsTreeConfig:
    """Configuration for smolagents tree orchestrator."""

    # Coordination strategy
    coordination_strategy: str = "hierarchical"

    # Proposal generation
    proposals_per_agent: int = 2
    enable_memory_isolation: bool = True

    # Output fusion
    fusion_strategy: str = "simple_concat"  # "simple_concat", "structured_combination"

    # Silent operation
    enable_silent_mode: bool = True

    # Coordination parameters
    max_coordination_attempts: int = 3
    coordination_timeout: float = 30.0


class SmolagentsTreeOrchestrator:
    """
    Main orchestrator for smolagents + tree GRPO integration.

    This class coordinates the complete workflow:
    1. Coordination planning via smolagents coordinator
    2. Multi-proposal generation from managed agents
    3. Cartesian product combination of agent outputs
    4. TreeGRPONode creation with proper credit assignment
    """

    def __init__(
        self,
        coordinator_agent: 'CodeAgent',
        config: Optional[SmolagentsTreeConfig] = None
    ):
        if not SMOLAGENTS_AVAILABLE:
            raise ImportError("smolagents is required but not available")

        self.config = config or SmolagentsTreeConfig()

        # Initialize core components
        self.coordinator = SmolagentsTreeCoordinator(
            smolagents_agent=coordinator_agent,
            coordination_strategy=self.config.coordination_strategy,
            enable_silent_mode=self.config.enable_silent_mode
        )

        self.combo_builder = ComboNodeBuilder(
            fusion_strategy=self.config.fusion_strategy
        )

        # Initialize proposal generators for managed agents
        self.proposal_generators = {}
        if hasattr(coordinator_agent, 'managed_agents'):
            for agent_name, agent in coordinator_agent.managed_agents.items():
                self.proposal_generators[agent_name] = MultiProposalGenerator(agent)

        # Metrics tracking
        self.expansion_history = []
        self.performance_metrics = {
            "total_expansions": 0,
            "successful_expansions": 0,
            "average_combo_nodes_per_expansion": 0.0,
            "coordination_success_rate": 0.0
        }

    def expand_tree_node(
        self,
        parent_node: TreeGRPONode,
        search_context: SearchContext
    ) -> List[TreeGRPONode]:
        """
        Expand a tree node using smolagents coordination and multi-agent generation.

        Args:
            parent_node: Node to expand from
            search_context: Current search state

        Returns:
            List of TreeGRPONode instances representing all agent output combinations
        """
        expansion_start_time = time.time()

        try:
            # Step 1: Coordination planning
            logger.info(f"Planning expansion for node {parent_node.node_id}")
            coordination_plan = self.coordinator.plan_expansion(parent_node, search_context)

            if not coordination_plan:
                logger.warning("No coordination plan generated - falling back to single agent")
                return self._fallback_single_agent_expansion(parent_node, search_context)

            # Step 2: Set up proposal generators with base memory
            parent_context = self._build_parent_context(parent_node)
            self._setup_proposal_generators(parent_context)

            # Step 3: Generate multiple proposals from each agent
            agent_outputs = self._generate_agent_proposals(coordination_plan)

            if not agent_outputs:
                logger.error("No agent outputs generated")
                return []

            # Step 4: Create combination nodes
            combo_nodes = self.combo_builder.create_combo_nodes(
                parent_node=parent_node,
                agent_outputs=agent_outputs,
                search_context=search_context
            )

            # Step 5: Post-process and validate nodes
            validated_nodes = self._post_process_nodes(combo_nodes, parent_node)

            # Step 6: Update metrics and history
            expansion_duration = time.time() - expansion_start_time
            self._update_expansion_metrics(
                coordination_plan=coordination_plan,
                agent_outputs=agent_outputs,
                combo_nodes=validated_nodes,
                duration=expansion_duration
            )

            logger.info(f"Successfully expanded node {parent_node.node_id} into {len(validated_nodes)} combinations")
            return validated_nodes

        except Exception as e:
            logger.error(f"Tree expansion failed: {e}")
            self.performance_metrics["total_expansions"] += 1
            return []

    def _build_parent_context(self, parent_node: TreeGRPONode) -> str:
        """Build context string from parent node."""
        context_parts = []

        # Add message history
        if parent_node.messages:
            for msg in parent_node.messages[-3:]:  # Last 3 messages for context
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                context_parts.append(f"{role}: {content}")

        # Add compressed context if available
        if hasattr(parent_node, 'compressed_context') and parent_node.compressed_context:
            context_parts.append(f"Context Summary: {parent_node.compressed_context}")

        # Add local generation text
        if hasattr(parent_node, 'local_generation_text') and parent_node.local_generation_text:
            context_parts.append(f"Previous Output: {parent_node.local_generation_text}")

        return "\n".join(context_parts)

    def _setup_proposal_generators(self, parent_context: str):
        """Set up all proposal generators with base memory state."""
        for generator in self.proposal_generators.values():
            generator.set_base_memory(parent_context)

    def _generate_agent_proposals(
        self,
        coordination_plan: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[AgentOutput]]:
        """Generate proposals from all agents according to coordination plan."""
        agent_outputs = {}

        for agent_name, plan in coordination_plan.items():
            if agent_name not in self.proposal_generators:
                logger.warning(f"No proposal generator for agent {agent_name}")
                continue

            try:
                generator = self.proposal_generators[agent_name]
                task = plan.get("task", "Analyze current context and provide solutions")
                proposals_needed = plan.get("proposals_needed", self.config.proposals_per_agent)

                # Generate multiple proposals with memory isolation
                proposals = generator.generate_multiple_proposals(
                    task=task,
                    n=proposals_needed
                )

                agent_outputs[agent_name] = proposals
                logger.debug(f"Generated {len(proposals)} proposals from {agent_name}")

            except Exception as e:
                logger.error(f"Failed to generate proposals for {agent_name}: {e}")
                continue

        return agent_outputs

    def _post_process_nodes(
        self,
        combo_nodes: List[TreeGRPONode],
        parent_node: TreeGRPONode
    ) -> List[TreeGRPONode]:
        """Post-process combination nodes for validation and enhancement."""
        validated_nodes = []

        for node in combo_nodes:
            try:
                # Validate node structure
                if not self._validate_combo_node(node):
                    logger.warning(f"Invalid combo node {node.node_id}, skipping")
                    continue

                # Enhance with additional metadata
                self._enhance_combo_node(node, parent_node)

                validated_nodes.append(node)

            except Exception as e:
                logger.error(f"Error post-processing node {node.node_id}: {e}")
                continue

        return validated_nodes

    def _validate_combo_node(self, node: TreeGRPONode) -> bool:
        """Validate combination node structure and content."""
        # Check required attributes
        if not hasattr(node, 'agent_contributions') or not node.agent_contributions:
            return False

        # Check that we have meaningful content
        if not node.local_generation_text or len(node.local_generation_text.strip()) < 10:
            return False

        # Check agent contributions
        for agent_name, contribution in node.agent_contributions.items():
            if not contribution.output_text or len(contribution.output_text.strip()) < 5:
                return False

        return True

    def _enhance_combo_node(self, node: TreeGRPONode, parent_node: TreeGRPONode):
        """Enhance combo node with additional metadata."""
        # Set agent information
        contributing_agents = list(node.agent_contributions.keys())
        node.agent_name = f"combo[{','.join(contributing_agents)}]"

        # Calculate diversity metrics
        if hasattr(node, 'agent_contributions'):
            outputs = [contrib.output_text for contrib in node.agent_contributions.values()]
            node.diversity_score = self._calculate_output_diversity(outputs)

        # Set generation boundaries
        if node.local_generation_text:
            node.generation_length = len(node.local_generation_text.split())

    def _calculate_output_diversity(self, outputs: List[str]) -> float:
        """Calculate diversity score for agent outputs."""
        if len(outputs) < 2:
            return 0.0

        # Simple diversity metric based on text similarity
        total_pairs = 0
        diverse_pairs = 0

        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                total_pairs += 1
                # Simple word overlap check
                words_i = set(outputs[i].lower().split())
                words_j = set(outputs[j].lower().split())
                overlap = len(words_i & words_j) / max(len(words_i | words_j), 1)
                if overlap < 0.7:  # Less than 70% overlap considered diverse
                    diverse_pairs += 1

        return diverse_pairs / max(total_pairs, 1)

    def _fallback_single_agent_expansion(
        self,
        parent_node: TreeGRPONode,
        search_context: SearchContext
    ) -> List[TreeGRPONode]:
        """Fallback to single agent expansion when coordination fails."""
        logger.info("Using fallback single agent expansion")

        # Use first available agent
        if not self.proposal_generators:
            return []

        first_agent_name = list(self.proposal_generators.keys())[0]
        generator = self.proposal_generators[first_agent_name]

        # Set up memory
        parent_context = self._build_parent_context(parent_node)
        generator.set_base_memory(parent_context)

        # Generate proposals
        try:
            proposals = generator.generate_multiple_proposals(
                task="Continue the current task by providing alternative approaches",
                n=self.config.proposals_per_agent
            )

            # Create simple combo nodes (one agent only)
            agent_outputs = {first_agent_name: proposals}
            combo_nodes = self.combo_builder.create_combo_nodes(
                parent_node=parent_node,
                agent_outputs=agent_outputs,
                search_context=search_context
            )

            return combo_nodes

        except Exception as e:
            logger.error(f"Fallback expansion failed: {e}")
            return []

    def _update_expansion_metrics(
        self,
        coordination_plan: Dict[str, Dict[str, Any]],
        agent_outputs: Dict[str, List[AgentOutput]],
        combo_nodes: List[TreeGRPONode],
        duration: float
    ):
        """Update performance metrics."""
        self.performance_metrics["total_expansions"] += 1

        if combo_nodes:
            self.performance_metrics["successful_expansions"] += 1

        # Update average combo nodes per expansion
        current_avg = self.performance_metrics["average_combo_nodes_per_expansion"]
        total_expansions = self.performance_metrics["total_expansions"]
        new_avg = ((current_avg * (total_expansions - 1)) + len(combo_nodes)) / total_expansions
        self.performance_metrics["average_combo_nodes_per_expansion"] = new_avg

        # Update coordination success rate
        self.performance_metrics["coordination_success_rate"] = (
            self.performance_metrics["successful_expansions"] /
            max(self.performance_metrics["total_expansions"], 1)
        )

        # Store expansion record
        expansion_record = {
            "timestamp": time.time(),
            "coordination_plan": coordination_plan,
            "num_agent_outputs": {k: len(v) for k, v in agent_outputs.items()},
            "num_combo_nodes": len(combo_nodes),
            "duration": duration
        }
        self.expansion_history.append(expansion_record)

        # Keep only recent history
        if len(self.expansion_history) > 100:
            self.expansion_history.pop(0)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()

    def decompose_rewards_for_tree(
        self,
        tree_nodes: List[TreeGRPONode],
        final_rewards: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Decompose rewards for all combination nodes in the tree.

        Args:
            tree_nodes: List of all TreeGRPONodes in the tree
            final_rewards: Dictionary mapping node IDs to their final rewards

        Returns:
            Dictionary mapping node IDs to agent reward decompositions
        """
        node_agent_rewards = {}

        for node in tree_nodes:
            if node.node_id in final_rewards and hasattr(node, 'agent_contributions'):
                node_reward = final_rewards[node.node_id]
                agent_rewards = node.decompose_node_reward(node_reward)
                node_agent_rewards[node.node_id] = agent_rewards

        return node_agent_rewards

    def get_coordination_summary(self) -> Dict[str, Any]:
        """Get summary of coordination decisions and outcomes."""
        if not self.expansion_history:
            return {"message": "No expansion history available"}

        recent_expansions = self.expansion_history[-10:]  # Last 10 expansions

        summary = {
            "total_expansions": len(self.expansion_history),
            "recent_expansions": len(recent_expansions),
            "average_combo_nodes": sum(exp["num_combo_nodes"] for exp in recent_expansions) / len(recent_expansions),
            "average_duration": sum(exp["duration"] for exp in recent_expansions) / len(recent_expansions),
            "agent_usage": {}
        }

        # Analyze agent usage patterns
        agent_usage = {}
        for exp in recent_expansions:
            for agent_name, num_outputs in exp["num_agent_outputs"].items():
                if agent_name not in agent_usage:
                    agent_usage[agent_name] = {"total_calls": 0, "total_outputs": 0}
                agent_usage[agent_name]["total_calls"] += 1
                agent_usage[agent_name]["total_outputs"] += num_outputs

        summary["agent_usage"] = agent_usage
        return summary


logger.info("Smolagents tree orchestrator module loaded successfully")