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
Smolagents integration for tree GRPO.

This module provides integration between smolagents' managed agent architecture
and our tree GRPO training system, enabling powerful multi-agent coordination
while maintaining node-level training semantics.
"""

import copy
import itertools
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from .tree_structures import TreeGRPONode, SearchContext

# Import smolagents components
try:
    from smolagents import CodeAgent, ToolCallingAgent, MultiStepAgent
    from smolagents.monitoring import LogLevel
    from smolagents.memory import AgentMemory, TaskStep
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False
    # Create dummy classes for type hints
    class CodeAgent: pass
    class ToolCallingAgent: pass
    class MultiStepAgent: pass
    class LogLevel:
        OFF = -1

logger = logging.getLogger(__name__)


@dataclass
class AgentOutput:
    """Container for individual agent output."""
    agent_name: str
    output_text: str
    tokens: Optional[List[int]] = None
    logprobs: Optional[List[float]] = None
    response_mask: Optional[List[int]] = None  # Added: mask from verl's AgentLoopOutput
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CombinedOutput:
    """Container for fused multi-agent outputs."""
    combined_text: str
    agent_contributions: Dict[str, AgentOutput]
    tokens: Optional[List[int]] = None
    logprobs: Optional[List[float]] = None
    response_mask: Optional[List[int]] = None  # Added: combined mask for training
    fusion_strategy: str = "simple_concat"


class SmolagentsTreeCoordinator:
    """
    Wrapper around smolagents CodeAgent to act as tree coordinator/planner.

    This coordinator analyzes the current tree context and decides how to
    deploy managed agents for the next level of tree expansion.
    """

    def __init__(
        self,
        smolagents_agent: MultiStepAgent,
        coordination_strategy: str = "hierarchical",
        enable_silent_mode: bool = True
    ):
        if not SMOLAGENTS_AVAILABLE:
            raise ImportError("smolagents is required but not available")

        self.agent = smolagents_agent
        self.coordination_strategy = coordination_strategy

        # Enable silent mode for backend operation
        if enable_silent_mode and hasattr(self.agent, 'logger'):
            self.agent.logger.level = LogLevel.OFF

        # Cache for coordination decisions
        self.coordination_history = []

    def plan_expansion(
        self,
        parent_node: TreeGRPONode,
        search_context: SearchContext
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze current context and create coordination plan for managed agents.

        Args:
            parent_node: Current node to expand from
            search_context: Current search state and configuration

        Returns:
            Dictionary mapping agent names to their tasks and configurations
        """
        # Build context for coordination decision
        coordination_context = self._build_coordination_context(parent_node, search_context)

        # Get available managed agents
        available_agents = list(self.agent.managed_agents.keys()) if hasattr(self.agent, 'managed_agents') else []

        if not available_agents:
            logger.warning("No managed agents available for coordination")
            return {}

        # Create coordination prompt
        coordination_prompt = f"""
        Analyze the current context and create a coordination plan for the available agents.

        Current Context:
        {coordination_context}

        Available Agents: {', '.join(available_agents)}

        Create a plan that:
        1. Assigns specific tasks to different agents based on their capabilities
        2. Specifies how many proposal variants each agent should generate (typically 2)
        3. Ensures tasks are complementary and will benefit from combination

        Respond in this format:
        AGENT_NAME: task_description | proposals_needed: N
        """

        try:
            # Use the coordinator to generate plan
            coordination_result = self.agent(coordination_prompt)

            # Parse coordination result into structured plan
            coordination_plan = self._parse_coordination_result(coordination_result, available_agents)

            # Cache this decision for learning
            self.coordination_history.append({
                "context": coordination_context,
                "plan": coordination_plan,
                "timestamp": time.time()
            })

            return coordination_plan

        except Exception as e:
            logger.error(f"Coordination planning failed: {e}")
            # Fallback: assign all agents to analyze current context
            return self._create_fallback_plan(available_agents)

    def _build_coordination_context(self, parent_node: TreeGRPONode, search_context: SearchContext) -> str:
        """Build context string for coordination decision."""
        context_parts = []

        # Current task/problem context
        if parent_node.messages:
            latest_message = parent_node.messages[-1]
            context_parts.append(f"Current Task: {latest_message.get('content', '')}")

        # Tree search progress
        context_parts.append(f"Search Progress: depth {search_context.current_depth}/{search_context.max_depth}")
        context_parts.append(f"Total nodes explored: {search_context.total_nodes_generated}")

        # Previous agent contributions (if any)
        if hasattr(parent_node, 'agent_contributions') and parent_node.agent_contributions:
            context_parts.append("Previous Agent Contributions:")
            for agent_name, contribution in parent_node.agent_contributions.items():
                context_parts.append(f"- {agent_name}: {contribution.output_text[:100]}...")

        # Quality trends
        if search_context.best_score > 0:
            context_parts.append(f"Best score so far: {search_context.best_score:.3f}")

        return "\n".join(context_parts)

    def _parse_coordination_result(self, result: str, available_agents: List[str]) -> Dict[str, Dict[str, Any]]:
        """Parse coordination result into structured plan."""
        plan = {}
        lines = result.strip().split('\n')

        for line in lines:
            line = line.strip()
            if ':' in line and any(agent in line for agent in available_agents):
                # Try to extract agent name and task
                for agent_name in available_agents:
                    if agent_name in line:
                        parts = line.split(':', 1)
                        if len(parts) >= 2:
                            task_part = parts[1].strip()

                            # Extract proposals_needed if specified
                            proposals_needed = 2  # default
                            if 'proposals_needed:' in task_part:
                                task_parts = task_part.split('|')
                                task_desc = task_parts[0].strip()
                                for part in task_parts[1:]:
                                    if 'proposals_needed:' in part:
                                        try:
                                            proposals_needed = int(part.split(':')[1].strip())
                                        except:
                                            pass
                            else:
                                task_desc = task_part

                            plan[agent_name] = {
                                "task": task_desc,
                                "proposals_needed": proposals_needed
                            }
                        break

        # Ensure all available agents have tasks if parsing failed partially
        for agent_name in available_agents:
            if agent_name not in plan:
                plan[agent_name] = {
                    "task": "Analyze the current context and provide alternative approaches",
                    "proposals_needed": 2
                }

        return plan

    def _create_fallback_plan(self, available_agents: List[str]) -> Dict[str, Dict[str, Any]]:
        """Create fallback coordination plan."""
        return {
            agent_name: {
                "task": f"As {agent_name}, analyze current context and provide alternative approaches",
                "proposals_needed": 2
            }
            for agent_name in available_agents
        }

    def update_coordination_policy(self, parent_context: str, coordination_decision: Dict, outcomes: List[TreeGRPONode]):
        """
        Update coordination policy based on outcomes.
        This can be used for meta-learning about coordination effectiveness.
        """
        # Calculate coordination quality metrics
        if outcomes:
            avg_quality = sum(node.quality_score for node in outcomes if hasattr(node, 'quality_score') and node.quality_score) / len(outcomes)
            diversity = len(set(str(node.agent_contributions) for node in outcomes if hasattr(node, 'agent_contributions')))

            coordination_quality = {
                "average_quality": avg_quality,
                "diversity_score": diversity,
                "num_outcomes": len(outcomes)
            }

            # Store for potential policy updates
            self.coordination_history[-1]["outcomes"] = coordination_quality

            logger.debug(f"Coordination quality: {coordination_quality}")


class MultiProposalGenerator:
    """
    Generator for multiple isolated proposals from a single managed agent.

    Implements memory isolation to ensure each proposal is generated independently
    without cross-contamination between different attempts.
    """

    def __init__(self, smolagents_agent: MultiStepAgent):
        if not SMOLAGENTS_AVAILABLE:
            raise ImportError("smolagents is required but not available")

        self.agent = smolagents_agent
        self.base_memory_state = None

        # Enable silent mode for backend operation
        if hasattr(self.agent, 'logger'):
            self.agent.logger.level = LogLevel.OFF

    def set_base_memory(self, parent_context: str):
        """Set the baseline memory state for isolated proposal generation."""
        # Reset agent memory
        self.agent.memory.reset()

        # Add parent context as initial task
        if parent_context:
            initial_task = TaskStep(task=parent_context)
            self.agent.memory.steps.append(initial_task)

        # Save this state as baseline
        self.base_memory_state = copy.deepcopy(self.agent.memory.steps)

    def generate_isolated_proposal(self, task: str, variant_id: int = 0) -> AgentOutput:
        """Generate a single proposal with isolated memory state."""
        # Restore base memory state
        self.agent.memory.steps = copy.deepcopy(self.base_memory_state)

        # Add slight variation to encourage diversity
        if variant_id > 0:
            varied_task = f"{task}\n\n(Alternative approach {variant_id + 1})"
        else:
            varied_task = task

        try:
            # Generate proposal
            result = self.agent(varied_task)

            # Extract tokens, logprobs, and mask if available
            tokens, logprobs, response_mask = self._extract_generation_data()

            return AgentOutput(
                agent_name=self.agent.name or "unknown_agent",
                output_text=str(result),
                tokens=tokens,
                logprobs=logprobs,
                response_mask=response_mask,  # Added: include mask
                metadata={
                    "variant_id": variant_id,
                    "task": varied_task,
                    "generation_time": time.time()
                }
            )

        except Exception as e:
            logger.error(f"Failed to generate proposal for {self.agent.name}: {e}")
            return AgentOutput(
                agent_name=self.agent.name or "unknown_agent",
                output_text=f"Error: {str(e)}",
                response_mask=[],  # Empty mask for error case
                metadata={"error": True, "variant_id": variant_id}
            )

    def generate_multiple_proposals(self, task: str, n: int = 2) -> List[AgentOutput]:
        """Generate multiple isolated proposals."""
        proposals = []

        for i in range(n):
            proposal = self.generate_isolated_proposal(task, variant_id=i)
            proposals.append(proposal)

            # Optional: add small delay for additional randomness
            if i < n - 1:
                time.sleep(0.1)

        # Restore base memory state
        if self.base_memory_state:
            self.agent.memory.steps = copy.deepcopy(self.base_memory_state)

        return proposals

    def _extract_generation_data(self) -> Tuple[Optional[List[int]], Optional[List[float]], Optional[List[int]]]:
        """Extract tokens, logprobs, and mask from recent generation if available."""
        try:
            # Check if the agent's memory has recent steps with generation data
            if self.agent.memory.steps:
                recent_step = self.agent.memory.steps[-1]
                if hasattr(recent_step, 'model_output_message') and recent_step.model_output_message:
                    # Try to extract tokens/logprobs/mask from model output
                    # This is agent-specific and may need customization

                    # For now, we'll need to get this from the agent's output
                    # when integrated with verl's AgentLoopOutput
                    pass

            # Note: In actual integration, this will be populated from
            # AgentLoopOutput.response_ids, response_logprobs, response_mask
            return None, None, None
        except Exception:
            return None, None, None


class ComboNodeBuilder:
    """
    Builder for creating TreeGRPONode instances from combinations of agent outputs.

    Handles the fusion of multiple agent outputs into a single coherent response
    and creates the corresponding tree node with proper training data.
    """

    def __init__(self, fusion_strategy: str = "simple_concat"):
        self.fusion_strategy = fusion_strategy

    def create_combo_nodes(
        self,
        parent_node: TreeGRPONode,
        agent_outputs: Dict[str, List[AgentOutput]],
        search_context: SearchContext
    ) -> List[TreeGRPONode]:
        """
        Create combination nodes from cartesian product of agent outputs.

        Args:
            parent_node: Parent node to expand from
            agent_outputs: Dictionary mapping agent names to their outputs
            search_context: Current search context

        Returns:
            List of TreeGRPONode instances representing all combinations
        """
        combo_nodes = []

        # Generate cartesian product of agent outputs
        agent_names = list(agent_outputs.keys())
        output_combinations = list(itertools.product(*agent_outputs.values()))

        for combo_idx, combo in enumerate(output_combinations):
            # Create agent contributions mapping
            agent_contributions = dict(zip(agent_names, combo))

            # Fuse agent outputs into combined output
            combined_output = self._fuse_agent_outputs(combo, agent_contributions)

            # Create TreeGRPONode from combined output
            combo_node = self._create_tree_node_from_combined_output(
                parent_node=parent_node,
                combined_output=combined_output,
                combo_idx=combo_idx,
                search_context=search_context
            )

            combo_nodes.append(combo_node)

        logger.info(f"Created {len(combo_nodes)} combination nodes from {len(agent_names)} agents")
        return combo_nodes

    def _fuse_agent_outputs(
        self,
        outputs: Tuple[AgentOutput, ...],
        agent_contributions: Dict[str, AgentOutput]
    ) -> CombinedOutput:
        """Fuse multiple agent outputs into a single combined output."""

        if self.fusion_strategy == "simple_concat":
            return self._simple_concatenation_fusion(outputs, agent_contributions)
        elif self.fusion_strategy == "structured_combination":
            return self._structured_combination_fusion(outputs, agent_contributions)
        else:
            # Fallback to simple concatenation
            return self._simple_concatenation_fusion(outputs, agent_contributions)

    def _simple_concatenation_fusion(
        self,
        outputs: Tuple[AgentOutput, ...],
        agent_contributions: Dict[str, AgentOutput]
    ) -> CombinedOutput:
        """Simple concatenation of agent outputs."""
        combined_text_parts = []

        for agent_name, output in agent_contributions.items():
            combined_text_parts.append(f"[{agent_name}]: {output.output_text}")

        combined_text = "\n\n".join(combined_text_parts)

        return CombinedOutput(
            combined_text=combined_text,
            agent_contributions=agent_contributions,
            fusion_strategy="simple_concat"
        )

    def _structured_combination_fusion(
        self,
        outputs: Tuple[AgentOutput, ...],
        agent_contributions: Dict[str, AgentOutput]
    ) -> CombinedOutput:
        """Structured combination with clear sections."""
        sections = []

        for agent_name, output in agent_contributions.items():
            section = f"""
## {agent_name.replace('_', ' ').title()} Contribution

{output.output_text}
"""
            sections.append(section)

        combined_text = "\n".join(sections)

        return CombinedOutput(
            combined_text=combined_text,
            agent_contributions=agent_contributions,
            fusion_strategy="structured_combination"
        )

    def _create_tree_node_from_combined_output(
        self,
        parent_node: TreeGRPONode,
        combined_output: CombinedOutput,
        combo_idx: int,
        search_context: SearchContext
    ) -> TreeGRPONode:
        """Create TreeGRPONode from combined agent output."""

        # Create new messages by extending parent messages
        new_messages = parent_node.messages.copy()
        new_messages.append({
            "role": "assistant",
            "content": combined_output.combined_text
        })

        # Create TreeGRPONode with combination metadata
        combo_node = TreeGRPONode(
            parent_id=parent_node.node_id,
            depth=parent_node.depth + 1,
            messages=new_messages,
            agent_name=f"combo_{combo_idx}",
            branch_reason=f"agent_combination_{combo_idx}",

            # Store agent contributions for credit assignment
            agent_contributions=combined_output.agent_contributions,

            # Initialize with combined text
            local_generation_text=combined_output.combined_text,

            # Node generation data (will be populated if tokens/logprobs available)
            node_generation_tokens=combined_output.tokens or [],
            node_generation_logprobs=combined_output.logprobs or [],
            node_generation_mask=combined_output.response_mask or [],

            # Combination metadata
            fusion_strategy=combined_output.fusion_strategy,
            combo_index=combo_idx
        )

        # Register node in search context
        search_context.register_node(combo_node)

        return combo_node


# Add agent_contributions field to TreeGRPONode if not already present
def extend_tree_grpo_node():
    """Extend TreeGRPONode with combination-specific fields."""

    # Add new fields to TreeGRPONode if they don't exist
    if not hasattr(TreeGRPONode, 'agent_contributions'):
        TreeGRPONode.agent_contributions = None

    if not hasattr(TreeGRPONode, 'fusion_strategy'):
        TreeGRPONode.fusion_strategy = "simple_concat"

    if not hasattr(TreeGRPONode, 'combo_index'):
        TreeGRPONode.combo_index = 0

    # Add reward decomposition method
    def decompose_node_reward(self, node_reward: float) -> Dict[str, float]:
        """
        Decompose node reward to individual agent contributions.

        Args:
            node_reward: Total reward received by this combination node

        Returns:
            Dictionary mapping agent names to their allocated rewards
        """
        if not hasattr(self, 'agent_contributions') or not self.agent_contributions:
            return {}

        agent_rewards = {}
        num_agents = len(self.agent_contributions)

        # Simple equal distribution (can be made more sophisticated)
        equal_share = node_reward / num_agents

        for agent_name, contribution in self.agent_contributions.items():
            # For now, equal distribution
            # Could be enhanced with contribution quality weighting
            agent_rewards[agent_name] = equal_share

        return agent_rewards

    # Attach method to TreeGRPONode
    TreeGRPONode.decompose_node_reward = decompose_node_reward


# Initialize extensions
extend_tree_grpo_node()

logger.info("Smolagents integration module loaded successfully")