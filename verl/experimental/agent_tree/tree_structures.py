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

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from uuid import uuid4

import torch
import numpy as np
from pydantic import BaseModel

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput


@dataclass
class TreeNode:
    """
    Represents a node in the agent tree search.

    Each node contains the state of the conversation/reasoning at a particular point,
    along with metadata about its position in the tree and quality metrics.
    """

    # Core content
    node_id: str = field(default_factory=lambda: str(uuid4()))
    messages: List[Dict[str, Any]] = field(default_factory=list)
    prompt_ids: List[int] = field(default_factory=list)
    response_ids: List[int] = field(default_factory=list)
    response_logprobs: List[float] = field(default_factory=list)

    # Tree structure
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0

    # Generation metadata
    agent_name: str = ""
    branch_reason: str = ""  # Why this branch was created
    generation_method: str = ""  # How this branch was generated

    # Evaluation metrics
    cumulative_score: float = 0.0
    quality_score: float = 0.0
    uncertainty_score: float = 0.0
    diversity_score: float = 0.0

    # Search state
    is_terminal: bool = False
    is_pruned: bool = False
    visit_count: int = 0
    last_update_time: float = field(default_factory=time.time)

    # Multi-modal support
    image_data: Optional[Any] = None
    multi_modal_keys: Optional[List[str]] = None

    # Tool execution results
    tool_calls: List[Any] = field(default_factory=list)
    tool_results: List[Any] = field(default_factory=list)
    tool_rewards: List[float] = field(default_factory=list)

    # Candidate tracking (for beam search / multiple generation)
    generation_candidates: List[List[int]] = field(default_factory=list)
    selected_candidate_idx: int = 0

    @classmethod
    def from_prompt(cls, messages: List[Dict[str, Any]], **kwargs) -> "TreeNode":
        """Create a root node from initial prompt messages."""
        return cls(
            messages=messages,
            depth=0,
            branch_reason="root",
            **kwargs
        )

    @classmethod
    def from_agent_output(cls, parent: "TreeNode", agent_output: AgentLoopOutput, agent_name: str) -> "TreeNode":
        """Create a child node from an agent's output."""
        return cls(
            parent_id=parent.node_id,
            depth=parent.depth + 1,
            messages=parent.messages.copy(),  # Will be updated with new messages
            prompt_ids=agent_output.prompt_ids,
            response_ids=agent_output.response_ids,
            response_logprobs=agent_output.response_logprobs or [],
            agent_name=agent_name,
            cumulative_score=parent.cumulative_score
        )

    def add_child(self, child_id: str):
        """Add a child node ID to this node."""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

    def update_scores(self, quality: float, uncertainty: float = None, diversity: float = None):
        """Update the node's evaluation scores."""
        self.quality_score = quality
        if uncertainty is not None:
            self.uncertainty_score = uncertainty
        if diversity is not None:
            self.diversity_score = diversity

        # Update cumulative score with weighted combination
        self.cumulative_score = (
            0.6 * self.quality_score +
            0.2 * (1.0 - self.uncertainty_score) +  # Lower uncertainty is better
            0.2 * self.diversity_score
        )
        self.last_update_time = time.time()

    def get_path_from_root(self, node_registry: Dict[str, "TreeNode"]) -> List["TreeNode"]:
        """Get the path from root to this node."""
        path = []
        current = self
        while current is not None:
            path.append(current)
            if current.parent_id is None:
                break
            current = node_registry.get(current.parent_id)
        return list(reversed(path))

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "messages": self.messages,
            "depth": self.depth,
            "agent_name": self.agent_name,
            "cumulative_score": self.cumulative_score,
            "quality_score": self.quality_score,
            "uncertainty_score": self.uncertainty_score,
            "diversity_score": self.diversity_score,
            "is_terminal": self.is_terminal,
            "branch_reason": self.branch_reason
        }


@dataclass
class SearchContext:
    """
    Contains the current state and configuration of the tree search process.
    """

    # Search configuration
    max_depth: int = 5
    max_branches_per_node: int = 3
    max_active_nodes: int = 10
    search_timeout: float = 300.0  # seconds

    # Current search state
    current_depth: int = 0
    total_nodes_generated: int = 0
    search_start_time: float = field(default_factory=time.time)
    stuck_count: int = 0  # Consecutive iterations without improvement

    # Quality tracking
    best_score: float = 0.0
    average_score: float = 0.0
    diversity_score: float = 0.0
    quality_decline: bool = False

    # Problem context
    problem_type: str = "general"
    difficulty_level: float = 0.5
    required_tools: List[str] = field(default_factory=list)

    # Sampling parameters
    sampling_params: Dict[str, Any] = field(default_factory=dict)

    # Node registry for tracking all nodes
    node_registry: Dict[str, TreeNode] = field(default_factory=dict)

    def register_node(self, node: TreeNode):
        """Register a new node in the context."""
        self.node_registry[node.node_id] = node
        if node.parent_id:
            parent = self.node_registry.get(node.parent_id)
            if parent:
                parent.add_child(node.node_id)

    def get_active_nodes(self) -> List[TreeNode]:
        """Get all non-terminal, non-pruned nodes."""
        return [
            node for node in self.node_registry.values()
            if not node.is_terminal and not node.is_pruned
        ]

    def get_leaf_nodes(self) -> List[TreeNode]:
        """Get all leaf nodes (nodes with no children)."""
        return [
            node for node in self.node_registry.values()
            if not node.children_ids and not node.is_pruned
        ]

    def should_terminate_search(self) -> bool:
        """Determine if the search should be terminated."""
        elapsed_time = time.time() - self.search_start_time

        return (
            elapsed_time > self.search_timeout or
            self.current_depth >= self.max_depth or
            len(self.get_active_nodes()) == 0 or
            self.stuck_count > 5
        )

    def update_search_metrics(self, nodes: List[TreeNode]):
        """Update search quality metrics based on current nodes."""
        if not nodes:
            return

        scores = [node.cumulative_score for node in nodes]
        new_best = max(scores)
        new_average = sum(scores) / len(scores)

        # Check for quality decline
        if new_best <= self.best_score and new_average <= self.average_score:
            self.stuck_count += 1
            self.quality_decline = True
        else:
            self.stuck_count = 0
            self.quality_decline = False

        self.best_score = max(self.best_score, new_best)
        self.average_score = 0.7 * self.average_score + 0.3 * new_average

        # Calculate diversity (variance in scores)
        if len(scores) > 1:
            variance = sum((s - new_average) ** 2 for s in scores) / len(scores)
            self.diversity_score = min(1.0, variance / 0.25)  # Normalize by expected variance


class NodeEvaluation(BaseModel):
    """
    Represents the evaluation of a tree node by an agent.
    """

    node_id: str
    agent_name: str
    quality_score: float  # 0.0 to 1.0
    confidence: float     # How confident the agent is in this evaluation
    reasoning: str        # Explanation of the evaluation

    # Detailed metrics
    accuracy: Optional[float] = None
    creativity: Optional[float] = None
    completeness: Optional[float] = None
    clarity: Optional[float] = None

    # Recommendations
    should_expand: bool = True
    recommended_tools: List[str] = []
    next_steps: List[str] = []

    # Metadata
    evaluation_time: float = field(default_factory=time.time)


class TreeSearchResult(BaseModel):
    """
    Final result of a tree search process.
    """

    # Best solutions found
    best_nodes: List[Dict[str, Any]]  # Top-k best nodes as dictionaries
    best_path: List[Dict[str, Any]]   # Best complete path from root to leaf

    # Search statistics
    total_nodes_explored: int
    max_depth_reached: int
    search_duration: float
    total_agent_calls: int

    # Quality metrics
    best_score: float
    average_score: float
    diversity_score: float

    # Agent performance
    agent_statistics: Dict[str, Dict[str, Any]]  # Performance per agent

    # Tree structure (for visualization)
    tree_structure: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True


@dataclass
class TreeGRPONode(TreeNode):
    """
    Extended TreeNode for GRPO-based tree rollout.

    Adds GRPO-specific functionality including reward propagation,
    trajectory tracking, hierarchical advantage computation, and
    intelligent context management.
    """

    # GRPO-specific reward tracking
    final_rewards: List[float] = field(default_factory=list)  # Final rewards from descendant leaves
    aggregated_reward: Optional[float] = None  # Computed reward based on descendants
    reward_statistics: Dict[str, float] = field(default_factory=dict)  # mean, std, min, max

    # Trajectory information for GRPO training - NODE-LEVEL
    node_generation_tokens: List[int] = field(default_factory=list)  # Tokens generated by THIS node only
    node_generation_logprobs: List[float] = field(default_factory=list)  # Log probs for THIS node's tokens
    node_generation_mask: List[int] = field(default_factory=list)  # Mask for THIS node's generation

    # Position tracking for clear boundaries
    generation_start_pos: int = 0  # Where this node starts generating in the sequence
    generation_end_pos: int = 0    # Where this node ends generating in the sequence
    generation_length: int = 0     # Number of tokens generated by this node

    # Full sequence context (for reference, but not for training)
    full_sequence_tokens: Optional[List[int]] = None  # Complete sequence from root to this node
    full_sequence_logprobs: Optional[List[float]] = None

    # Context management
    compressed_context: Optional[str] = None  # Compressed context from path to root
    local_generation_text: str = ""  # Human-readable text of this node's generation
    context_summary: Dict[str, str] = field(default_factory=dict)  # Key insights from previous agents

    # Tree structure for GRPO grouping
    sibling_ids: List[str] = field(default_factory=list)  # Nodes with same parent
    descendant_leaf_ids: List[str] = field(default_factory=list)  # All leaf descendants

    # GRPO training metadata
    grpo_group_id: Optional[str] = None  # Group ID for GRPO advantage computation
    node_advantage: Optional[torch.Tensor] = None  # Advantage for THIS node's generation
    node_returns: Optional[torch.Tensor] = None    # Returns for THIS node's generation
    is_leaf: bool = False

    # Agent and specialization info
    agent_type: str = "general"  # Type of agent that generated this node
    agent_confidence: float = 1.0  # Confidence score from the generating agent

    def collect_leaf_rewards(self, node_registry: Dict[str, "TreeGRPONode"]) -> List[float]:
        """Collect final rewards from all descendant leaf nodes."""
        if self.is_leaf:
            return self.final_rewards.copy()

        all_rewards = []
        for child_id in self.children_ids:
            child = node_registry.get(child_id)
            if child:
                all_rewards.extend(child.collect_leaf_rewards(node_registry))

        self.final_rewards = all_rewards
        return all_rewards

    def compute_reward_statistics(self) -> Dict[str, float]:
        """Compute statistical measures of collected leaf rewards."""
        if not self.final_rewards:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        rewards_array = np.array(self.final_rewards)
        stats = {
            "mean": float(np.mean(rewards_array)),
            "std": float(np.std(rewards_array)),
            "min": float(np.min(rewards_array)),
            "max": float(np.max(rewards_array)),
            "count": len(rewards_array)
        }
        self.reward_statistics = stats
        return stats

    def compute_aggregated_reward(self, aggregation_config: "RewardAggregationConfig") -> float:
        """Compute aggregated reward based on configuration."""
        if not self.reward_statistics:
            self.compute_reward_statistics()

        stats = self.reward_statistics
        if stats["count"] == 0:
            return 0.0

        if aggregation_config.method == "mean":
            reward = stats["mean"]
        elif aggregation_config.method == "mean_minus_std":
            reward = stats["mean"] - aggregation_config.std_penalty * stats["std"]
        elif aggregation_config.method == "mean_plus_std":
            reward = stats["mean"] + aggregation_config.std_bonus * stats["std"]
        elif aggregation_config.method == "weighted_stats":
            reward = (
                aggregation_config.weights.get("mean", 1.0) * stats["mean"] +
                aggregation_config.weights.get("std", 0.0) * stats["std"] +
                aggregation_config.weights.get("min", 0.0) * stats["min"] +
                aggregation_config.weights.get("max", 0.0) * stats["max"]
            )
        elif aggregation_config.method == "risk_adjusted":
            # Sharpe ratio-like: (mean - risk_free_rate) / std
            risk_free = aggregation_config.risk_free_rate
            if stats["std"] > 1e-8:
                reward = (stats["mean"] - risk_free) / stats["std"]
            else:
                reward = stats["mean"] - risk_free
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_config.method}")

        self.aggregated_reward = reward
        return reward

    def get_siblings(self, node_registry: Dict[str, "TreeGRPONode"]) -> List["TreeGRPONode"]:
        """Get all sibling nodes (nodes with same parent)."""
        if not self.parent_id:
            return [self]  # Root node has no siblings

        parent = node_registry.get(self.parent_id)
        if not parent:
            return [self]

        siblings = []
        for child_id in parent.children_ids:
            child = node_registry.get(child_id)
            if child:
                siblings.append(child)

        return siblings

    def set_generation_boundary(self, start_pos: int, tokens: List[int], logprobs: List[float]):
        """Set the generation boundary and data for this node."""
        self.generation_start_pos = start_pos
        self.node_generation_tokens = tokens
        self.node_generation_logprobs = logprobs
        self.generation_length = len(tokens)
        self.generation_end_pos = start_pos + len(tokens)

        # Create mask for this node's generation
        self.node_generation_mask = [1] * len(tokens)

    def get_node_trajectory_data(self) -> Dict[str, torch.Tensor]:
        """Get trajectory data for this specific node (not full sequence)."""
        if not self.node_generation_tokens:
            raise ValueError(f"Node {self.node_id} has no generation data")

        return {
            "tokens": torch.tensor(self.node_generation_tokens),
            "logprobs": torch.tensor(self.node_generation_logprobs),
            "mask": torch.tensor(self.node_generation_mask),
            "aggregated_reward": torch.tensor(self.aggregated_reward or 0.0),
            "node_advantage": self.node_advantage,
            "node_returns": self.node_returns,
            "depth": torch.tensor(self.depth),
            "generation_length": torch.tensor(self.generation_length),
            "node_id": self.node_id
        }

    def prepare_grpo_data(self) -> Dict[str, torch.Tensor]:
        """Prepare data for GRPO training - node-level version."""
        trajectory_data = self.get_node_trajectory_data()

        # Create token-level rewards (broadcast node reward to all tokens)
        if self.aggregated_reward is not None:
            # Assign the aggregated reward to the last token of this node's generation
            token_rewards = torch.zeros(self.generation_length)
            if self.generation_length > 0:
                token_rewards[-1] = self.aggregated_reward
            trajectory_data["token_level_rewards"] = token_rewards
        else:
            trajectory_data["token_level_rewards"] = torch.zeros(self.generation_length)

        # Use the node-level mask
        trajectory_data["response_mask"] = trajectory_data["mask"]
        trajectory_data["old_log_probs"] = trajectory_data["logprobs"]

        return trajectory_data

    def build_context_for_child(
        self,
        context_manager: Optional[Any] = None,
        sibling_nodes: List["TreeGRPONode"] = None
    ) -> str:
        """Build compressed context for a child node."""
        if context_manager is None:
            # Fallback to simple concatenation (legacy behavior)
            current_messages = [msg["content"] for msg in self.messages if msg.get("content")]
            return " | ".join(current_messages)

        # Use intelligent context management
        current_info = {
            "depth": self.depth + 1,  # Child will be one level deeper
            "agent_type": getattr(self, "agent_type", "general"),
            "content": self.local_generation_text or self.messages[-1].get("content", "") if self.messages else ""
        }

        # Prepare sibling data
        sibling_data = []
        if sibling_nodes:
            for sibling in sibling_nodes:
                sibling_data.append({
                    "content": sibling.local_generation_text,
                    "agent_type": getattr(sibling, "agent_type", "general"),
                    "confidence": getattr(sibling, "agent_confidence", 1.0)
                })

        # Build context using the context manager
        try:
            import asyncio
            # Note: This should ideally be called from an async context
            # For now, we'll provide a simplified synchronous version
            if hasattr(context_manager, "build_simple_context"):
                return context_manager.build_simple_context(
                    current_info, self.compressed_context, sibling_data
                )
            else:
                # Fallback
                return self.compressed_context or self._simple_context_fallback()
        except Exception as e:
            import logging
            logging.warning(f"Context management failed: {e}, using fallback")
            return self._simple_context_fallback()

    def _simple_context_fallback(self) -> str:
        """Simple fallback context building when advanced methods fail."""
        if self.compressed_context:
            return self.compressed_context
        elif self.messages:
            # Take last few messages
            recent_messages = self.messages[-3:] if len(self.messages) > 3 else self.messages
            return " | ".join(msg.get("content", "") for msg in recent_messages if msg.get("content"))
        else:
            return ""

    def update_context_summary(self, insights: Dict[str, str]):
        """Update the context summary with new insights."""
        self.context_summary.update(insights)


@dataclass
class RewardAggregationConfig:
    """Configuration for reward aggregation in tree GRPO."""

    method: str = "mean_minus_std"  # mean, mean_minus_std, mean_plus_std, weighted_stats, risk_adjusted

    # Simple aggregation parameters
    std_penalty: float = 0.1  # For mean_minus_std
    std_bonus: float = 0.1    # For mean_plus_std

    # Weighted aggregation parameters
    weights: Dict[str, float] = field(default_factory=lambda: {
        "mean": 1.0, "std": -0.2, "min": -0.05, "max": 0.1
    })

    # Risk-adjusted parameters
    risk_free_rate: float = 0.0


@dataclass
class TreeStructureConfig:
    """Configuration for tree branching structure."""

    max_depth: int = 3

    # Branching pattern options (mutually exclusive)
    uniform_branching: Optional[int] = None  # Same branches per level
    branching_schedule: Optional[List[int]] = None  # Custom per-level
    custom_branching: Optional[Dict[str, int]] = None  # Depth-specific mapping

    # Adaptive branching
    adaptive_branching: Optional[Dict[str, Any]] = None

    # Derived schedule (computed from above options)
    _computed_schedule: Optional[List[int]] = None

    def get_branching_schedule(self) -> List[int]:
        """Get the computed branching schedule."""
        if self._computed_schedule is not None:
            return self._computed_schedule

        if self.uniform_branching is not None:
            self._computed_schedule = [self.uniform_branching] * self.max_depth
        elif self.branching_schedule is not None:
            # Pad or truncate to max_depth
            schedule = self.branching_schedule[:self.max_depth]
            while len(schedule) < self.max_depth:
                schedule.append(schedule[-1] if schedule else 2)
            self._computed_schedule = schedule
        elif self.custom_branching is not None:
            schedule = []
            for depth in range(1, self.max_depth + 1):
                key = f"depth_{depth}"
                branches = self.custom_branching.get(key, 2)
                schedule.append(branches)
            self._computed_schedule = schedule
        else:
            # Default: uniform 2-way branching
            self._computed_schedule = [2] * self.max_depth

        return self._computed_schedule

    def get_branches_for_depth(self, depth: int) -> int:
        """Get number of branches for a specific depth level."""
        schedule = self.get_branching_schedule()
        if depth <= 0 or depth > len(schedule):
            return 0
        return schedule[depth - 1]


@dataclass
class GroupingConfig:
    """Configuration for GRPO grouping strategy."""

    method: str = "by_parent_and_depth"  # by_parent_and_depth, by_depth_only, by_reward_range

    # Additional grouping parameters
    reward_range_bins: int = 5  # For by_reward_range method
    min_group_size: int = 2    # Minimum nodes per group
    max_group_size: int = 8    # Maximum nodes per group


@dataclass
class TreeGRPOConfig:
    """Complete configuration for tree-based GRPO."""

    tree_structure: TreeStructureConfig = field(default_factory=TreeStructureConfig)
    reward_aggregation: RewardAggregationConfig = field(default_factory=RewardAggregationConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)

    # GRPO-specific parameters
    epsilon: float = 1e-6
    norm_adv_by_std: bool = True

    # Training parameters
    separate_depth_training: bool = True  # Train each depth separately
    propagate_gradients: bool = False     # Whether to propagate gradients up the tree


class TreeStructureBuilder:
    """Builder for creating tree structures based on configuration."""

    def __init__(self, config: TreeStructureConfig):
        self.config = config
        self.branching_schedule = config.get_branching_schedule()

    def compute_total_nodes(self) -> int:
        """Compute total number of nodes in the complete tree."""
        total = 1  # Root node
        nodes_at_depth = 1

        for depth in range(1, self.config.max_depth + 1):
            branches = self.config.get_branches_for_depth(depth)
            nodes_at_depth *= branches
            total += nodes_at_depth

        return total

    def compute_leaf_count(self) -> int:
        """Compute number of leaf nodes."""
        if not self.branching_schedule:
            return 1

        leaf_count = 1
        for branches in self.branching_schedule:
            leaf_count *= branches

        return leaf_count

    def generate_tree_template(self) -> Dict[str, Any]:
        """Generate tree structure template."""
        return {
            "max_depth": self.config.max_depth,
            "branching_schedule": self.branching_schedule,
            "total_nodes": self.compute_total_nodes(),
            "leaf_count": self.compute_leaf_count(),
            "structure_summary": self._generate_structure_summary()
        }

    def _generate_structure_summary(self) -> str:
        """Generate human-readable structure summary."""
        if not self.branching_schedule:
            return "1 (root only)"

        nodes_per_level = [1]  # Root
        for branches in self.branching_schedule:
            nodes_per_level.append(nodes_per_level[-1] * branches)

        summary = "→".join(str(count) for count in nodes_per_level)
        return summary