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
Tree-based GRPO implementation for multi-agent hierarchical rollout.

This module implements a critic-free GRPO variant that uses tree-structured rollouts
where each node's reward is computed from statistics of subsequent leaf rewards,
supporting flexible branching patterns and hierarchical advantage computation.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch

from verl.trainer.config import AlgoConfig
from verl.utils import group_mean_std, as_torch_index
from .tree_structures import (
    TreeGRPONode,
    TreeGRPOConfig,
    RewardAggregationConfig,
    GroupingConfig,
    TreeStructureConfig,
    TreeStructureBuilder
)

logger = logging.getLogger(__name__)


class TreeRewardPropagator:
    """
    Handles reward propagation from leaves to internal nodes in the tree.
    """

    def __init__(self, config: RewardAggregationConfig):
        self.config = config

    def propagate_rewards(self, node_registry: Dict[str, TreeGRPONode]) -> Dict[str, float]:
        """
        Propagate rewards from leaf nodes up to internal nodes.

        Args:
            node_registry: Dictionary mapping node IDs to TreeGRPONode instances

        Returns:
            Dictionary mapping node IDs to their computed aggregated rewards
        """
        # Find all leaf nodes and ensure they have final rewards
        leaf_nodes = [node for node in node_registry.values() if node.is_leaf]

        if not leaf_nodes:
            logger.warning("No leaf nodes found for reward propagation")
            return {}

        # Verify leaf nodes have rewards
        for leaf in leaf_nodes:
            if not leaf.final_rewards:
                logger.error(f"Leaf node {leaf.node_id} has no final rewards")

        # Propagate rewards bottom-up using topological sort
        propagation_order = self._get_propagation_order(node_registry)
        aggregated_rewards = {}

        for node_id in propagation_order:
            node = node_registry[node_id]

            # Collect rewards from descendants
            node.collect_leaf_rewards(node_registry)

            # Compute aggregated reward
            if node.final_rewards:
                reward = node.compute_aggregated_reward(self.config)
                aggregated_rewards[node_id] = reward
                logger.debug(f"Node {node_id} (depth={node.depth}): "
                           f"aggregated_reward={reward:.4f}, "
                           f"stats={node.reward_statistics}")
            else:
                aggregated_rewards[node_id] = 0.0
                logger.warning(f"Node {node_id} has no descendant rewards")

        return aggregated_rewards

    def _get_propagation_order(self, node_registry: Dict[str, TreeGRPONode]) -> List[str]:
        """Get nodes in bottom-up topological order for reward propagation."""
        # Group nodes by depth (deeper nodes first)
        depth_groups = defaultdict(list)
        for node_id, node in node_registry.items():
            depth_groups[node.depth].append(node_id)

        # Process from deepest to shallowest
        propagation_order = []
        for depth in sorted(depth_groups.keys(), reverse=True):
            propagation_order.extend(depth_groups[depth])

        return propagation_order


class TreeGRPOGrouper:
    """
    Handles grouping of tree nodes for GRPO advantage computation.
    """

    def __init__(self, config: GroupingConfig):
        self.config = config

    def create_groups(self, node_registry: Dict[str, TreeGRPONode]) -> Dict[str, List[str]]:
        """
        Create groups of nodes for GRPO advantage computation.

        Args:
            node_registry: Dictionary of all nodes

        Returns:
            Dictionary mapping group IDs to lists of node IDs
        """
        if self.config.method == "by_parent_and_depth":
            return self._group_by_parent_and_depth(node_registry)
        elif self.config.method == "by_depth_only":
            return self._group_by_depth_only(node_registry)
        elif self.config.method == "by_reward_range":
            return self._group_by_reward_range(node_registry)
        else:
            raise ValueError(f"Unknown grouping method: {self.config.method}")

    def _group_by_parent_and_depth(self, node_registry: Dict[str, TreeGRPONode]) -> Dict[str, List[str]]:
        """Group nodes by parent and depth (siblings)."""
        groups = defaultdict(list)

        for node_id, node in node_registry.items():
            if node.depth == 0:
                # Root node forms its own group
                group_id = f"root_depth_0"
            else:
                # Group by parent and depth
                parent_id = node.parent_id or "no_parent"
                group_id = f"parent_{parent_id}_depth_{node.depth}"

            groups[group_id].append(node_id)
            node.grpo_group_id = group_id

        # Filter groups by size constraints
        return self._filter_groups_by_size(groups)

    def _group_by_depth_only(self, node_registry: Dict[str, TreeGRPONode]) -> Dict[str, List[str]]:
        """Group nodes by depth level only."""
        groups = defaultdict(list)

        for node_id, node in node_registry.items():
            group_id = f"depth_{node.depth}"
            groups[group_id].append(node_id)
            node.grpo_group_id = group_id

        return self._filter_groups_by_size(groups)

    def _group_by_reward_range(self, node_registry: Dict[str, TreeGRPONode]) -> Dict[str, List[str]]:
        """Group nodes by reward range bins."""
        # Collect all aggregated rewards
        rewards = []
        node_rewards = {}

        for node_id, node in node_registry.items():
            if node.aggregated_reward is not None:
                rewards.append(node.aggregated_reward)
                node_rewards[node_id] = node.aggregated_reward

        if not rewards:
            # Fallback to depth grouping
            return self._group_by_depth_only(node_registry)

        # Create reward bins
        min_reward, max_reward = min(rewards), max(rewards)
        if max_reward - min_reward < 1e-8:
            # All rewards are the same, use single group
            group_id = "uniform_rewards"
            return {group_id: list(node_rewards.keys())}

        bin_size = (max_reward - min_reward) / self.config.reward_range_bins
        groups = defaultdict(list)

        for node_id, reward in node_rewards.items():
            bin_idx = min(int((reward - min_reward) / bin_size), self.config.reward_range_bins - 1)
            group_id = f"reward_bin_{bin_idx}"
            groups[group_id].append(node_id)
            node_registry[node_id].grpo_group_id = group_id

        return self._filter_groups_by_size(groups)

    def _filter_groups_by_size(self, groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Filter groups by minimum and maximum size constraints."""
        filtered_groups = {}

        for group_id, node_ids in groups.items():
            if len(node_ids) >= self.config.min_group_size:
                # Split large groups if necessary
                if len(node_ids) <= self.config.max_group_size:
                    filtered_groups[group_id] = node_ids
                else:
                    # Split into smaller groups
                    for i in range(0, len(node_ids), self.config.max_group_size):
                        chunk = node_ids[i:i + self.config.max_group_size]
                        if len(chunk) >= self.config.min_group_size:
                            filtered_groups[f"{group_id}_split_{i//self.config.max_group_size}"] = chunk

        logger.info(f"Created {len(filtered_groups)} GRPO groups after filtering")
        return filtered_groups


class TreeGRPOAdvantageComputer:
    """
    Computes GRPO advantages for tree-structured trajectories.
    """

    def __init__(self, config: TreeGRPOConfig):
        self.config = config
        self.grouper = TreeGRPOGrouper(config.grouping)

    def compute_advantages(
        self,
        node_registry: Dict[str, TreeGRPONode],
        groups: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute GRPO advantages for all nodes in the tree.

        Args:
            node_registry: Dictionary of all nodes
            groups: Optional pre-computed groups, if None will create new groups

        Returns:
            Dictionary mapping node IDs to (advantages, returns) tensors
        """
        if groups is None:
            groups = self.grouper.create_groups(node_registry)

        all_advantages = {}

        if self.config.separate_depth_training:
            # Compute advantages separately for each depth level
            depth_groups = self._group_by_depth(groups, node_registry)
            for depth, depth_group_dict in depth_groups.items():
                depth_advantages = self._compute_depth_advantages(depth_group_dict, node_registry)
                all_advantages.update(depth_advantages)
        else:
            # Compute advantages for all groups together
            for group_id, node_ids in groups.items():
                group_advantages = self._compute_group_advantages(node_ids, node_registry)
                all_advantages.update(group_advantages)

        return all_advantages

    def _group_by_depth(
        self,
        groups: Dict[str, List[str]],
        node_registry: Dict[str, TreeGRPONode]
    ) -> Dict[int, Dict[str, List[str]]]:
        """Group the existing groups by depth level."""
        depth_groups = defaultdict(dict)

        for group_id, node_ids in groups.items():
            if not node_ids:
                continue

            # Get depth of first node (all nodes in group should have same depth for some grouping methods)
            first_node = node_registry[node_ids[0]]
            depth = first_node.depth

            # Verify all nodes in group have compatible depths
            for node_id in node_ids:
                node_depth = node_registry[node_id].depth
                if self.config.grouping.method == "by_parent_and_depth" and node_depth != depth:
                    logger.warning(f"Inconsistent depths in group {group_id}: {depth} vs {node_depth}")

            depth_groups[depth][group_id] = node_ids

        return depth_groups

    def _compute_depth_advantages(
        self,
        depth_groups: Dict[str, List[str]],
        node_registry: Dict[str, TreeGRPONode]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute advantages for all groups at a specific depth."""
        all_advantages = {}

        for group_id, node_ids in depth_groups.items():
            group_advantages = self._compute_group_advantages(node_ids, node_registry)
            all_advantages.update(group_advantages)

        return all_advantages

    def _compute_group_advantages(
        self,
        node_ids: List[str],
        node_registry: Dict[str, TreeGRPONode]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute GRPO advantages for a single group of nodes using node-level data."""
        if len(node_ids) < 2:
            logger.warning(f"Group with {len(node_ids)} nodes, using zero advantages")
            advantages = {}
            for node_id in node_ids:
                node = node_registry[node_id]
                if node.generation_length > 0:
                    zero_advantages = torch.zeros(node.generation_length)
                    advantages[node_id] = (zero_advantages, zero_advantages)
            return advantages

        # Collect node-level data from all nodes in the group
        group_data = []
        max_length = 0

        for node_id in node_ids:
            node = node_registry[node_id]
            try:
                data = node.prepare_grpo_data()
                data["node_id"] = node_id
                group_data.append(data)
                max_length = max(max_length, node.generation_length)
            except ValueError as e:
                logger.warning(f"Skipping node {node_id}: {e}")

        if len(group_data) < 2:
            logger.warning(f"Insufficient valid nodes in group: {len(group_data)}")
            return {}

        # Pad sequences to same length for batch processing
        padded_rewards = []
        padded_masks = []
        node_rewards = []

        for data in group_data:
            reward_seq = data["token_level_rewards"]
            mask_seq = data["response_mask"]
            node_reward = data["aggregated_reward"]

            # Pad to max length
            if len(reward_seq) < max_length:
                padding_length = max_length - len(reward_seq)
                reward_seq = torch.cat([reward_seq, torch.zeros(padding_length)])
                mask_seq = torch.cat([mask_seq, torch.zeros(padding_length)])

            padded_rewards.append(reward_seq)
            padded_masks.append(mask_seq)
            node_rewards.append(node_reward)

        # Stack tensors for batch processing
        token_level_rewards = torch.stack(padded_rewards)
        response_mask = torch.stack(padded_masks)
        aggregated_rewards = torch.stack(node_rewards)

        # Create index array for GRPO grouping (all nodes in same group)
        index = np.zeros(len(group_data), dtype=int)

        # Use node-level reward assignment strategy:
        # Distribute the aggregated reward across the node's generation
        modified_rewards = token_level_rewards.clone()

        for i, (data, mask) in enumerate(zip(group_data, padded_masks)):
            node = node_registry[data["node_id"]]

            if node.generation_length > 0:
                # Strategy 1: Assign full reward to last token of node's generation
                valid_positions = mask.nonzero().flatten()
                if len(valid_positions) > 0:
                    last_valid_pos = valid_positions[-1].item()
                    modified_rewards[i, :] = 0.0
                    modified_rewards[i, last_valid_pos] = aggregated_rewards[i]

                # Alternative Strategy 2 (commented): Distribute across all tokens
                # reward_per_token = aggregated_rewards[i] / node.generation_length
                # for pos in valid_positions:
                #     modified_rewards[i, pos] = reward_per_token

        # Compute GRPO advantages using modified rewards
        advantages, returns = self._compute_grpo_outcome_advantage(
            modified_rewards, response_mask, index,
            self.config.epsilon, self.config.norm_adv_by_std
        )

        # Map back to individual nodes with proper sequence lengths
        result = {}
        for i, data in enumerate(group_data):
            node_id = data["node_id"]
            node = node_registry[node_id]

            # Trim to actual node generation length
            node_advantages = advantages[i][:node.generation_length]
            node_returns = returns[i][:node.generation_length]

            result[node_id] = (node_advantages, node_returns)

            # Store in node for reference
            node_registry[node_id].node_advantage = node_advantages
            node_registry[node_id].node_returns = node_returns

        return result

    def _compute_grpo_outcome_advantage(
        self,
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        index: np.ndarray,
        epsilon: float = 1e-6,
        norm_adv_by_std: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GRPO advantages for tree nodes.

        This is adapted from the standard GRPO implementation but uses
        aggregated rewards from tree structure rather than raw final rewards.
        """
        scores = token_level_rewards.sum(dim=-1)

        id2score = defaultdict(list)
        id2mean = {}
        id2std = {}

        with torch.no_grad():
            bsz = scores.shape[0]
            for i in range(bsz):
                id2score[index[i]].append(scores[i])

            for idx in id2score:
                if len(id2score[idx]) == 1:
                    id2mean[idx] = torch.tensor(0.0)
                    id2std[idx] = torch.tensor(1.0)
                elif len(id2score[idx]) > 1:
                    scores_tensor = torch.stack(id2score[idx])
                    id2mean[idx] = torch.mean(scores_tensor)
                    id2std[idx] = torch.std(scores_tensor)
                else:
                    raise ValueError(f"No score in group index: {idx}")

            for i in range(bsz):
                if norm_adv_by_std:
                    scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
                else:
                    scores[i] = scores[i] - id2mean[index[i]]

            scores = scores.unsqueeze(-1) * response_mask

        return scores, scores


def compute_grpo_tree_advantage(
    tree_nodes: Dict[str, TreeGRPONode],
    config: TreeGRPOConfig,
    final_rewards: Optional[Dict[str, List[float]]] = None,
    **kwargs
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Main entry point for computing tree-based GRPO advantages.

    Args:
        tree_nodes: Dictionary mapping node IDs to TreeGRPONode instances
        config: Tree GRPO configuration
        final_rewards: Optional dictionary mapping leaf node IDs to their final rewards

    Returns:
        Dictionary mapping node IDs to (advantages, returns) tensors
    """
    logger.info(f"Computing tree GRPO advantages for {len(tree_nodes)} nodes")

    # Set final rewards for leaf nodes if provided
    if final_rewards:
        for node_id, rewards in final_rewards.items():
            if node_id in tree_nodes:
                tree_nodes[node_id].final_rewards = rewards
                tree_nodes[node_id].is_leaf = True

    # Step 1: Propagate rewards from leaves to internal nodes
    logger.info("Step 1: Propagating rewards from leaves to internal nodes")
    propagator = TreeRewardPropagator(config.reward_aggregation)
    aggregated_rewards = propagator.propagate_rewards(tree_nodes)

    # Step 2: Create groups for GRPO computation
    logger.info("Step 2: Creating groups for GRPO computation")
    grouper = TreeGRPOGrouper(config.grouping)
    groups = grouper.create_groups(tree_nodes)

    # Step 3: Compute advantages
    logger.info("Step 3: Computing GRPO advantages")
    advantage_computer = TreeGRPOAdvantageComputer(config)
    advantages = advantage_computer.compute_advantages(tree_nodes, groups)

    logger.info(f"Computed advantages for {len(advantages)} nodes across {len(groups)} groups")

    return advantages


def create_tree_grpo_config_template(template_name: str) -> TreeGRPOConfig:
    """Create predefined tree GRPO configuration templates."""

    if template_name == "binary_tree":
        # Classic 1→2→4→8 pattern
        return TreeGRPOConfig(
            tree_structure=TreeStructureConfig(
                max_depth=3,
                uniform_branching=2
            ),
            reward_aggregation=RewardAggregationConfig(
                method="mean_minus_std",
                std_penalty=0.1
            ),
            grouping=GroupingConfig(
                method="by_parent_and_depth",
                min_group_size=2,
                max_group_size=8
            ),
            separate_depth_training=True
        )

    elif template_name == "ternary_tree":
        # 1→3→9→27 pattern
        return TreeGRPOConfig(
            tree_structure=TreeStructureConfig(
                max_depth=3,
                uniform_branching=3
            ),
            reward_aggregation=RewardAggregationConfig(
                method="weighted_stats",
                weights={"mean": 1.0, "std": -0.15, "max": 0.1}
            ),
            grouping=GroupingConfig(
                method="by_parent_and_depth"
            )
        )

    elif template_name == "asymmetric":
        # Custom asymmetric pattern
        return TreeGRPOConfig(
            tree_structure=TreeStructureConfig(
                max_depth=4,
                branching_schedule=[3, 2, 4, 1]  # 1→3→6→24→24
            ),
            reward_aggregation=RewardAggregationConfig(
                method="risk_adjusted",
                risk_free_rate=0.0
            ),
            grouping=GroupingConfig(
                method="by_depth_only"
            )
        )

    elif template_name == "conservative":
        # Risk-averse configuration
        return TreeGRPOConfig(
            tree_structure=TreeStructureConfig(
                max_depth=2,
                uniform_branching=2
            ),
            reward_aggregation=RewardAggregationConfig(
                method="mean_minus_std",
                std_penalty=0.3  # Heavy penalty for variance
            ),
            grouping=GroupingConfig(
                method="by_parent_and_depth"
            ),
            separate_depth_training=True,
            norm_adv_by_std=True
        )

    else:
        raise ValueError(f"Unknown template: {template_name}. "
                        f"Available: binary_tree, ternary_tree, asymmetric, conservative")