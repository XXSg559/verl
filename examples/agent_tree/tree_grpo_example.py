#!/usr/bin/env python3
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
Example script demonstrating Tree-based GRPO functionality.

This script shows how to:
1. Create tree GRPO configurations
2. Generate tree-structured rollouts
3. Compute hierarchical advantages
4. Integrate with verl training
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add verl to path
sys.path.append(str(Path(__file__).parent.parent))

from verl.experimental.agent_tree import (
    TreeGRPONode,
    TreeGRPOConfig,
    TreeStructureConfig,
    RewardAggregationConfig,
    GroupingConfig,
    compute_grpo_tree_advantage,
    create_tree_grpo_config_template,
    get_tree_grpo_config_template,
    get_available_grpo_templates
)

import torch
import numpy as np

logger = logging.getLogger(__name__)


def create_mock_tree_nodes() -> dict:
    """Create a mock tree structure for demonstration."""
    print("Creating mock tree structure (1→2→4 pattern)...")

    nodes = {}

    # Root node
    root = TreeGRPONode.from_prompt(
        messages=[{"role": "user", "content": "Solve: 2x + 3 = 7"}],
        depth=0
    )
    root.token_level_rewards = torch.zeros(1, 10)  # Mock token rewards
    root.response_mask = torch.ones(1, 10)
    root.old_log_probs = torch.randn(1, 10) * 0.1
    nodes[root.node_id] = root

    # Depth 1: 2 children
    for i in range(2):
        child = TreeGRPONode(
            parent_id=root.node_id,
            depth=1,
            messages=root.messages + [{"role": "assistant", "content": f"Approach {i+1}: "}],
            agent_name=f"explorer_depth1_{i}"
        )
        child.token_level_rewards = torch.zeros(1, 15)
        child.response_mask = torch.ones(1, 15)
        child.old_log_probs = torch.randn(1, 15) * 0.1

        root.add_child(child.node_id)
        nodes[child.node_id] = child

        # Depth 2: 2 children each (4 total leaf nodes)
        for j in range(2):
            grandchild = TreeGRPONode(
                parent_id=child.node_id,
                depth=2,
                messages=child.messages + [{"role": "assistant", "content": f"Step {j+1}: "}],
                agent_name=f"explorer_depth2_{i}_{j}",
                is_leaf=True
            )
            grandchild.token_level_rewards = torch.zeros(1, 20)
            grandchild.response_mask = torch.ones(1, 20)
            grandchild.old_log_probs = torch.randn(1, 20) * 0.1

            # Assign mock final rewards (simulate reward model output)
            reward = 0.5 + np.random.normal(0, 0.2)  # Base reward + noise
            grandchild.final_rewards = [max(0.0, min(1.0, reward))]

            child.add_child(grandchild.node_id)
            nodes[grandchild.node_id] = grandchild

    print(f"Created tree with {len(nodes)} nodes:")
    depth_counts = {}
    for node in nodes.values():
        depth_counts[node.depth] = depth_counts.get(node.depth, 0) + 1

    structure_summary = "→".join(str(depth_counts[d]) for d in sorted(depth_counts.keys()))
    print(f"Structure: {structure_summary}")

    return nodes


def demonstrate_tree_grpo_configs():
    """Demonstrate different tree GRPO configuration templates."""
    print("\n=== Tree GRPO Configuration Templates ===")

    available_templates = get_available_grpo_templates()
    print(f"Available templates: {available_templates}")

    for template_name in available_templates[:3]:  # Show first 3 templates
        print(f"\n{template_name.upper()} Template:")
        try:
            config = get_tree_grpo_config_template(template_name)
            tree_config = config["tree_grpo_config"]

            # Show key configuration details
            tree_structure = tree_config["tree_structure"]
            reward_agg = tree_config["reward_aggregation"]
            grouping = tree_config["grouping"]

            print(f"  Tree Structure:")
            if "uniform_branching" in tree_structure:
                branches = tree_structure["uniform_branching"]
                depth = tree_structure["max_depth"]
                print(f"    Pattern: {branches}-way branching, {depth} levels")
            elif "branching_schedule" in tree_structure:
                schedule = tree_structure["branching_schedule"]
                print(f"    Pattern: {schedule}")

            print(f"  Reward Aggregation: {reward_agg['method']}")
            print(f"  Grouping Strategy: {grouping['method']}")

        except Exception as e:
            print(f"  Error: {e}")


async def demonstrate_tree_grpo_computation():
    """Demonstrate tree GRPO advantage computation."""
    print("\n=== Tree GRPO Advantage Computation ===")

    # Create mock tree structure
    tree_nodes = create_mock_tree_nodes()

    # Create GRPO configuration
    config = create_tree_grpo_config_template("binary_tree")
    print(f"Using configuration: {config.tree_structure.get_branching_schedule()}")

    # Collect final rewards for leaf nodes
    final_rewards = {}
    for node in tree_nodes.values():
        if node.is_leaf and node.final_rewards:
            final_rewards[node.node_id] = node.final_rewards

    print(f"Final rewards from {len(final_rewards)} leaf nodes:")
    for node_id, rewards in final_rewards.items():
        print(f"  {node_id}: {rewards[0]:.3f}")

    try:
        # Compute tree GRPO advantages
        print("\nComputing tree GRPO advantages...")
        advantages = compute_grpo_tree_advantage(
            tree_nodes=tree_nodes,
            config=config,
            final_rewards=final_rewards
        )

        print(f"Computed advantages for {len(advantages)} nodes:")
        for node_id, (adv, ret) in advantages.items():
            node = tree_nodes[node_id]
            print(f"  Node {node_id} (depth={node.depth}): "
                 f"advantage_mean={adv.mean().item():.4f}, "
                 f"return_mean={ret.mean().item():.4f}")

        # Show reward propagation results
        print("\nReward propagation results:")
        for node in tree_nodes.values():
            if node.aggregated_reward is not None:
                stats = node.reward_statistics
                print(f"  {node.node_id} (depth={node.depth}): "
                     f"aggregated={node.aggregated_reward:.4f}, "
                     f"stats={stats}")

    except Exception as e:
        print(f"Error computing advantages: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_configuration_patterns():
    """Demonstrate different branching patterns and their properties."""
    print("\n=== Tree Structure Patterns ===")

    patterns = [
        ("Binary Tree", {"max_depth": 3, "uniform_branching": 2}),
        ("Ternary Tree", {"max_depth": 3, "uniform_branching": 3}),
        ("Asymmetric", {"max_depth": 4, "branching_schedule": [3, 2, 4, 1]}),
        ("Conservative", {"max_depth": 2, "uniform_branching": 2}),
    ]

    for name, structure_config in patterns:
        print(f"\n{name}:")
        config = TreeStructureConfig(**structure_config)

        from verl.experimental.agent_tree.tree_structures import TreeStructureBuilder
        builder = TreeStructureBuilder(config)

        template = builder.generate_tree_template()
        print(f"  Structure: {template['structure_summary']}")
        print(f"  Total nodes: {template['total_nodes']}")
        print(f"  Leaf nodes: {template['leaf_count']}")
        print(f"  Branching schedule: {template['branching_schedule']}")


def main():
    """Main demonstration function."""
    print("Tree-based GRPO Demonstration")
    print("=" * 50)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    async def run_demos():
        """Run all demonstrations."""
        demonstrate_tree_grpo_configs()
        demonstrate_configuration_patterns()
        await demonstrate_tree_grpo_computation()

        print("\n" + "=" * 50)
        print("Demonstration completed!")
        print("\nTo use Tree GRPO in your verl training:")
        print("1. Use the configuration example in examples/tree_grpo_config_example.yaml")
        print("2. Set algorithm.grpo.advantage_estimator to 'grpo_tree'")
        print("3. Configure tree_grpo_config with your desired branching pattern")
        print("4. Run training - the system will use tree-based reward propagation")
        print("\nKey benefits:")
        print("- No critic required (critic-free GRPO)")
        print("- Hierarchical reward assignment based on descendant statistics")
        print("- Flexible branching patterns (not limited to 1→2→4→8)")
        print("- Multi-agent tree rollouts with configurable specializations")

    # Run the async demonstrations
    asyncio.run(run_demos())


if __name__ == "__main__":
    main()