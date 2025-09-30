#!/usr/bin/env python3

"""
Example: Smolagents + Tree GRPO Integration

This example demonstrates how to use smolagents managed agents with tree GRPO
for multi-agent collaborative problem solving with hierarchical reward propagation.
"""

import os
import sys
import logging
from typing import Dict, Any

# Add verl to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from verl.experimental.agent_tree import (
    TreeGRPONode, SearchContext, TreeGRPOConfig,
    SMOLAGENTS_INTEGRATION_AVAILABLE
)

if SMOLAGENTS_INTEGRATION_AVAILABLE:
    from verl.experimental.agent_tree import (
        SmolagentsTreeOrchestrator, SmolagentsTreeConfig,
        SmolagentsTreeCoordinator, MultiProposalGenerator
    )
    from smolagents import CodeAgent, ToolCallingAgent, LiteLLMModel
    from smolagents.tools import WebSearchTool, PythonInterpreterTool
else:
    print("Smolagents integration not available. Please install smolagents.")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_research_agent(model):
    """Create a research-focused agent."""
    return ToolCallingAgent(
        tools=[WebSearchTool()],
        model=model,
        name="research_agent",
        description="Specializes in web research and information gathering.",
        verbosity_level=0,  # Silent mode
        max_steps=5
    )


def create_coding_agent(model):
    """Create a coding-focused agent."""
    return ToolCallingAgent(
        tools=[PythonInterpreterTool()],
        model=model,
        name="coding_agent",
        description="Specializes in code generation, analysis, and execution.",
        verbosity_level=0,  # Silent mode
        max_steps=5
    )


def create_reasoning_agent(model):
    """Create a reasoning-focused agent."""
    return ToolCallingAgent(
        tools=[],  # Pure reasoning, no tools needed
        model=model,
        name="reasoning_agent",
        description="Specializes in logical analysis, problem decomposition, and strategic thinking.",
        verbosity_level=0,  # Silent mode
        max_steps=3
    )


def setup_coordinator_with_managed_agents():
    """Set up the main coordinator agent with managed sub-agents."""

    # Initialize model (replace with your preferred model)
    model = LiteLLMModel(
        model_id="openai/gpt-4",
        # Add your model configuration here
    )

    # Create specialized managed agents
    research_agent = create_research_agent(model)
    coding_agent = create_coding_agent(model)
    reasoning_agent = create_reasoning_agent(model)

    # Create coordinator agent with managed agents
    coordinator = CodeAgent(
        tools=[],  # Coordinator focuses on coordination, managed agents do the work
        model=model,
        managed_agents=[research_agent, coding_agent, reasoning_agent],
        max_steps=10,
        verbosity_level=1,  # Some visibility for coordination decisions
        planning_interval=3,
        name="tree_coordinator",
        description="Coordinates multiple specialized agents for complex problem solving."
    )

    return coordinator


def create_example_tree_structure():
    """Create initial tree structure for the example."""

    # Create root node with initial problem
    initial_messages = [
        {
            "role": "user",
            "content": """
            Problem: Analyze the impact of AI on job markets in the tech industry.

            Requirements:
            1. Research current trends and statistics
            2. Identify specific job categories affected
            3. Analyze code automation impact on software development roles
            4. Provide data-driven insights and predictions

            Provide a comprehensive analysis with supporting evidence.
            """
        }
    ]

    root_node = TreeGRPONode.from_prompt(
        messages=initial_messages,
        node_id="root_analysis_task"
    )

    # Create search context
    search_context = SearchContext(
        max_depth=4,
        max_branches_per_node=6,  # 2 agents * 3 proposals each = 6 combinations max
        max_active_nodes=20,
        problem_type="research_analysis",
        difficulty_level=0.7
    )

    search_context.register_node(root_node)

    return root_node, search_context


def run_smolagents_tree_expansion_example():
    """Run a complete example of smolagents tree expansion."""

    logger.info("Starting Smolagents + Tree GRPO example")

    # 1. Set up coordinator with managed agents
    logger.info("Setting up coordinator with managed agents...")
    coordinator = setup_coordinator_with_managed_agents()

    # 2. Create smolagents tree configuration
    config = SmolagentsTreeConfig(
        coordination_strategy="hierarchical",
        proposals_per_agent=2,
        enable_memory_isolation=True,
        fusion_strategy="structured_combination",
        enable_silent_mode=True
    )

    # 3. Initialize orchestrator
    orchestrator = SmolagentsTreeOrchestrator(
        coordinator_agent=coordinator,
        config=config
    )

    # 4. Create initial tree structure
    root_node, search_context = create_example_tree_structure()

    logger.info(f"Created root node: {root_node.node_id}")
    logger.info(f"Available managed agents: {list(coordinator.managed_agents.keys())}")

    # 5. Perform tree expansion
    logger.info("Expanding tree with multi-agent coordination...")

    try:
        # First level expansion
        level_1_nodes = orchestrator.expand_tree_node(root_node, search_context)
        logger.info(f"Level 1 expansion created {len(level_1_nodes)} combination nodes")

        # Display combination results
        for i, node in enumerate(level_1_nodes):
            logger.info(f"\nCombination Node {i+1}:")
            if hasattr(node, 'agent_contributions'):
                for agent_name, contribution in node.agent_contributions.items():
                    logger.info(f"  {agent_name}: {contribution.output_text[:100]}...")
            logger.info(f"  Combined output: {node.local_generation_text[:150]}...")

        # Second level expansion (expand best node from level 1)
        if level_1_nodes:
            best_node = level_1_nodes[0]  # For demo, pick first node
            logger.info(f"\nExpanding best node from level 1: {best_node.node_id}")

            level_2_nodes = orchestrator.expand_tree_node(best_node, search_context)
            logger.info(f"Level 2 expansion created {len(level_2_nodes)} combination nodes")

        # 6. Get performance metrics
        metrics = orchestrator.get_performance_metrics()
        logger.info(f"\nPerformance Metrics:")
        logger.info(f"  Total expansions: {metrics['total_expansions']}")
        logger.info(f"  Successful expansions: {metrics['successful_expansions']}")
        logger.info(f"  Average combo nodes per expansion: {metrics['average_combo_nodes_per_expansion']:.2f}")
        logger.info(f"  Coordination success rate: {metrics['coordination_success_rate']:.2f}")

        # 7. Get coordination summary
        coordination_summary = orchestrator.get_coordination_summary()
        logger.info(f"\nCoordination Summary:")
        logger.info(f"  Recent expansions: {coordination_summary['recent_expansions']}")
        logger.info(f"  Average combo nodes: {coordination_summary['average_combo_nodes']:.2f}")
        logger.info(f"  Agent usage: {coordination_summary['agent_usage']}")

        return orchestrator, search_context

    except Exception as e:
        logger.error(f"Tree expansion failed: {e}")
        raise


def demonstrate_reward_decomposition(orchestrator, search_context):
    """Demonstrate reward decomposition for tree GRPO training."""

    logger.info("\nDemonstrating reward decomposition for tree GRPO training...")

    # Get all nodes from search context
    all_nodes = list(search_context.node_registry.values())
    combination_nodes = [node for node in all_nodes if hasattr(node, 'agent_contributions')
                        and node.agent_contributions]

    if not combination_nodes:
        logger.warning("No combination nodes found for reward decomposition")
        return

    # Simulate final rewards for demonstration
    final_rewards = {}
    for i, node in enumerate(combination_nodes):
        # Simulate different reward values
        final_rewards[node.node_id] = 0.8 - (i * 0.1)  # Decreasing rewards

    logger.info(f"Simulated final rewards: {final_rewards}")

    # Decompose rewards to individual agents
    node_agent_rewards = orchestrator.decompose_rewards_for_tree(
        tree_nodes=combination_nodes,
        final_rewards=final_rewards
    )

    logger.info(f"\nReward decomposition results:")
    for node_id, agent_rewards in node_agent_rewards.items():
        logger.info(f"  Node {node_id}:")
        for agent_name, reward in agent_rewards.items():
            logger.info(f"    {agent_name}: {reward:.3f}")


def main():
    """Main example runner."""

    if not SMOLAGENTS_INTEGRATION_AVAILABLE:
        logger.error("Smolagents integration not available")
        return

    try:
        # Run the main example
        orchestrator, search_context = run_smolagents_tree_expansion_example()

        # Demonstrate reward decomposition
        demonstrate_reward_decomposition(orchestrator, search_context)

        logger.info("\n✅ Smolagents + Tree GRPO example completed successfully!")

    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise


if __name__ == "__main__":
    main()