#!/usr/bin/env python3

"""
Example: Math Problem Solving with Three-Agent Tree GRPO

This example demonstrates how to use three specialized agents for mathematical
problem solving with tree GRPO training:
- Math Analyzer: Analyzes problem type and strategy
- Math Solver: Generates solution steps and calculations
- Math Verifier: Verifies correctness and provides validation
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
from verl.experimental.agent_tree.reward_calculators import (
    CodeExecutionRewardCalculator, create_code_execution_calculator
)

if SMOLAGENTS_INTEGRATION_AVAILABLE:
    from verl.experimental.agent_tree import (
        SmolagentsTreeOrchestrator, SmolagentsTreeConfig,
        SmolagentsTreeCoordinator, MultiProposalGenerator
    )
    from smolagents import CodeAgent, ToolCallingAgent, LiteLLMModel
    from smolagents.tools import PythonInterpreterTool
else:
    print("Smolagents integration not available. Please install smolagents.")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_math_analyzer_agent(model):
    """Create a mathematical problem analysis agent."""
    return ToolCallingAgent(
        tools=[],  # No tools needed, pure analysis
        model=model,
        name="math_analyzer",
        description="Specializes in analyzing mathematical problems, identifying problem types, extracting key information, and suggesting solution strategies.",
        verbosity_level=0,  # Silent mode
        max_steps=3
    )


def create_math_solver_agent(model):
    """Create a mathematical solution generation agent."""
    return ToolCallingAgent(
        tools=[PythonInterpreterTool()],  # Can use Python for calculations
        model=model,
        name="math_solver",
        description="Specializes in generating mathematical solution steps, performing calculations, and providing detailed problem-solving procedures.",
        verbosity_level=0,  # Silent mode
        max_steps=5
    )


def create_math_verifier_agent(model):
    """Create a mathematical verification agent."""
    return ToolCallingAgent(
        tools=[PythonInterpreterTool()],  # Can use Python for verification
        model=model,
        name="math_verifier",
        description="Specializes in verifying mathematical solutions, checking calculation accuracy, and validating problem-solving logic.",
        verbosity_level=0,  # Silent mode
        max_steps=3
    )


def setup_math_coordinator_with_agents():
    """Set up the main coordinator agent with math-specialized sub-agents."""

    # Initialize model (replace with your preferred model)
    model = LiteLLMModel(
        model_id="openai/gpt-4",
        # Add your model configuration here
    )

    # Create specialized math agents
    analyzer_agent = create_math_analyzer_agent(model)
    solver_agent = create_math_solver_agent(model)
    verifier_agent = create_math_verifier_agent(model)

    # Create coordinator agent with math-focused managed agents
    coordinator = CodeAgent(
        tools=[],  # Coordinator focuses on coordination, managed agents do the work
        model=model,
        managed_agents=[analyzer_agent, solver_agent, verifier_agent],
        max_steps=10,
        verbosity_level=1,  # Some visibility for coordination decisions
        planning_interval=3,
        name="math_coordinator",
        description="Coordinates three specialized math agents for comprehensive problem solving: analysis, solution generation, and verification."
    )

    return coordinator


def create_math_problem_structure():
    """Create initial tree structure for mathematical problem solving."""

    # Create root node with a mathematical problem
    initial_messages = [
        {
            "role": "user",
            "content": """
            Solve the following quadratic equation:

            2x² - 7x + 3 = 0

            Please provide:
            1. Analysis of the problem type and approach
            2. Step-by-step solution process
            3. Verification of the answer

            Show all work and explain your reasoning.
            """
        }
    ]

    root_node = TreeGRPONode.from_prompt(
        messages=initial_messages,
        node_id="root_quadratic_equation"
    )

    # Create search context for math problem solving
    search_context = SearchContext(
        max_depth=4,
        max_branches_per_node=8,  # 3 agents * 2-3 proposals each
        max_active_nodes=20,
        problem_type="mathematical_solving",
        difficulty_level=0.6
    )

    search_context.register_node(root_node)

    return root_node, search_context


def run_math_solving_tree_expansion():
    """Run a complete math problem solving example with tree expansion."""

    logger.info("🧮 Starting Math Problem Solving with Three-Agent Tree GRPO")

    # 1. Set up coordinator with math-specialized agents
    logger.info("Setting up math coordinator with specialized agents...")
    coordinator = setup_math_coordinator_with_agents()

    # 2. Create math-focused tree configuration
    config = SmolagentsTreeConfig(
        coordination_strategy="hierarchical",
        proposals_per_agent=2,  # Each agent generates 2 approaches
        enable_memory_isolation=True,
        fusion_strategy="structured_combination",
        enable_silent_mode=True
    )

    # 3. Initialize orchestrator
    orchestrator = SmolagentsTreeOrchestrator(
        coordinator_agent=coordinator,
        config=config
    )

    # 4. Create math problem tree structure
    root_node, search_context = create_math_problem_structure()

    logger.info(f"Created root node: {root_node.node_id}")
    logger.info(f"Available math agents: {list(coordinator.managed_agents.keys())}")

    # 5. Perform tree expansion with math agents
    logger.info("Expanding tree with math-specialized multi-agent coordination...")

    try:
        # First level expansion - different approaches to the quadratic equation
        level_1_nodes = orchestrator.expand_tree_node(root_node, search_context)
        logger.info(f"Level 1 expansion created {len(level_1_nodes)} combination nodes")

        # Display math solving approaches
        for i, node in enumerate(level_1_nodes):
            logger.info(f"\n📐 Math Solution Approach {i+1}:")
            if hasattr(node, 'agent_contributions'):
                for agent_name, contribution in node.agent_contributions.items():
                    preview = contribution.output_text[:150].replace('\n', ' ')
                    logger.info(f"  {agent_name}: {preview}...")

            solution_preview = node.local_generation_text[:200].replace('\n', ' ')
            logger.info(f"  Combined approach: {solution_preview}...")

        # 6. Get performance metrics
        metrics = orchestrator.get_performance_metrics()
        logger.info(f"\n📊 Performance Metrics:")
        logger.info(f"  Total expansions: {metrics['total_expansions']}")
        logger.info(f"  Successful expansions: {metrics['successful_expansions']}")
        logger.info(f"  Average combination nodes per expansion: {metrics['average_combo_nodes_per_expansion']:.2f}")
        logger.info(f"  Coordination success rate: {metrics['coordination_success_rate']:.2f}")

        # 7. Demonstrate second level expansion on best approach
        if level_1_nodes:
            # For demo, select first node for further expansion
            selected_node = level_1_nodes[0]
            logger.info(f"\n🔍 Expanding selected approach: {selected_node.node_id}")

            level_2_nodes = orchestrator.expand_tree_node(selected_node, search_context)
            logger.info(f"Level 2 expansion created {len(level_2_nodes)} refined solution nodes")

        return orchestrator, search_context

    except Exception as e:
        logger.error(f"Math solving tree expansion failed: {e}")
        raise


def demonstrate_math_solution_evaluation(orchestrator, search_context):
    """Demonstrate evaluation of mathematical solutions."""

    logger.info("\n🎯 Demonstrating Mathematical Solution Evaluation...")

    # Get all combination nodes from search context
    all_nodes = list(search_context.node_registry.values())
    math_solution_nodes = [node for node in all_nodes if hasattr(node, 'agent_contributions')
                          and node.agent_contributions]

    if not math_solution_nodes:
        logger.warning("No math solution nodes found for evaluation")
        return

    # Expected solution for the quadratic equation 2x² - 7x + 3 = 0
    # Using quadratic formula: x = (7 ± √(49-24))/4 = (7 ± 5)/4
    # So x = 3 or x = 0.5
    expected_solutions = {"x1": 3.0, "x2": 0.5}

    logger.info(f"Expected solutions: x = {expected_solutions['x1']} or x = {expected_solutions['x2']}")

    # Simulate evaluation of different solution approaches
    solution_evaluations = {}
    for i, node in enumerate(math_solution_nodes):
        # Simulate different quality scores based on approach completeness
        base_score = 0.7  # Base score for attempting the problem

        # Bonus for including analysis
        if any("quadratic" in contrib.output_text.lower() or "formula" in contrib.output_text.lower()
               for contrib in node.agent_contributions.values() if contrib.output_text):
            base_score += 0.1

        # Bonus for including verification
        if any("verify" in contrib.output_text.lower() or "check" in contrib.output_text.lower()
               for contrib in node.agent_contributions.values() if contrib.output_text):
            base_score += 0.1

        # Add some variation
        variation = 0.1 * (i % 3 - 1)  # -0.1, 0, +0.1
        final_score = min(1.0, max(0.0, base_score + variation))

        solution_evaluations[node.node_id] = final_score

    logger.info(f"\n📈 Math Solution Evaluation Results:")
    for node_id, score in solution_evaluations.items():
        node = next(n for n in math_solution_nodes if n.node_id == node_id)
        logger.info(f"  {node_id}: {score:.3f}")

        # Show which agents contributed
        if hasattr(node, 'agent_contributions'):
            contributors = list(node.agent_contributions.keys())
            logger.info(f"    Contributors: {', '.join(contributors)}")

    # Find best performing approach
    best_node_id = max(solution_evaluations, key=solution_evaluations.get)
    best_score = solution_evaluations[best_node_id]

    logger.info(f"\n🏆 Best math solution approach: {best_node_id} (score: {best_score:.3f})")

    return solution_evaluations


def main():
    """Main math solving example runner."""

    if not SMOLAGENTS_INTEGRATION_AVAILABLE:
        logger.error("Smolagents integration not available")
        return

    try:
        # Run the math solving tree expansion example
        orchestrator, search_context = run_math_solving_tree_expansion()

        # Demonstrate solution evaluation
        solution_evaluations = demonstrate_math_solution_evaluation(orchestrator, search_context)

        # Get coordination summary
        coordination_summary = orchestrator.get_coordination_summary()
        logger.info(f"\n📋 Math Coordination Summary:")
        logger.info(f"  Total expansions: {coordination_summary['total_expansions']}")
        logger.info(f"  Agent usage patterns: {coordination_summary['agent_usage']}")

        logger.info("\n✅ Math Problem Solving with Three-Agent Tree GRPO completed successfully!")
        logger.info("🧮 Demonstrated: Analysis → Solution → Verification workflow")

    except Exception as e:
        logger.error(f"❌ Math solving example failed: {e}")
        raise


if __name__ == "__main__":
    main()