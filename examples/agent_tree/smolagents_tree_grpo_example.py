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
from verl.experimental.agent_tree.reward_calculators import (
    CodeExecutionRewardCalculator, create_code_execution_calculator
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


def demonstrate_execution_rewards():
    """Demonstrate execution-based reward calculation for code generation."""

    logger.info("\n🧮 Demonstrating execution-based reward calculation...")

    # Create a code execution reward calculator
    reward_calculator = create_code_execution_calculator(
        execution_timeout=5.0,
        execution_success_reward=0.5,
        correctness_reward=0.5,
        syntax_error_penalty=-0.3,
        authorized_imports=['math', 'random', 'datetime']
    )

    logger.info("Created CodeExecutionRewardCalculator")

    # Create example code generation task
    initial_messages = [
        {
            "role": "user",
            "content": """
            Write Python code to calculate the fibonacci number for n=10.
            The code should print the result.
            """
        }
    ]

    root_node = TreeGRPONode.from_prompt(
        messages=initial_messages,
        node_id="root_fibonacci"
    )

    # Create mock child nodes with different code outputs
    test_nodes = []

    # Node 1: Correct fibonacci implementation
    node1 = TreeGRPONode(
        parent_id=root_node.node_id,
        depth=1,
        messages=root_node.messages + [{
            "role": "assistant",
            "content": """Here's the fibonacci code:
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
print(result)
```
"""
        }],
        agent_name="coding_agent",
        branch_reason="recursive_approach",
        local_generation_text="""Here's the fibonacci code:
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
print(result)
```
""",
        node_id="fib_recursive"
    )
    test_nodes.append(node1)

    # Node 2: Iterative approach (also correct)
    node2 = TreeGRPONode(
        parent_id=root_node.node_id,
        depth=1,
        messages=root_node.messages + [{
            "role": "assistant",
            "content": """Here's an iterative fibonacci implementation:
```python
def fibonacci_iter(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

result = fibonacci_iter(10)
print(result)
```
"""
        }],
        agent_name="coding_agent",
        branch_reason="iterative_approach",
        local_generation_text="""Here's an iterative fibonacci implementation:
```python
def fibonacci_iter(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

result = fibonacci_iter(10)
print(result)
```
""",
        node_id="fib_iterative"
    )
    test_nodes.append(node2)

    # Node 3: Incorrect implementation (syntax error)
    node3 = TreeGRPONode(
        parent_id=root_node.node_id,
        depth=1,
        messages=root_node.messages + [{
            "role": "assistant",
            "content": """Here's fibonacci code with an error:
```python
def fibonacci(n):
    if n <= 1
        return n  # Missing colon!
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
print(result)
```
"""
        }],
        agent_name="coding_agent",
        branch_reason="error_approach",
        local_generation_text="""Here's fibonacci code with an error:
```python
def fibonacci(n):
    if n <= 1
        return n  # Missing colon!
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
print(result)
```
""",
        node_id="fib_error"
    )
    test_nodes.append(node3)

    # Node 4: Wrong logic but syntactically correct
    node4 = TreeGRPONode(
        parent_id=root_node.node_id,
        depth=1,
        messages=root_node.messages + [{
            "role": "assistant",
            "content": """Here's fibonacci code that's wrong:
```python
def fibonacci(n):
    return n * 2  # Wrong logic!

result = fibonacci(10)
print(result)
```
"""
        }],
        agent_name="coding_agent",
        branch_reason="wrong_logic",
        local_generation_text="""Here's fibonacci code that's wrong:
```python
def fibonacci(n):
    return n * 2  # Wrong logic!

result = fibonacci(10)
print(result)
```
""",
        node_id="fib_wrong"
    )
    test_nodes.append(node4)

    # Expected result for fibonacci(10)
    expected_results = {
        "fib_recursive": 55,
        "fib_iterative": 55,
        "fib_error": 55,
        "fib_wrong": 55
    }

    logger.info(f"Testing {len(test_nodes)} different code implementations")

    # Calculate rewards for each node
    from verl.experimental.agent_tree.reward_calculators import TreeGRPORewardCalculator
    tree_calculator = TreeGRPORewardCalculator(reward_calculator)

    node_rewards = tree_calculator.calculate_node_rewards(
        tree_nodes=test_nodes,
        expected_results=expected_results
    )

    logger.info("\n📊 Execution reward results:")
    for node_id, reward in node_rewards.items():
        node = next(n for n in test_nodes if n.node_id == node_id)
        logger.info(f"  {node_id}: {reward:.3f} ({node.branch_reason})")

    # Get performance summary
    performance_summary = tree_calculator.get_performance_summary(node_rewards)
    logger.info(f"\n📈 Performance summary: {performance_summary}")

    return node_rewards


def run_code_generation_training_example():
    """Run a complete code generation training example with execution rewards."""

    logger.info("🚀 Starting Code Generation Training Example with Execution Rewards")

    # 1. Set up coordinator with coding-focused agents
    coordinator = setup_coordinator_with_managed_agents()

    # 2. Create execution reward calculator
    execution_calculator = create_code_execution_calculator(
        execution_timeout=10.0,
        execution_success_reward=0.6,
        correctness_reward=0.4,
        syntax_error_penalty=-0.4,
        runtime_error_penalty=-0.3,
        authorized_imports=['math', 'random', 'itertools', 'collections']
    )

    # 3. Create configuration with execution rewards enabled
    config = SmolagentsTreeConfig(
        coordination_strategy="hierarchical",
        proposals_per_agent=2,
        enable_memory_isolation=True,
        fusion_strategy="structured_combination",
        enable_silent_mode=True,
        enable_execution_rewards=True,
        reward_calculator=execution_calculator
    )

    # 4. Initialize orchestrator with reward calculation
    orchestrator = SmolagentsTreeOrchestrator(
        coordinator_agent=coordinator,
        config=config
    )

    logger.info("✅ Orchestrator configured with execution rewards")

    # 5. Create code generation task
    code_gen_messages = [
        {
            "role": "user",
            "content": """
            Write Python code to solve this problem:

            Calculate the sum of all prime numbers less than 100.
            Print the result.

            Requirements:
            - Implement a function to check if a number is prime
            - Use this function to find all primes < 100
            - Sum them and print the total
            """
        }
    ]

    root_node = TreeGRPONode.from_prompt(
        messages=code_gen_messages,
        node_id="root_prime_sum"
    )

    search_context = SearchContext(
        max_depth=3,
        max_branches_per_node=4,
        max_active_nodes=15,
        problem_type="code_generation"
    )

    search_context.register_node(root_node)

    # 6. Perform tree expansion
    logger.info("🌳 Expanding tree with multi-agent code generation...")
    level_1_nodes = orchestrator.expand_tree_node(root_node, search_context)

    logger.info(f"Generated {len(level_1_nodes)} first-level combination nodes")

    # 7. Evaluate nodes with execution rewards
    qa_pairs = {}
    expected_prime_sum = 1060  # Sum of primes < 100

    for node in level_1_nodes:
        qa_pairs[node.node_id] = {
            "question": "Sum of primes less than 100",
            "expected_answer": expected_prime_sum
        }

    evaluation_results = orchestrator.evaluate_tree_with_execution_rewards(
        tree_nodes=level_1_nodes,
        qa_pairs=qa_pairs
    )

    logger.info("\n🎯 Code Generation Evaluation Results:")
    logger.info(f"Performance: {evaluation_results['performance_summary']}")
    logger.info(f"Tree Analysis: {evaluation_results['tree_analysis']}")

    # 8. Find best performing node for further expansion
    best_node_id = max(evaluation_results['node_rewards'],
                      key=evaluation_results['node_rewards'].get)
    best_node = next(n for n in level_1_nodes if n.node_id == best_node_id)
    best_reward = evaluation_results['node_rewards'][best_node_id]

    logger.info(f"\n🏆 Best performing node: {best_node_id} (reward: {best_reward:.3f})")
    logger.info(f"Best node output preview: {best_node.local_generation_text[:200]}...")

    return orchestrator, evaluation_results


def main():
    """Main example runner with execution rewards demonstration."""

    if not SMOLAGENTS_INTEGRATION_AVAILABLE:
        logger.error("Smolagents integration not available")
        return

    try:
        # Demonstrate basic execution rewards
        logger.info("=" * 60)
        demonstrate_execution_rewards()

        # Run the original smolagents tree expansion example
        logger.info("=" * 60)
        orchestrator, search_context = run_smolagents_tree_expansion_example()

        # Demonstrate reward decomposition
        logger.info("=" * 60)
        demonstrate_reward_decomposition(orchestrator, search_context)

        # Run the new code generation training example with execution rewards
        logger.info("=" * 60)
        code_orchestrator, evaluation_results = run_code_generation_training_example()

        logger.info("\n🎉 All examples completed successfully!")
        logger.info("✅ Smolagents + Tree GRPO integration with execution rewards is working!")

    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise


if __name__ == "__main__":
    main()