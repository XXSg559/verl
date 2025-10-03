#!/usr/bin/env python3

"""
Code Generation Training with Smolagents + Tree GRPO

This module demonstrates how to set up code generation training using our
smolagents + tree GRPO integration with multi-agent collaboration.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import subprocess
import tempfile

# Add verl to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from verl.experimental.agent_tree import (
    TreeGRPONode, SearchContext,
    SMOLAGENTS_INTEGRATION_AVAILABLE
)

if SMOLAGENTS_INTEGRATION_AVAILABLE:
    from verl.experimental.agent_tree import (
        SmolagentsTreeOrchestrator, SmolagentsTreeConfig
    )
    from smolagents import CodeAgent, ToolCallingAgent, LiteLLMModel
    from smolagents.tools import PythonInterpreterTool
else:
    print("Smolagents integration not available")
    sys.exit(1)

logger = logging.getLogger(__name__)


@dataclass
class CodeGenerationTask:
    """Data structure for code generation training tasks."""

    # Problem definition
    problem_id: str
    problem_description: str
    function_signature: str
    requirements: List[str]
    constraints: List[str]

    # Test cases
    test_cases: List[Dict[str, Any]]  # {"input": {...}, "expected_output": ...}
    edge_cases: List[Dict[str, Any]]
    performance_tests: List[Dict[str, Any]]  # For complexity testing

    # Reference solutions (optional)
    reference_solutions: List[str]

    # Evaluation criteria
    correctness_weight: float = 0.4
    efficiency_weight: float = 0.2
    readability_weight: float = 0.2
    robustness_weight: float = 0.2

    # Metadata
    difficulty_level: float  # 0.0 to 1.0
    time_complexity_target: str  # "O(n)", "O(log n)", etc.
    space_complexity_target: str
    programming_concepts: List[str]  # ["dynamic_programming", "graph_theory", etc.]


class CodeExecutionReward:
    """Reward calculator for code generation tasks."""

    def __init__(self):
        self.execution_timeout = 5.0  # seconds

    def evaluate_code_solution(
        self,
        code: str,
        task: CodeGenerationTask
    ) -> Dict[str, float]:
        """Evaluate a code solution and return detailed rewards."""

        rewards = {
            "correctness": 0.0,
            "efficiency": 0.0,
            "readability": 0.0,
            "robustness": 0.0,
            "overall": 0.0
        }

        try:
            # 1. Correctness evaluation
            correctness_score = self._evaluate_correctness(code, task)
            rewards["correctness"] = correctness_score

            # 2. Efficiency evaluation (only if code is correct)
            if correctness_score > 0.8:
                efficiency_score = self._evaluate_efficiency(code, task)
                rewards["efficiency"] = efficiency_score

            # 3. Readability evaluation
            readability_score = self._evaluate_readability(code)
            rewards["readability"] = readability_score

            # 4. Robustness evaluation
            robustness_score = self._evaluate_robustness(code, task)
            rewards["robustness"] = robustness_score

            # 5. Overall weighted score
            rewards["overall"] = (
                rewards["correctness"] * task.correctness_weight +
                rewards["efficiency"] * task.efficiency_weight +
                rewards["readability"] * task.readability_weight +
                rewards["robustness"] * task.robustness_weight
            )

        except Exception as e:
            logger.error(f"Error evaluating code: {e}")
            # Penalize code that fails to evaluate
            rewards["overall"] = -0.5

        return rewards

    def _evaluate_correctness(self, code: str, task: CodeGenerationTask) -> float:
        """Evaluate correctness by running test cases."""
        if not task.test_cases:
            return 0.5  # No test cases available

        passed_tests = 0
        total_tests = len(task.test_cases)

        for test_case in task.test_cases:
            try:
                if self._run_test_case(code, test_case):
                    passed_tests += 1
            except Exception:
                continue  # Test failed

        return passed_tests / max(total_tests, 1)

    def _run_test_case(self, code: str, test_case: Dict[str, Any]) -> bool:
        """Run a single test case."""
        try:
            # Create a temporary file with the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                test_code = f"""
{code}

# Test execution
try:
    input_data = {test_case['input']}
    if isinstance(input_data, dict):
        result = {task.function_signature.split('(')[0]}(**input_data)
    else:
        result = {task.function_signature.split('(')[0]}(*input_data)

    expected = {test_case['expected_output']}
    print("RESULT:", result)
    print("EXPECTED:", expected)
    print("MATCH:", result == expected)
except Exception as e:
    print("ERROR:", str(e))
"""
                f.write(test_code)
                f.flush()

                # Run the test
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.execution_timeout
                )

                # Check if test passed
                output = result.stdout
                return "MATCH: True" in output

        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False
        finally:
            # Clean up temporary file
            try:
                os.unlink(f.name)
            except:
                pass

    def _evaluate_efficiency(self, code: str, task: CodeGenerationTask) -> float:
        """Evaluate code efficiency (simplified version)."""
        # Simplified efficiency metrics
        efficiency_score = 0.5  # Base score

        # Check for common efficiency patterns
        if "for" in code and "while" not in code:
            efficiency_score += 0.1  # Single loop preferred over nested

        if "dict" in code or "set" in code:
            efficiency_score += 0.2  # Good data structure usage

        if len(code.split('\n')) < 20:  # Concise solution bonus
            efficiency_score += 0.1

        # Check against complexity targets
        if task.time_complexity_target == "O(n)" and "sorted" not in code:
            efficiency_score += 0.1

        return min(efficiency_score, 1.0)

    def _evaluate_readability(self, code: str) -> float:
        """Evaluate code readability."""
        readability_score = 0.5  # Base score

        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]

        # Check for comments
        comment_lines = [line for line in lines if line.strip().startswith('#')]
        if comment_lines:
            readability_score += 0.2

        # Check for descriptive variable names
        if any(len(word) > 3 for line in non_empty_lines for word in line.split()
               if word.isidentifier()):
            readability_score += 0.2

        # Check for reasonable line length
        if all(len(line) < 100 for line in lines):
            readability_score += 0.1

        return min(readability_score, 1.0)

    def _evaluate_robustness(self, code: str, task: CodeGenerationTask) -> float:
        """Evaluate code robustness with edge cases."""
        if not task.edge_cases:
            return 0.5  # No edge cases to test

        passed_edge_cases = 0
        total_edge_cases = len(task.edge_cases)

        for edge_case in task.edge_cases:
            try:
                if self._run_test_case(code, edge_case):
                    passed_edge_cases += 1
            except Exception:
                continue

        return passed_edge_cases / max(total_edge_cases, 1)


def create_code_generation_agents(model):
    """Create specialized agents for code generation."""

    # Algorithm design agent
    algorithm_agent = ToolCallingAgent(
        tools=[],
        model=model,
        name="algorithm_agent",
        description="Specializes in algorithm design, problem decomposition, and complexity analysis.",
        verbosity_level=0,
        max_steps=3
    )

    # Implementation agent
    implementation_agent = ToolCallingAgent(
        tools=[PythonInterpreterTool()],
        model=model,
        name="implementation_agent",
        description="Specializes in writing clean, efficient Python code implementations.",
        verbosity_level=0,
        max_steps=5
    )

    # Testing agent
    testing_agent = ToolCallingAgent(
        tools=[PythonInterpreterTool()],
        model=model,
        name="testing_agent",
        description="Specializes in creating test cases, debugging, and code validation.",
        verbosity_level=0,
        max_steps=4
    )

    # Optimization agent
    optimization_agent = ToolCallingAgent(
        tools=[PythonInterpreterTool()],
        model=model,
        name="optimization_agent",
        description="Specializes in code optimization, performance improvement, and refactoring.",
        verbosity_level=0,
        max_steps=4
    )

    return [algorithm_agent, implementation_agent, testing_agent, optimization_agent]


def setup_code_generation_coordinator():
    """Set up coordinator for code generation tasks."""

    # Initialize model
    model = LiteLLMModel(
        model_id="openai/gpt-4",
        temperature=0.7,
        max_tokens=2048
    )

    # Create specialized agents
    agents = create_code_generation_agents(model)

    # Create coordinator
    coordinator = CodeAgent(
        tools=[PythonInterpreterTool()],  # Coordinator can also execute code for validation
        model=model,
        managed_agents=agents,
        name="code_coordinator",
        description="Coordinates multiple agents for comprehensive code generation and validation.",
        max_steps=8,
        verbosity_level=1
    )

    return coordinator


def create_sample_training_data() -> List[CodeGenerationTask]:
    """Create sample training data for code generation."""

    tasks = [
        CodeGenerationTask(
            problem_id="two_sum",
            problem_description="Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
            function_signature="def two_sum(nums: List[int], target: int) -> List[int]:",
            requirements=[
                "Return indices, not the values",
                "You may assume exactly one solution exists",
                "Cannot use the same element twice"
            ],
            constraints=[
                "2 <= nums.length <= 10^4",
                "-10^9 <= nums[i] <= 10^9",
                "-10^9 <= target <= 10^9"
            ],
            test_cases=[
                {"input": {"nums": [2, 7, 11, 15], "target": 9}, "expected_output": [0, 1]},
                {"input": {"nums": [3, 2, 4], "target": 6}, "expected_output": [1, 2]},
                {"input": {"nums": [3, 3], "target": 6}, "expected_output": [0, 1]}
            ],
            edge_cases=[
                {"input": {"nums": [-1, -2, -3, -4, -5], "target": -8}, "expected_output": [2, 4]},
                {"input": {"nums": [0, 4, 3, 0], "target": 0}, "expected_output": [0, 3]}
            ],
            reference_solutions=[
                """def two_sum(nums, target):
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []"""
            ],
            difficulty_level=0.3,
            time_complexity_target="O(n)",
            space_complexity_target="O(n)",
            programming_concepts=["hash_map", "array_traversal"]
        ),

        CodeGenerationTask(
            problem_id="binary_search",
            problem_description="Implement binary search algorithm to find target value in sorted array.",
            function_signature="def binary_search(arr: List[int], target: int) -> int:",
            requirements=[
                "Return index of target if found, -1 otherwise",
                "Array is sorted in ascending order",
                "Use O(log n) time complexity"
            ],
            constraints=[
                "0 <= arr.length <= 10^4",
                "-10^4 <= arr[i], target <= 10^4"
            ],
            test_cases=[
                {"input": {"arr": [1, 3, 5, 7, 9], "target": 5}, "expected_output": 2},
                {"input": {"arr": [1, 3, 5, 7, 9], "target": 6}, "expected_output": -1},
                {"input": {"arr": [], "target": 1}, "expected_output": -1}
            ],
            edge_cases=[
                {"input": {"arr": [1], "target": 1}, "expected_output": 0},
                {"input": {"arr": [1, 2], "target": 2}, "expected_output": 1}
            ],
            difficulty_level=0.4,
            time_complexity_target="O(log n)",
            space_complexity_target="O(1)",
            programming_concepts=["binary_search", "divide_and_conquer"]
        )
    ]

    return tasks


def run_code_generation_training_example():
    """Run a complete code generation training example."""

    logger.info("Starting code generation training with smolagents + tree GRPO")

    # 1. Setup coordinator and agents
    coordinator = setup_code_generation_coordinator()

    # 2. Create orchestrator
    config = SmolagentsTreeConfig(
        proposals_per_agent=2,  # Each agent generates 2 code solutions
        fusion_strategy="structured_combination",
        enable_silent_mode=True
    )

    orchestrator = SmolagentsTreeOrchestrator(coordinator, config)

    # 3. Create sample training data
    training_tasks = create_sample_training_data()

    # 4. Setup reward calculator
    reward_calculator = CodeExecutionReward()

    # Process first task as example
    task = training_tasks[0]
    logger.info(f"Processing task: {task.problem_id}")

    # 5. Create initial tree node
    initial_messages = [
        {
            "role": "user",
            "content": f"""
Problem: {task.problem_description}

Function Signature: {task.function_signature}

Requirements:
{chr(10).join('- ' + req for req in task.requirements)}

Constraints:
{chr(10).join('- ' + constraint for constraint in task.constraints)}

Test Cases:
{json.dumps(task.test_cases, indent=2)}

Please provide multiple high-quality code solutions with different approaches.
"""
        }
    ]

    root_node = TreeGRPONode.from_prompt(messages=initial_messages)
    search_context = SearchContext(
        max_depth=3,
        max_branches_per_node=8,  # 4 agents * 2 proposals = 8 combinations
        problem_type="code_generation"
    )

    # 6. Expand tree and generate code solutions
    logger.info("Expanding tree with multi-agent code generation...")
    level_1_nodes = orchestrator.expand_tree_node(root_node, search_context)

    logger.info(f"Generated {len(level_1_nodes)} combination solutions")

    # 7. Evaluate solutions and compute rewards
    solution_rewards = {}
    for i, node in enumerate(level_1_nodes):
        logger.info(f"\nEvaluating solution {i+1}:")

        # Extract code from combined output
        code_solution = extract_code_from_node_output(node.local_generation_text)

        if code_solution:
            # Evaluate the code
            rewards = reward_calculator.evaluate_code_solution(code_solution, task)
            solution_rewards[node.node_id] = rewards["overall"]

            logger.info(f"  Correctness: {rewards['correctness']:.3f}")
            logger.info(f"  Efficiency: {rewards['efficiency']:.3f}")
            logger.info(f"  Readability: {rewards['readability']:.3f}")
            logger.info(f"  Robustness: {rewards['robustness']:.3f}")
            logger.info(f"  Overall: {rewards['overall']:.3f}")

            # Show agent contributions
            if hasattr(node, 'agent_contributions'):
                logger.info("  Agent contributions:")
                for agent_name, contribution in node.agent_contributions.items():
                    logger.info(f"    {agent_name}: {contribution.output_text[:100]}...")
        else:
            solution_rewards[node.node_id] = -1.0  # Invalid solution
            logger.info("  No valid code solution found")

    # 8. Decompose rewards to agents
    agent_rewards = orchestrator.decompose_rewards_for_tree(
        tree_nodes=level_1_nodes,
        final_rewards=solution_rewards
    )

    logger.info(f"\nAgent reward decomposition:")
    for node_id, rewards in agent_rewards.items():
        logger.info(f"  Node {node_id}:")
        for agent_name, reward in rewards.items():
            logger.info(f"    {agent_name}: {reward:.3f}")

    # 9. Show best solution
    if solution_rewards:
        best_node_id = max(solution_rewards.keys(), key=lambda k: solution_rewards[k])
        best_node = next(node for node in level_1_nodes if node.node_id == best_node_id)
        best_code = extract_code_from_node_output(best_node.local_generation_text)

        logger.info(f"\nBest solution (reward: {solution_rewards[best_node_id]:.3f}):")
        logger.info(f"```python\n{best_code}\n```")

    return orchestrator, solution_rewards, agent_rewards


def extract_code_from_node_output(output_text: str) -> Optional[str]:
    """Extract Python code from node output text."""

    # Look for code blocks
    lines = output_text.split('\n')
    code_lines = []
    in_code_block = False

    for line in lines:
        if line.strip().startswith('```python') or line.strip().startswith('```'):
            in_code_block = True
            continue
        elif line.strip() == '```' and in_code_block:
            break
        elif in_code_block:
            code_lines.append(line)
        elif line.strip().startswith('def ') and not in_code_block:
            # Found function definition outside code block
            code_lines = [line]
            # Continue collecting until we find the end of function
            in_function = True

    code = '\n'.join(code_lines).strip()

    # Basic validation - must contain a function definition
    if 'def ' in code:
        return code

    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        orchestrator, solution_rewards, agent_rewards = run_code_generation_training_example()
        logger.info("\n✅ Code generation training example completed successfully!")

    except Exception as e:
        logger.error(f"❌ Training example failed: {e}")
        raise