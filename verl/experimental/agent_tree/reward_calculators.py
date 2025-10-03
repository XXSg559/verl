#!/usr/bin/env python3

"""
Reward calculators for agent tree training.

This module provides different reward calculation strategies for tree-based
GRPO training, with a focus on code generation tasks using execution results.
"""

import re
import time
import logging
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    stdout: Optional[str] = None


@dataclass
class RewardResult:
    """Result of reward calculation."""
    total_reward: float
    execution_reward: float = 0.0
    correctness_reward: float = 0.0
    error_penalty: float = 0.0
    timeout_penalty: float = 0.0
    details: Dict[str, Any] = None


class BaseRewardCalculator(ABC):
    """Base class for reward calculators."""

    @abstractmethod
    def calculate_reward(self, agent_output: str, expected_result: Any, **kwargs) -> RewardResult:
        """Calculate reward for agent output."""
        pass


class CodeExecutionRewardCalculator(BaseRewardCalculator):
    """
    Reward calculator based on Python code execution results.

    This calculator:
    1. Extracts Python code from agent output
    2. Executes the code safely
    3. Compares execution results with expected output
    4. Assigns rewards based on execution success and correctness
    """

    def __init__(
        self,
        execution_timeout: float = 10.0,
        execution_success_reward: float = 0.5,
        correctness_reward: float = 0.5,
        syntax_error_penalty: float = -0.3,
        runtime_error_penalty: float = -0.3,
        timeout_penalty: float = -0.2,
        tolerance: float = 1e-6,
        max_code_length: int = 10000,
        authorized_imports: Optional[List[str]] = None
    ):
        """
        Initialize code execution reward calculator.

        Args:
            execution_timeout: Maximum time allowed for code execution
            execution_success_reward: Reward for successful code execution
            correctness_reward: Reward for correct output
            syntax_error_penalty: Penalty for syntax errors
            runtime_error_penalty: Penalty for runtime errors
            timeout_penalty: Penalty for execution timeout
            tolerance: Tolerance for numerical comparisons
            max_code_length: Maximum allowed code length
            authorized_imports: List of authorized import modules
        """
        self.execution_timeout = execution_timeout
        self.execution_success_reward = execution_success_reward
        self.correctness_reward = correctness_reward
        self.syntax_error_penalty = syntax_error_penalty
        self.runtime_error_penalty = runtime_error_penalty
        self.timeout_penalty = timeout_penalty
        self.tolerance = tolerance
        self.max_code_length = max_code_length
        self.authorized_imports = authorized_imports or []

        # Initialize Python interpreter tool
        self._init_python_tool()

    def _init_python_tool(self):
        """Initialize Python interpreter tool."""
        try:
            # Import smolagents tools
            from smolagents.default_tools import PythonInterpreterTool
            self.python_tool = PythonInterpreterTool(
                authorized_imports=self.authorized_imports
            )
            logger.info("Initialized PythonInterpreterTool for reward calculation")
        except ImportError as e:
            logger.warning(f"Could not import smolagents: {e}. Using fallback execution.")
            self.python_tool = None

    def extract_python_code(self, text: str) -> List[str]:
        """
        Extract Python code blocks from text.

        Args:
            text: Input text that may contain code blocks

        Returns:
            List of extracted code strings
        """
        # Pattern for code blocks with ```python or ```
        code_patterns = [
            r'```python\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
            r'<code>(.*?)</code>',
            r'`([^`\n]+)`'  # Inline code
        ]

        extracted_codes = []

        for pattern in code_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                code = match.strip()
                if code and len(code) <= self.max_code_length:
                    # Basic validation - should contain some Python-like syntax
                    if any(keyword in code for keyword in ['print', 'def', 'import', 'for', 'if', 'while', '=']):
                        extracted_codes.append(code)

        # If no code blocks found, try to extract code from the entire text
        if not extracted_codes:
            # Look for lines that look like Python code
            lines = text.split('\n')
            code_lines = []
            for line in lines:
                line = line.strip()
                if line and (
                    line.startswith(('print(', 'def ', 'import ', 'from ')) or
                    '=' in line or
                    line.startswith(('if ', 'for ', 'while ', 'try:'))
                ):
                    code_lines.append(line)

            if code_lines:
                code = '\n'.join(code_lines)
                if len(code) <= self.max_code_length:
                    extracted_codes.append(code)

        return extracted_codes

    def execute_code_safely(self, code: str) -> ExecutionResult:
        """
        Execute Python code safely and return results.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with execution details
        """
        start_time = time.time()

        try:
            if self.python_tool:
                # Use smolagents PythonInterpreterTool
                result = self.python_tool(code)
                execution_time = time.time() - start_time

                # Parse the result - PythonInterpreterTool returns "Stdout:\n...\nOutput: ..."
                if "Stdout:\n" in result and "\nOutput: " in result:
                    parts = result.split("\nOutput: ", 1)
                    stdout = parts[0].replace("Stdout:\n", "")
                    output = parts[1] if len(parts) > 1 else ""
                else:
                    stdout = ""
                    output = result

                return ExecutionResult(
                    success=True,
                    output=output.strip(),
                    stdout=stdout.strip(),
                    execution_time=execution_time
                )
            else:
                # Fallback: use exec (less safe)
                return self._execute_with_exec(code, start_time)

        except SyntaxError as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                error=f"SyntaxError: {str(e)}",
                execution_time=execution_time
            )
        except TimeoutError:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                error="Execution timeout",
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                execution_time=execution_time
            )

    def _execute_with_exec(self, code: str, start_time: float) -> ExecutionResult:
        """Fallback execution using exec (less safe)."""
        import io
        import sys
        from contextlib import redirect_stdout

        # Capture stdout
        stdout_capture = io.StringIO()

        try:
            # Create execution namespace
            namespace = {
                '__builtins__': __builtins__,
                'print': print
            }

            with redirect_stdout(stdout_capture):
                exec(code, namespace)

            execution_time = time.time() - start_time
            stdout_content = stdout_capture.getvalue()

            # Try to get the last expression result if no explicit return
            output = stdout_content.strip()

            return ExecutionResult(
                success=True,
                output=output,
                stdout=stdout_content,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                execution_time=execution_time
            )

    def compare_results(self, actual: str, expected: Any) -> Tuple[bool, float]:
        """
        Compare actual execution result with expected result.

        Args:
            actual: Actual execution output
            expected: Expected result (can be string, number, list, etc.)

        Returns:
            Tuple of (is_correct, similarity_score)
        """
        if actual is None:
            return False, 0.0

        # Convert expected to string for comparison
        if isinstance(expected, (int, float)):
            expected_str = str(expected)
        elif isinstance(expected, (list, tuple)):
            expected_str = str(expected)
        else:
            expected_str = str(expected)

        actual_clean = actual.strip()
        expected_clean = expected_str.strip()

        # Exact match
        if actual_clean == expected_clean:
            return True, 1.0

        # Try numerical comparison if both look like numbers
        try:
            actual_num = float(actual_clean)
            expected_num = float(expected_clean)
            if abs(actual_num - expected_num) <= self.tolerance:
                return True, 1.0
        except ValueError:
            pass

        # Partial match - check if expected is contained in actual
        if expected_clean.lower() in actual_clean.lower():
            return True, 0.7

        # Character-level similarity (simple approach)
        if len(expected_clean) > 0:
            common_chars = sum(1 for a, e in zip(actual_clean, expected_clean) if a == e)
            similarity = common_chars / max(len(actual_clean), len(expected_clean))
            if similarity > 0.8:
                return True, similarity

        return False, 0.0

    def calculate_reward(
        self,
        agent_output: str,
        expected_result: Any,
        **kwargs
    ) -> RewardResult:
        """
        Calculate reward based on code execution results.

        Args:
            agent_output: Agent's output text containing code
            expected_result: Expected execution result
            **kwargs: Additional parameters

        Returns:
            RewardResult with detailed reward breakdown
        """
        # Extract code from agent output
        code_blocks = self.extract_python_code(agent_output)

        if not code_blocks:
            # No code found - minimal penalty
            return RewardResult(
                total_reward=-0.1,
                details={"error": "No code found in output"}
            )

        # Execute the first (or best) code block
        best_result = None
        best_reward = float('-inf')

        for i, code in enumerate(code_blocks):
            execution_result = self.execute_code_safely(code)

            # Calculate reward components
            execution_reward = 0.0
            correctness_reward = 0.0
            error_penalty = 0.0
            timeout_penalty = 0.0

            if execution_result.success:
                execution_reward = self.execution_success_reward

                # Check correctness
                is_correct, similarity = self.compare_results(
                    execution_result.output,
                    expected_result
                )

                if is_correct:
                    correctness_reward = self.correctness_reward * similarity
            else:
                # Apply penalties based on error type
                if execution_result.error:
                    if "SyntaxError" in execution_result.error:
                        error_penalty = self.syntax_error_penalty
                    elif "timeout" in execution_result.error.lower():
                        timeout_penalty = self.timeout_penalty
                    else:
                        error_penalty = self.runtime_error_penalty

            total_reward = execution_reward + correctness_reward + error_penalty + timeout_penalty

            reward_result = RewardResult(
                total_reward=total_reward,
                execution_reward=execution_reward,
                correctness_reward=correctness_reward,
                error_penalty=error_penalty,
                timeout_penalty=timeout_penalty,
                details={
                    "code_block_index": i,
                    "code": code,
                    "execution_result": execution_result,
                    "expected_result": expected_result,
                    "execution_time": execution_result.execution_time
                }
            )

            if total_reward > best_reward:
                best_reward = total_reward
                best_result = reward_result

        return best_result or RewardResult(
            total_reward=-0.5,
            details={"error": "All code executions failed"}
        )


class TreeGRPORewardCalculator:
    """
    Wrapper for integrating reward calculators with Tree GRPO training.

    Handles node-level reward calculation and tree-level reward propagation.
    """

    def __init__(self, base_calculator: BaseRewardCalculator):
        """
        Initialize tree GRPO reward calculator.

        Args:
            base_calculator: Base reward calculator (e.g., CodeExecutionRewardCalculator)
        """
        self.base_calculator = base_calculator

    def calculate_node_rewards(
        self,
        tree_nodes: List,  # List of TreeGRPONode
        expected_results: Dict[str, Any],
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate rewards for multiple tree nodes.

        Args:
            tree_nodes: List of TreeGRPONode instances
            expected_results: Dict mapping node_id to expected result
            **kwargs: Additional parameters

        Returns:
            Dict mapping node_id to reward value
        """
        node_rewards = {}

        for node in tree_nodes:
            node_id = node.node_id

            # Get expected result for this node
            expected_result = expected_results.get(node_id)
            if expected_result is None:
                # If no specific expected result, use a default or skip
                logger.warning(f"No expected result for node {node_id}")
                node_rewards[node_id] = 0.0
                continue

            # Calculate reward using base calculator
            reward_result = self.base_calculator.calculate_reward(
                agent_output=node.local_generation_text,
                expected_result=expected_result,
                **kwargs
            )

            node_rewards[node_id] = reward_result.total_reward

            # Store detailed results in node for debugging
            if hasattr(node, 'reward_details'):
                node.reward_details = reward_result.details

        return node_rewards

    def get_performance_summary(self, reward_results: Dict[str, float]) -> Dict[str, Any]:
        """Get performance summary from reward results."""
        if not reward_results:
            return {"error": "No reward results"}

        values = list(reward_results.values())
        return {
            "total_nodes": len(values),
            "mean_reward": sum(values) / len(values),
            "max_reward": max(values),
            "min_reward": min(values),
            "positive_rewards": sum(1 for v in values if v > 0),
            "negative_rewards": sum(1 for v in values if v < 0),
            "success_rate": sum(1 for v in values if v > 0) / len(values)
        }


# Convenience function for creating common reward calculators
def create_code_execution_calculator(**kwargs) -> CodeExecutionRewardCalculator:
    """Create a CodeExecutionRewardCalculator with default settings."""
    return CodeExecutionRewardCalculator(**kwargs)


def create_tree_grpo_calculator(base_calculator: BaseRewardCalculator) -> TreeGRPORewardCalculator:
    """Create a TreeGRPORewardCalculator with the given base calculator."""
    return TreeGRPORewardCalculator(base_calculator)