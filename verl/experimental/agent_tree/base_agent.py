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

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from .tree_structures import TreeNode, SearchContext, NodeEvaluation


class AgentRole(Enum):
    """Defines the role of an agent in the tree search process."""

    EXPLORER = "explorer"           # Generates new branches/solutions
    EVALUATOR = "evaluator"         # Evaluates quality of nodes/solutions
    COORDINATOR = "coordinator"     # Coordinates the search process
    SYNTHESIZER = "synthesizer"     # Synthesizes results from multiple agents
    VALIDATOR = "validator"         # Validates solutions
    CRITIC = "critic"              # Provides critical analysis
    CREATIVE = "creative"          # Provides creative/unconventional thinking


class BaseTreeAgent(ABC):
    """
    Abstract base class for all tree search agents.

    This class defines the standard interface that all specialized agents must implement
    to participate in collaborative tree search. Each agent can have different roles
    and specializations, but must conform to this interface for interoperability.
    """

    def __init__(
        self,
        name: str,
        role: AgentRole,
        specialization: Optional[str] = None,
        model_path: Optional[str] = None,
        tools: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the base tree agent.

        Args:
            name: Unique identifier for this agent
            role: The role this agent plays in tree search
            specialization: Optional specialization area (e.g., "mathematical", "creative")
            model_path: Path to the language model used by this agent
            tools: List of tools available to this agent
            config: Additional configuration parameters
        """
        self.name = name
        self.role = role
        self.specialization = specialization
        self.model_path = model_path
        self.tools = tools or []
        self.config = config or {}

        # Performance tracking
        self.generation_count = 0
        self.evaluation_count = 0
        self.success_rate = 0.0

    @abstractmethod
    async def generate_branches(
        self,
        node: TreeNode,
        context: SearchContext,
        num_branches: int = 3
    ) -> List[TreeNode]:
        """
        Generate new branches (child nodes) from the given node.

        Args:
            node: The parent node to expand
            context: Current search context and state
            num_branches: Number of branches to generate

        Returns:
            List of newly generated child nodes
        """
        pass

    @abstractmethod
    async def evaluate_node(
        self,
        node: TreeNode,
        context: SearchContext
    ) -> NodeEvaluation:
        """
        Evaluate the quality/promise of a given node.

        Args:
            node: The node to evaluate
            context: Current search context and state

        Returns:
            Evaluation results for the node
        """
        pass

    @abstractmethod
    async def should_expand(
        self,
        node: TreeNode,
        context: SearchContext
    ) -> bool:
        """
        Determine whether the given node should be expanded further.

        Args:
            node: The node to check for expansion
            context: Current search context and state

        Returns:
            True if the node should be expanded, False otherwise
        """
        pass

    async def coordinate_search(
        self,
        nodes: List[TreeNode],
        context: SearchContext
    ) -> Dict[str, Any]:
        """
        Coordinate the search process (for coordinator agents).

        Args:
            nodes: Current active nodes in the search
            context: Current search context and state

        Returns:
            Coordination instructions/plan
        """
        # Default implementation for non-coordinator agents
        return {"action": "continue", "instructions": []}

    async def synthesize_results(
        self,
        evaluations: List[NodeEvaluation],
        context: SearchContext
    ) -> List[TreeNode]:
        """
        Synthesize multiple evaluation results (for synthesizer agents).

        Args:
            evaluations: List of node evaluations from multiple agents
            context: Current search context and state

        Returns:
            Filtered and ranked list of nodes
        """
        # Default implementation for non-synthesizer agents
        return []

    def get_specialization_prompt(self, task_type: str) -> str:
        """
        Get a specialized prompt template for the given task type.

        Args:
            task_type: Type of task ("branch", "evaluate", "coordinate", etc.)

        Returns:
            Formatted prompt template for this agent's specialization
        """
        base_prompts = {
            "branch": f"As a {self.specialization or 'general'} specialist, explore different approaches to: {{problem}}",
            "evaluate": f"From a {self.specialization or 'general'} perspective, assess: {{solution}}",
            "coordinate": f"Coordinate the {self.specialization or 'general'} aspects of: {{search_state}}"
        }
        return base_prompts.get(task_type, "Analyze: {problem}")

    def update_performance_metrics(self, success: bool):
        """Update agent performance metrics."""
        if self.role == AgentRole.EXPLORER:
            self.generation_count += 1
        elif self.role == AgentRole.EVALUATOR:
            self.evaluation_count += 1

        # Update success rate with exponential moving average
        alpha = 0.1
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', role={self.role.value}, specialization='{self.specialization}')"