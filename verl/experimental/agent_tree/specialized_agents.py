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

import asyncio
import logging
from typing import Any, Dict, List, Optional

from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
from verl.workers.rollout.replica import TokenOutput

from .base_agent import BaseTreeAgent, AgentRole
from .tree_structures import NodeEvaluation, SearchContext, TreeNode

logger = logging.getLogger(__name__)


class SpecializedTreeAgent(BaseTreeAgent):
    """
    Base class for specialized tree agents that integrate with verl's infrastructure.

    This class provides common functionality for agents that use language models
    and tools, while allowing specialization through prompt engineering and
    tool selection.
    """

    def __init__(
        self,
        name: str,
        role: AgentRole,
        specialization: str,
        server_manager: AsyncLLMServerManager,
        tokenizer: Any,
        model_path: Optional[str] = None,
        tools: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the specialized tree agent.

        Args:
            name: Unique identifier for this agent
            role: The role this agent plays in tree search
            specialization: Area of specialization
            server_manager: AsyncLLMServerManager for generating responses
            tokenizer: Tokenizer for text processing
            model_path: Path to the language model
            tools: List of available tools
            config: Additional configuration
        """
        super().__init__(name, role, specialization, model_path, tools, config)

        self.server_manager = server_manager
        self.tokenizer = tokenizer

        # Load specialization-specific prompts and configurations
        self.prompt_templates = self._load_prompt_templates()
        self.generation_config = self._load_generation_config()

        # Performance tuning parameters
        self.temperature = self.config.get("temperature", 0.8)
        self.max_tokens = self.config.get("max_tokens", 200)
        self.top_p = self.config.get("top_p", 0.9)

    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates for this agent's specialization."""
        base_templates = {
            "exploration_prompt": (
                "You are a {specialization} specialist. "
                "Analyze the following conversation and generate {num_branches} different approaches to continue.\n"
                "Focus on {specialization} aspects and provide diverse, high-quality solutions.\n\n"
                "Previous conversation:\n{conversation}\n\n"
                "Generate {num_branches} different continuations, each marked with [BRANCH_X]:"
            ),
            "evaluation_prompt": (
                "As a {specialization} expert, evaluate the quality of this response:\n\n"
                "Response: {response}\n\n"
                "Criteria:\n"
                "1. Accuracy and correctness\n"
                "2. Clarity and coherence\n"
                "3. Completeness\n"
                "4. {specialization}-specific quality\n\n"
                "Provide a score from 0.0 to 1.0 and brief reasoning:"
            ),
            "expansion_check_prompt": (
                "As a {specialization} specialist, determine if this conversation should be expanded further.\n\n"
                "Conversation: {conversation}\n\n"
                "Consider:\n"
                "1. Is the problem fully solved?\n"
                "2. Are there alternative approaches worth exploring?\n"
                "3. Would {specialization} analysis benefit from more exploration?\n\n"
                "Respond with YES or NO and brief reasoning:"
            )
        }

        # Specialization-specific templates
        specialization_templates = {
            "mathematical": {
                "exploration_prompt": (
                    "You are a mathematical reasoning specialist. "
                    "Analyze the math problem and generate {num_branches} different solution approaches.\n"
                    "Consider various mathematical techniques, shortcuts, and verification methods.\n\n"
                    "Problem context:\n{conversation}\n\n"
                    "Generate {num_branches} different mathematical approaches, each marked with [BRANCH_X]:"
                ),
                "evaluation_prompt": (
                    "Evaluate this mathematical solution:\n\n"
                    "Solution: {response}\n\n"
                    "Check for:\n"
                    "1. Mathematical correctness\n"
                    "2. Logical reasoning steps\n"
                    "3. Computational accuracy\n"
                    "4. Completeness of solution\n\n"
                    "Score (0.0-1.0) and reasoning:"
                )
            },
            "creative": {
                "exploration_prompt": (
                    "You are a creative thinking specialist. "
                    "Generate {num_branches} innovative and unconventional approaches to this challenge.\n"
                    "Think outside the box and explore novel perspectives.\n\n"
                    "Challenge: {conversation}\n\n"
                    "Generate {num_branches} creative solutions, each marked with [BRANCH_X]:"
                ),
                "evaluation_prompt": (
                    "Evaluate the creativity and innovation of this response:\n\n"
                    "Response: {response}\n\n"
                    "Consider:\n"
                    "1. Originality and novelty\n"
                    "2. Practical feasibility\n"
                    "3. Creative value\n"
                    "4. Potential impact\n\n"
                    "Score (0.0-1.0) and reasoning:"
                )
            },
            "logical": {
                "exploration_prompt": (
                    "You are a logical reasoning specialist. "
                    "Apply rigorous logical analysis to generate {num_branches} systematic approaches.\n"
                    "Focus on logical consistency, clear reasoning chains, and valid conclusions.\n\n"
                    "Problem: {conversation}\n\n"
                    "Generate {num_branches} logical approaches, each marked with [BRANCH_X]:"
                ),
                "evaluation_prompt": (
                    "Evaluate the logical validity of this reasoning:\n\n"
                    "Reasoning: {response}\n\n"
                    "Check for:\n"
                    "1. Logical consistency\n"
                    "2. Valid inference steps\n"
                    "3. Sound argumentation\n"
                    "4. Absence of fallacies\n\n"
                    "Score (0.0-1.0) and reasoning:"
                )
            }
        }

        # Merge base templates with specialization-specific ones
        templates = base_templates.copy()
        if self.specialization in specialization_templates:
            templates.update(specialization_templates[self.specialization])

        return templates

    def _load_generation_config(self) -> Dict[str, Any]:
        """Load generation configuration for this agent."""
        base_config = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": ["[BRANCH_", "\n\n---", "USER:", "ASSISTANT:"]
        }

        # Specialization-specific adjustments
        if self.specialization == "mathematical":
            base_config.update({
                "temperature": 0.3,  # Lower temperature for math
                "top_p": 0.8
            })
        elif self.specialization == "creative":
            base_config.update({
                "temperature": 1.0,  # Higher temperature for creativity
                "top_p": 0.95
            })
        elif self.specialization == "logical":
            base_config.update({
                "temperature": 0.5,  # Moderate temperature for logic
                "top_p": 0.85
            })

        return base_config

    async def generate_branches(
        self,
        node: TreeNode,
        context: SearchContext,
        num_branches: int = 3
    ) -> List[TreeNode]:
        """
        Generate new branches from the given node using this agent's specialization.
        """
        try:
            # Prepare the conversation context
            conversation_text = self._format_conversation(node.messages)

            # Create specialized prompt
            prompt_template = self.prompt_templates.get("exploration_prompt",
                                                       self.prompt_templates["exploration_prompt"])
            prompt = prompt_template.format(
                specialization=self.specialization,
                conversation=conversation_text,
                num_branches=num_branches
            )

            # Prepare prompt IDs for generation
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

            # Generate multiple candidates using vLLM's multi-candidate support
            sampling_params = context.sampling_params.copy()
            sampling_params.update(self.generation_config)
            sampling_params.update({
                "n": num_branches,
                "best_of": num_branches * 2,  # Generate more candidates, select best
            })

            # Generate response
            output = await self.server_manager.generate(
                request_id=f"{self.name}_{node.node_id}",
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=node.image_data
            )

            # Parse the output into separate branches
            branches = self._parse_branches(output, num_branches)

            # Create new tree nodes
            new_nodes = []
            for i, branch_content in enumerate(branches):
                branch_ids = self.tokenizer.encode(branch_content, add_special_tokens=False)

                new_node = TreeNode(
                    parent_id=node.node_id,
                    depth=node.depth + 1,
                    messages=node.messages + [{"role": "assistant", "content": branch_content}],
                    prompt_ids=node.prompt_ids + prompt_ids,
                    response_ids=branch_ids,
                    agent_name=self.name,
                    branch_reason=f"{self.specialization}_exploration",
                    generation_method="multi_branch",
                    generation_candidates=[branch.encode() if hasattr(branch, 'encode') else str(branch).encode() for branch in branches],
                    selected_candidate_idx=i
                )

                new_nodes.append(new_node)

            logger.debug(f"Agent {self.name} generated {len(new_nodes)} branches for node {node.node_id}")
            return new_nodes

        except Exception as e:
            logger.error(f"Agent {self.name} failed to generate branches: {e}")
            return []

    async def evaluate_node(
        self,
        node: TreeNode,
        context: SearchContext
    ) -> NodeEvaluation:
        """
        Evaluate the quality of a node using this agent's expertise.
        """
        try:
            # Get the response to evaluate
            if node.messages and node.messages[-1]["role"] == "assistant":
                response_text = node.messages[-1]["content"]
            else:
                response_text = self.tokenizer.decode(node.response_ids, skip_special_tokens=True)

            # Create evaluation prompt
            eval_template = self.prompt_templates.get("evaluation_prompt")
            eval_prompt = eval_template.format(
                specialization=self.specialization,
                response=response_text
            )

            # Generate evaluation
            prompt_ids = self.tokenizer.encode(eval_prompt, add_special_tokens=True)
            sampling_params = {
                "temperature": 0.3,  # Low temperature for consistent evaluation
                "max_tokens": 150,
                "top_p": 0.8
            }

            output = await self.server_manager.generate(
                request_id=f"{self.name}_eval_{node.node_id}",
                prompt_ids=prompt_ids,
                sampling_params=sampling_params
            )

            # Parse evaluation results
            evaluation_text = output.token_ids if hasattr(output, 'token_ids') else str(output)
            if isinstance(evaluation_text, list):
                evaluation_text = self.tokenizer.decode(evaluation_text, skip_special_tokens=True)

            quality_score, reasoning = self._parse_evaluation(evaluation_text)

            return NodeEvaluation(
                node_id=node.node_id,
                agent_name=self.name,
                quality_score=quality_score,
                confidence=0.8,  # Could be made dynamic
                reasoning=reasoning,
                should_expand=quality_score < 0.9  # Expand if not near perfect
            )

        except Exception as e:
            logger.error(f"Agent {self.name} failed to evaluate node {node.node_id}: {e}")
            # Return default evaluation
            return NodeEvaluation(
                node_id=node.node_id,
                agent_name=self.name,
                quality_score=0.5,
                confidence=0.1,
                reasoning=f"Evaluation failed: {str(e)}"
            )

    async def should_expand(
        self,
        node: TreeNode,
        context: SearchContext
    ) -> bool:
        """
        Determine if this node should be expanded further.
        """
        try:
            # Basic heuristics
            if node.depth >= context.max_depth:
                return False

            if node.is_terminal:
                return False

            # Use LLM for more sophisticated expansion decision
            conversation_text = self._format_conversation(node.messages)

            expansion_template = self.prompt_templates.get("expansion_check_prompt")
            if expansion_template:
                prompt = expansion_template.format(
                    specialization=self.specialization,
                    conversation=conversation_text
                )

                prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
                sampling_params = {
                    "temperature": 0.2,
                    "max_tokens": 50,
                    "top_p": 0.8
                }

                output = await self.server_manager.generate(
                    request_id=f"{self.name}_expand_{node.node_id}",
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params
                )

                response_text = output.token_ids if hasattr(output, 'token_ids') else str(output)
                if isinstance(response_text, list):
                    response_text = self.tokenizer.decode(response_text, skip_special_tokens=True)

                return "YES" in response_text.upper()

            # Fallback to score-based decision
            return node.cumulative_score < 0.8 and node.depth < context.max_depth - 1

        except Exception as e:
            logger.warning(f"Agent {self.name} expansion check failed: {e}")
            # Conservative fallback
            return node.depth < context.max_depth - 2

    def _format_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """Format conversation messages into a readable string."""
        formatted_lines = []
        for message in messages[-5:]:  # Last 5 messages to avoid too long context
            role = message.get("role", "unknown").upper()
            content = message.get("content", "")
            formatted_lines.append(f"{role}: {content}")
        return "\n".join(formatted_lines)

    def _parse_branches(self, output: TokenOutput, expected_branches: int) -> List[str]:
        """Parse the output into separate branches."""
        if hasattr(output, 'token_ids'):
            text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        else:
            text = str(output)

        # Look for branch markers
        branches = []
        lines = text.split('\n')
        current_branch = []

        for line in lines:
            if '[BRANCH_' in line.upper():
                if current_branch:
                    branches.append('\n'.join(current_branch).strip())
                    current_branch = []
                # Add the content after the marker
                marker_end = line.upper().find(']')
                if marker_end != -1:
                    content = line[marker_end + 1:].strip()
                    if content:
                        current_branch.append(content)
            else:
                if current_branch or not branches:  # First branch might not have marker
                    current_branch.append(line)

        # Add the last branch
        if current_branch:
            branches.append('\n'.join(current_branch).strip())

        # Ensure we have the expected number of branches
        if len(branches) < expected_branches:
            # Split the content if we didn't get enough branches
            while len(branches) < expected_branches and branches:
                longest_branch = max(branches, key=len)
                idx = branches.index(longest_branch)
                parts = longest_branch.split('. ')
                if len(parts) > 1:
                    mid = len(parts) // 2
                    branches[idx] = '. '.join(parts[:mid]) + '.'
                    branches.insert(idx + 1, '. '.join(parts[mid:]))
                else:
                    break

        return branches[:expected_branches]

    def _parse_evaluation(self, evaluation_text: str) -> tuple[float, str]:
        """Parse evaluation text to extract score and reasoning."""
        try:
            # Look for score patterns
            import re
            score_patterns = [
                r'score[:\s]*(\d*\.?\d+)',
                r'(\d*\.?\d+)/10',
                r'(\d*\.?\d+)\s*(?:out of|/)\s*(?:10|1\.0|1)',
                r'rating[:\s]*(\d*\.?\d+)'
            ]

            score = 0.5  # Default score
            for pattern in score_patterns:
                match = re.search(pattern, evaluation_text.lower())
                if match:
                    found_score = float(match.group(1))
                    # Normalize to 0-1 range
                    if found_score > 1.0:
                        found_score = found_score / 10.0
                    score = max(0.0, min(1.0, found_score))
                    break

            # Extract reasoning (everything after the score)
            reasoning = evaluation_text.strip()

            return score, reasoning

        except Exception as e:
            logger.warning(f"Failed to parse evaluation: {e}")
            return 0.5, evaluation_text[:200]  # Truncate long text