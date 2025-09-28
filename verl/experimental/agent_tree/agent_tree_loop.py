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

import logging
import time
from typing import Any, Dict, List, Optional

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

from .agent_factory import AgentFactory
from .agent_pool import AgentPool
from .tree_search_engine import TreeSearchEngine
from .tree_structures import SearchContext, TreeNode, TreeGRPONode, TreeGRPOConfig
from .context_management import (
    HierarchicalContextManager,
    ContextCompressionConfig,
    SimpleContextCompressor
)

logger = logging.getLogger(__name__)


@register("agent_tree")
class AgentTreeLoop(ToolAgentLoop):
    """
    Agent Tree Loop that extends ToolAgentLoop with multi-agent tree search capabilities.

    This class integrates the agent tree search framework with verl's existing
    agent loop infrastructure, providing backwards compatibility while adding
    powerful tree-based exploration and multi-agent collaboration.
    """

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        """Initialize class-level components for agent tree search."""
        # Call parent initialization first
        super().init_class(config, tokenizer, processor, **kwargs)

        # Initialize tree search specific components
        cls.tree_config = config.actor_rollout_ref.rollout.get("tree_config", {})
        cls.agent_tree_config = config.actor_rollout_ref.rollout.get("agent_tree_config", {})

        # Tree search parameters
        cls.max_depth = cls.tree_config.get("max_depth", 4)
        cls.max_branches_per_node = cls.tree_config.get("max_branches_per_node", 3)
        cls.max_active_nodes = cls.tree_config.get("max_active_nodes", 8)
        cls.search_timeout = cls.tree_config.get("search_timeout", 180.0)
        cls.coordination_strategy = cls.tree_config.get("coordination_strategy", "hierarchical")

        # Enable tree search mode
        cls.enable_tree_search = cls.tree_config.get("enable", True)
        cls.fallback_to_single_agent = cls.tree_config.get("fallback_to_single_agent", True)

        # Tree GRPO specific configuration
        cls.tree_grpo_config = config.actor_rollout_ref.rollout.get("tree_grpo_config")
        cls.enable_tree_grpo = cls.tree_grpo_config is not None

        if cls.enable_tree_grpo:
            # Import tree GRPO components
            from .grpo_tree import create_tree_grpo_config_template
            from .tree_structures import TreeStructureConfig, RewardAggregationConfig, GroupingConfig

            # Create or validate tree GRPO configuration
            if isinstance(cls.tree_grpo_config, str):
                # Use predefined template
                cls.grpo_config = create_tree_grpo_config_template(cls.tree_grpo_config)
            elif isinstance(cls.tree_grpo_config, dict):
                # Create from dictionary configuration
                tree_structure = TreeStructureConfig(**cls.tree_grpo_config.get("tree_structure", {}))
                reward_aggregation = RewardAggregationConfig(**cls.tree_grpo_config.get("reward_aggregation", {}))
                grouping = GroupingConfig(**cls.tree_grpo_config.get("grouping", {}))

                cls.grpo_config = TreeGRPOConfig(
                    tree_structure=tree_structure,
                    reward_aggregation=reward_aggregation,
                    grouping=grouping,
                    epsilon=cls.tree_grpo_config.get("epsilon", 1e-6),
                    norm_adv_by_std=cls.tree_grpo_config.get("norm_adv_by_std", True),
                    separate_depth_training=cls.tree_grpo_config.get("separate_depth_training", True),
                    propagate_gradients=cls.tree_grpo_config.get("propagate_gradients", False)
                )
            else:
                cls.grpo_config = cls.tree_grpo_config
        else:
            cls.grpo_config = None

        logger.info(f"AgentTreeLoop initialized: tree_search={cls.enable_tree_search}, "
                   f"tree_grpo={cls.enable_tree_grpo}, max_depth={cls.max_depth}, "
                   f"coordination={cls.coordination_strategy}")

    def __init__(self, *args, **kwargs):
        """Initialize the AgentTreeLoop instance."""
        super().__init__(*args, **kwargs)

        # Initialize tree search components
        self.agent_pool: Optional[AgentPool] = None
        self.tree_search_engine: Optional[TreeSearchEngine] = None

        # Context management
        self.context_manager: Optional[HierarchicalContextManager] = None
        self.context_config: Optional[ContextCompressionConfig] = None

        # Performance tracking
        self.tree_search_stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "average_search_time": 0.0,
            "average_nodes_explored": 0.0,
            "fallback_count": 0
        }

        # Initialize components if tree search is enabled
        if self.enable_tree_search:
            self._initialize_tree_search_components()

    def _initialize_tree_search_components(self):
        """Initialize the agent pool and tree search engine."""
        try:
            # Create agent pool from configuration
            if self.agent_tree_config:
                self.agent_pool = AgentFactory.create_agent_pool(
                    self.agent_tree_config,
                    self.server_manager,
                    self.tokenizer
                )
            else:
                # Create predefined agents if no specific config
                self.agent_pool = AgentFactory.create_predefined_agents(
                    self.server_manager,
                    self.tokenizer,
                    specializations=["mathematical", "creative", "logical"]
                )

            # Create tree search engine
            search_config = {
                "max_depth": self.max_depth,
                "max_branches_per_node": self.max_branches_per_node,
                "max_active_nodes": self.max_active_nodes,
                "search_timeout": self.search_timeout,
                "coordination_strategy": self.coordination_strategy
            }

            self.tree_search_engine = TreeSearchEngine(
                agent_pool=self.agent_pool,
                search_config=search_config
            )

            # Initialize context management
            self._initialize_context_management()

            logger.info(f"Tree search components initialized: {len(self.agent_pool)} agents")

        except Exception as e:
            logger.error(f"Failed to initialize tree search components: {e}")
            if not self.fallback_to_single_agent:
                raise
            logger.warning("Falling back to single agent mode")
            self.enable_tree_search = False

    def _initialize_context_management(self):
        """Initialize the context management system."""
        try:
            # Create context compression configuration
            context_config_dict = getattr(self, "tree_config", {}).get("context_config", {})

            self.context_config = ContextCompressionConfig(
                max_context_length=context_config_dict.get("max_context_length", 1024),
                compression_ratio=context_config_dict.get("compression_ratio", 0.5),
                depth_scaling_factor=context_config_dict.get("depth_scaling_factor", 0.8),
                preserve_keywords=context_config_dict.get("preserve_keywords"),
                agent_type_weights=context_config_dict.get("agent_type_weights")
            )

            # Initialize hierarchical context manager
            self.context_manager = HierarchicalContextManager(self.context_config)

            logger.info(f"Context management initialized: max_length={self.context_config.max_context_length}, "
                       f"compression_ratio={self.context_config.compression_ratio}")

        except Exception as e:
            logger.warning(f"Failed to initialize context management: {e}, using simple fallback")
            # Create a simple fallback configuration
            self.context_config = ContextCompressionConfig()
            self.context_manager = HierarchicalContextManager(self.context_config)

    async def run(self, sampling_params: Dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        Run the agent loop with optional tree search.

        This method decides whether to use tree search or fall back to the
        standard single-agent approach based on configuration and problem complexity.
        """
        # Determine if we should use tree search for this request
        use_tree_search = self._should_use_tree_search(kwargs)

        if use_tree_search and self.enable_tree_search and self.tree_search_engine:
            return await self._run_tree_search(sampling_params, **kwargs)
        else:
            # Fall back to standard single-agent processing
            if use_tree_search:
                self.tree_search_stats["fallback_count"] += 1
                logger.debug("Falling back to single agent mode")
            return await super().run(sampling_params, **kwargs)

    def _should_use_tree_search(self, kwargs: Dict[str, Any]) -> bool:
        """
        Determine whether to use tree search for this request.

        Args:
            kwargs: Request parameters

        Returns:
            True if tree search should be used, False otherwise
        """
        # Check if tree search is explicitly disabled
        if not self.enable_tree_search:
            return False

        # Check for explicit tree search request
        if kwargs.get("force_tree_search", False):
            return True

        if kwargs.get("disable_tree_search", False):
            return False

        # Heuristics based on problem characteristics
        messages = kwargs.get("raw_prompt", [])
        if not messages:
            return False

        # Check for problem indicators that benefit from tree search
        last_message = messages[-1].get("content", "").lower() if messages else ""

        tree_search_indicators = [
            # Mathematical problems
            "solve", "calculate", "prove", "equation", "mathematical",
            # Complex reasoning
            "analyze", "compare", "evaluate", "strategy", "approach",
            # Creative tasks
            "creative", "design", "brainstorm", "alternative", "innovative",
            # Multi-step problems
            "step by step", "multiple ways", "different approaches", "explore"
        ]

        return any(indicator in last_message for indicator in tree_search_indicators)

    async def _run_tree_search(self, sampling_params: Dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        Execute tree search using multiple agents.

        Args:
            sampling_params: Parameters for text generation
            **kwargs: Request parameters

        Returns:
            Agent loop output with the best solution found
        """
        start_time = time.time()
        self.tree_search_stats["total_searches"] += 1

        try:
            # Create initial tree node from the request
            messages = kwargs.get("raw_prompt", [])
            image_data = kwargs.get("image_data")

            initial_node = TreeNode.from_prompt(
                messages=messages,
                image_data=image_data
            )

            # Determine problem type for agent selection
            problem_type = self._classify_problem_type(messages)
            difficulty_level = self._estimate_difficulty(messages)

            logger.info(f"Starting tree search: problem_type={problem_type}, difficulty={difficulty_level:.2f}")

            # Execute tree search
            search_result = await self.tree_search_engine.search(
                initial_node=initial_node,
                sampling_params=sampling_params,
                problem_type=problem_type,
                difficulty_level=difficulty_level
            )

            # Convert tree search result to agent loop output
            output = self._convert_tree_result_to_agent_output(search_result, initial_node)

            # Update statistics
            search_time = time.time() - start_time
            self._update_search_statistics(search_result, search_time, success=True)

            logger.info(f"Tree search completed successfully in {search_time:.2f}s, "
                       f"explored {search_result.total_nodes_explored} nodes, "
                       f"best score: {search_result.best_score:.3f}")

            return output

        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"Tree search failed after {search_time:.2f}s: {e}")
            self._update_search_statistics(None, search_time, success=False)

            # Fall back to single agent if tree search fails
            if self.fallback_to_single_agent:
                logger.info("Falling back to single agent mode due to tree search failure")
                self.tree_search_stats["fallback_count"] += 1
                return await super().run(sampling_params, **kwargs)
            else:
                raise

    def _classify_problem_type(self, messages: List[Dict[str, Any]]) -> str:
        """
        Classify the type of problem based on the conversation content.

        Args:
            messages: Conversation messages

        Returns:
            Problem type classification
        """
        if not messages:
            return "general"

        content = " ".join(msg.get("content", "") for msg in messages).lower()

        # Mathematical problem indicators
        math_keywords = ["calculate", "solve", "equation", "mathematical", "proof", "theorem", "formula"]
        if any(keyword in content for keyword in math_keywords):
            return "mathematical"

        # Creative problem indicators
        creative_keywords = ["creative", "design", "brainstorm", "artistic", "innovative", "original"]
        if any(keyword in content for keyword in creative_keywords):
            return "creative"

        # Coding problem indicators
        code_keywords = ["code", "program", "algorithm", "function", "debug", "programming"]
        if any(keyword in content for keyword in code_keywords):
            return "coding"

        # Planning problem indicators
        planning_keywords = ["plan", "strategy", "organize", "schedule", "project", "management"]
        if any(keyword in content for keyword in planning_keywords):
            return "planning"

        # Analysis problem indicators
        analysis_keywords = ["analyze", "compare", "evaluate", "assess", "review", "examine"]
        if any(keyword in content for keyword in analysis_keywords):
            return "analysis"

        return "general"

    def _estimate_difficulty(self, messages: List[Dict[str, Any]]) -> float:
        """
        Estimate the difficulty level of the problem.

        Args:
            messages: Conversation messages

        Returns:
            Difficulty estimate (0.0 to 1.0)
        """
        if not messages:
            return 0.5

        content = " ".join(msg.get("content", "") for msg in messages).lower()

        # Base difficulty
        difficulty = 0.5

        # Complexity indicators
        complexity_indicators = {
            "multiple": 0.1,
            "complex": 0.2,
            "advanced": 0.2,
            "difficult": 0.2,
            "challenging": 0.2,
            "step by step": 0.1,
            "comprehensive": 0.1,
            "thorough": 0.1,
            "detailed": 0.1
        }

        for indicator, weight in complexity_indicators.items():
            if indicator in content:
                difficulty += weight

        # Length-based difficulty (longer problems tend to be more complex)
        if len(content) > 500:
            difficulty += 0.1
        if len(content) > 1000:
            difficulty += 0.1

        return min(1.0, difficulty)

    def _convert_tree_result_to_agent_output(
        self,
        search_result: "TreeSearchResult",
        initial_node: TreeNode
    ) -> AgentLoopOutput:
        """
        Convert tree search result to standard agent loop output format.

        Args:
            search_result: Results from tree search
            initial_node: Initial node of the search

        Returns:
            Agent loop output
        """
        # Get the best solution
        if search_result.best_path:
            # Use the best path from root to leaf
            best_node_data = search_result.best_path[-1]
            prompt_ids = best_node_data.get("prompt_ids", initial_node.prompt_ids)
            response_ids = best_node_data.get("response_ids", [])
            response_logprobs = best_node_data.get("response_logprobs", [])
            response_mask = [1] * len(response_ids)  # All tokens are generated
        elif search_result.best_nodes:
            # Use the highest scoring node
            best_node_data = search_result.best_nodes[0]
            prompt_ids = best_node_data.get("prompt_ids", initial_node.prompt_ids)
            response_ids = best_node_data.get("response_ids", [])
            response_logprobs = best_node_data.get("response_logprobs", [])
            response_mask = [1] * len(response_ids)
        else:
            # Fallback: empty response
            prompt_ids = initial_node.prompt_ids
            response_ids = []
            response_logprobs = []
            response_mask = []

        # Create metadata about the tree search
        metadata = {
            "tree_search_used": True,
            "total_nodes_explored": search_result.total_nodes_explored,
            "max_depth_reached": search_result.max_depth_reached,
            "search_duration": search_result.search_duration,
            "best_score": search_result.best_score,
            "agent_statistics": search_result.agent_statistics,
            "coordination_strategy": self.coordination_strategy
        }

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            extra_info=metadata
        )

    def _update_search_statistics(
        self,
        search_result: Optional["TreeSearchResult"],
        search_time: float,
        success: bool
    ):
        """Update performance statistics for tree search."""
        if success:
            self.tree_search_stats["successful_searches"] += 1

        # Update average search time with exponential moving average
        alpha = 0.1
        current_avg = self.tree_search_stats["average_search_time"]
        self.tree_search_stats["average_search_time"] = alpha * search_time + (1 - alpha) * current_avg

        # Update average nodes explored
        if search_result:
            current_avg_nodes = self.tree_search_stats["average_nodes_explored"]
            self.tree_search_stats["average_nodes_explored"] = (
                alpha * search_result.total_nodes_explored + (1 - alpha) * current_avg_nodes
            )

    def get_tree_search_statistics(self) -> Dict[str, Any]:
        """Get current tree search performance statistics."""
        stats = self.tree_search_stats.copy()

        if stats["total_searches"] > 0:
            stats["success_rate"] = stats["successful_searches"] / stats["total_searches"]
        else:
            stats["success_rate"] = 0.0

        if self.agent_pool:
            stats["agent_pool_stats"] = self.agent_pool.get_pool_statistics()

        return stats

    def reset_statistics(self):
        """Reset all performance statistics."""
        self.tree_search_stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "average_search_time": 0.0,
            "average_nodes_explored": 0.0,
            "fallback_count": 0
        }

        if self.agent_pool:
            self.agent_pool.reset_workload()

    async def run_tree_grpo_rollout(
        self,
        sampling_params: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a tree-based GRPO rollout following the configured branching pattern.

        This method generates a complete tree structure according to the GRPO configuration,
        collects final rewards from leaf nodes, and prepares the tree structure for
        hierarchical advantage computation.

        Args:
            sampling_params: Parameters for text generation
            **kwargs: Request parameters including raw_prompt, image_data, etc.

        Returns:
            Dictionary containing:
            - tree_nodes: Dict mapping node IDs to TreeGRPONode instances
            - final_rewards: Dict mapping leaf node IDs to their final rewards
            - tree_metadata: Information about the tree structure
        """
        if not self.enable_tree_grpo or not self.grpo_config:
            raise RuntimeError("Tree GRPO is not enabled or configured")

        start_time = time.time()
        logger.info("Starting tree GRPO rollout")

        try:
            # Create root node
            messages = kwargs.get("raw_prompt", [])
            image_data = kwargs.get("image_data")

            root_node = TreeGRPONode.from_prompt(
                messages=messages,
                image_data=image_data,
                depth=0
            )

            # Generate complete tree structure
            tree_nodes = await self._generate_tree_structure(
                root_node, sampling_params, **kwargs
            )

            # Collect final rewards from leaf nodes
            final_rewards = await self._collect_final_rewards(tree_nodes)

            # Prepare tree metadata
            tree_metadata = self._create_tree_metadata(tree_nodes, start_time)

            logger.info(f"Tree GRPO rollout completed: {len(tree_nodes)} nodes, "
                       f"{len(final_rewards)} leaf nodes, "
                       f"duration={time.time() - start_time:.2f}s")

            return {
                "tree_nodes": tree_nodes,
                "final_rewards": final_rewards,
                "tree_metadata": tree_metadata,
                "grpo_config": self.grpo_config
            }

        except Exception as e:
            logger.error(f"Tree GRPO rollout failed: {e}")
            raise

    async def _generate_tree_structure(
        self,
        root_node: TreeGRPONode,
        sampling_params: Dict[str, Any],
        **kwargs
    ) -> Dict[str, TreeGRPONode]:
        """Generate the complete tree structure according to the branching schedule."""
        tree_nodes = {root_node.node_id: root_node}
        current_level_nodes = [root_node]

        for depth in range(1, self.grpo_config.tree_structure.max_depth + 1):
            branches_per_node = self.grpo_config.tree_structure.get_branches_for_depth(depth)
            if branches_per_node <= 0:
                break

            next_level_nodes = []
            logger.info(f"Generating depth {depth}: {branches_per_node} branches per node")

            for parent_node in current_level_nodes:
                # Generate children for this parent
                children = await self._generate_children(
                    parent_node, branches_per_node, sampling_params, **kwargs
                )

                for child in children:
                    child.depth = depth
                    child.parent_id = parent_node.node_id
                    parent_node.add_child(child.node_id)
                    tree_nodes[child.node_id] = child
                    next_level_nodes.append(child)

            current_level_nodes = next_level_nodes

        # Mark leaf nodes
        for node in tree_nodes.values():
            if not node.children_ids:
                node.is_leaf = True

        return tree_nodes

    async def _generate_children(
        self,
        parent_node: TreeGRPONode,
        num_children: int,
        sampling_params: Dict[str, Any],
        **kwargs
    ) -> List[TreeGRPONode]:
        """Generate child nodes for a given parent using intelligent context management."""
        children = []

        # Build compressed context for children using context manager
        try:
            parent_context = parent_node.build_context_for_child(
                context_manager=self.context_manager,
                sibling_nodes=[]  # No siblings yet, will be updated later
            )
        except Exception as e:
            logger.warning(f"Context building failed: {e}, using fallback")
            parent_context = parent_node._simple_context_fallback()

        # Prepare sampling params for multiple candidates
        multi_sampling_params = sampling_params.copy()
        multi_sampling_params["n"] = num_children
        multi_sampling_params["temperature"] = max(0.1, multi_sampling_params.get("temperature", 0.7))

        # Calculate parent's total sequence length for position tracking
        parent_sequence_length = sum(len(msg.get("content", "").split()) for msg in parent_node.messages)

        try:
            # Generate multiple responses
            for i in range(num_children):
                # Create child node with proper initialization
                child = TreeGRPONode(
                    parent_id=parent_node.node_id,
                    depth=parent_node.depth + 1,
                    messages=parent_node.messages.copy(),
                    agent_name=f"agent_depth_{parent_node.depth + 1}_{i}",
                    agent_type=self._select_agent_type_for_child(parent_node, i),
                    compressed_context=parent_context
                )

                # Generate response for this child
                single_output = await self._generate_response_for_node(
                    child, sampling_params, parent_context, **kwargs
                )

                # Set generation boundaries and data
                if single_output and hasattr(single_output, 'response_ids'):
                    start_pos = parent_sequence_length
                    tokens = single_output.response_ids
                    logprobs = getattr(single_output, 'response_logprobs', [0.0] * len(tokens))

                    child.set_generation_boundary(start_pos, tokens, logprobs)
                    child.local_generation_text = self._tokens_to_text(tokens)

                    # Update messages with new content
                    if child.local_generation_text:
                        child.messages.append({
                            "role": "assistant",
                            "content": child.local_generation_text,
                            "agent_type": child.agent_type
                        })
                else:
                    # Handle generation failure
                    logger.warning(f"Failed to generate response for child {i}")
                    child.set_generation_boundary(parent_sequence_length, [], [])
                    child.local_generation_text = f"[Generation failed for {child.agent_name}]"

                children.append(child)

        except Exception as e:
            logger.error(f"Failed to generate children for node {parent_node.node_id}: {e}")
            # Create minimal children to maintain tree structure
            for i in range(num_children):
                child = TreeGRPONode(
                    parent_id=parent_node.node_id,
                    depth=parent_node.depth + 1,
                    messages=parent_node.messages.copy(),
                    agent_name=f"failed_agent_depth_{parent_node.depth + 1}_{i}",
                    agent_type="general",
                    compressed_context=parent_context
                )
                child.set_generation_boundary(parent_sequence_length, [], [])
                child.local_generation_text = "[Failed generation]"
                children.append(child)

        return children

    def _select_agent_type_for_child(self, parent_node: TreeGRPONode, child_index: int) -> str:
        """Select appropriate agent type for a child node."""
        # Simple strategy: rotate through available agent types
        available_types = ["mathematical", "creative", "logical", "general"]

        if hasattr(parent_node, 'agent_type') and parent_node.agent_type in available_types:
            # Choose different type from parent to encourage diversity
            parent_type_index = available_types.index(parent_node.agent_type)
            child_type_index = (parent_type_index + child_index + 1) % len(available_types)
            return available_types[child_type_index]
        else:
            return available_types[child_index % len(available_types)]

    async def _generate_response_for_node(
        self,
        node: TreeGRPONode,
        sampling_params: Dict[str, Any],
        context: str,
        **kwargs
    ) -> Optional[AgentLoopOutput]:
        """Generate a response for a specific node with compressed context."""
        # Prepare input with compressed context instead of full history
        modified_kwargs = kwargs.copy()

        # Use compressed context as the prompt
        if context:
            context_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": context}
            ]
            modified_kwargs["raw_prompt"] = context_messages

        # Select appropriate agent if available
        if self.agent_pool:
            try:
                agent = self.agent_pool.select_agent_for_specialization(node.agent_type)
                if agent:
                    return await agent.generate_response(node, sampling_params, **modified_kwargs)
            except Exception as e:
                logger.warning(f"Agent selection failed: {e}, using fallback")

        # Fallback to parent class generation
        return await super().run(sampling_params, **modified_kwargs)

    def _tokens_to_text(self, tokens: List[int]) -> str:
        """Convert token IDs to human-readable text."""
        if not tokens:
            return ""

        try:
            if hasattr(self, 'tokenizer') and self.tokenizer:
                return self.tokenizer.decode(tokens, skip_special_tokens=True)
            else:
                # Fallback: just join token IDs as strings
                return " ".join(f"tok_{token}" for token in tokens)
        except Exception as e:
            logger.warning(f"Token decoding failed: {e}")
            return f"[{len(tokens)} tokens]"

    async def _generate_response(
        self,
        node: TreeGRPONode,
        sampling_params: Dict[str, Any],
        **kwargs
    ) -> AgentLoopOutput:
        """Generate a response for a given node using the appropriate agent."""
        # Select an appropriate agent based on the node's context
        if self.agent_pool:
            agent = self.agent_pool.select_agent_for_node(node)
            if agent:
                return await agent.generate_response(node, sampling_params, **kwargs)

        # Fallback to parent class generation
        return await super().run(sampling_params, **kwargs)

    async def _collect_final_rewards(self, tree_nodes: Dict[str, TreeGRPONode]) -> Dict[str, List[float]]:
        """Collect final rewards from all leaf nodes."""
        final_rewards = {}
        leaf_nodes = [node for node in tree_nodes.values() if node.is_leaf]

        logger.info(f"Collecting rewards from {len(leaf_nodes)} leaf nodes")

        for leaf_node in leaf_nodes:
            try:
                # In a real implementation, this would call a reward model
                # For now, we'll use a placeholder reward based on response quality
                reward = await self._compute_leaf_reward(leaf_node)
                final_rewards[leaf_node.node_id] = [reward]
                leaf_node.final_rewards = [reward]
            except Exception as e:
                logger.error(f"Failed to compute reward for leaf {leaf_node.node_id}: {e}")
                final_rewards[leaf_node.node_id] = [0.0]
                leaf_node.final_rewards = [0.0]

        return final_rewards

    async def _compute_leaf_reward(self, leaf_node: TreeGRPONode) -> float:
        """
        Compute reward for a leaf node.

        In a real implementation, this would call a reward model or use human feedback.
        For now, we provide a placeholder implementation.
        """
        # Placeholder: Use response length and some heuristics
        response_length = len(leaf_node.response_ids) if leaf_node.response_ids else 0

        # Base reward from response length (longer responses might be more detailed)
        base_reward = min(1.0, response_length / 100.0)

        # Add some randomness to simulate reward model variance
        import random
        noise = random.gauss(0, 0.1)
        reward = max(0.0, min(1.0, base_reward + noise))

        logger.debug(f"Leaf node {leaf_node.node_id} reward: {reward:.3f}")
        return reward

    def _create_tree_metadata(self, tree_nodes: Dict[str, TreeGRPONode], start_time: float) -> Dict[str, Any]:
        """Create metadata about the generated tree structure."""
        depth_counts = {}
        for node in tree_nodes.values():
            depth_counts[node.depth] = depth_counts.get(node.depth, 0) + 1

        leaf_count = sum(1 for node in tree_nodes.values() if node.is_leaf)

        return {
            "total_nodes": len(tree_nodes),
            "leaf_nodes": leaf_count,
            "max_depth": max(node.depth for node in tree_nodes.values()) if tree_nodes else 0,
            "depth_distribution": depth_counts,
            "branching_schedule": self.grpo_config.tree_structure.get_branching_schedule(),
            "generation_duration": time.time() - start_time,
            "tree_structure_summary": self._get_tree_structure_summary(tree_nodes)
        }

    def _get_tree_structure_summary(self, tree_nodes: Dict[str, TreeGRPONode]) -> str:
        """Generate a human-readable summary of the tree structure."""
        depth_counts = {}
        for node in tree_nodes.values():
            depth_counts[node.depth] = depth_counts.get(node.depth, 0) + 1

        if not depth_counts:
            return "Empty tree"

        summary_parts = []
        for depth in sorted(depth_counts.keys()):
            summary_parts.append(str(depth_counts[depth]))

        return "→".join(summary_parts)

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the tree search system."""
        health_status = {
            "tree_search_enabled": self.enable_tree_search,
            "agent_pool_size": len(self.agent_pool) if self.agent_pool else 0,
            "tree_search_engine_ready": self.tree_search_engine is not None,
            "statistics": self.get_tree_search_statistics()
        }

        # Test agent pool if available
        if self.agent_pool:
            try:
                # Try to get agents for different roles
                explorers = self.agent_pool.get_agents_by_role("EXPLORER")
                evaluators = self.agent_pool.get_agents_by_role("EVALUATOR")

                health_status["agent_pool_details"] = {
                    "explorers": len(explorers),
                    "evaluators": len(evaluators),
                    "total_agents": len(self.agent_pool)
                }
            except Exception as e:
                health_status["agent_pool_error"] = str(e)

        return health_status