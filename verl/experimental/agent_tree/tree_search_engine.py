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
import time
from typing import Any, Dict, List, Optional

from .agent_pool import AgentPool
from .base_agent import AgentRole, BaseTreeAgent
from .tree_structures import NodeEvaluation, SearchContext, TreeNode, TreeSearchResult

logger = logging.getLogger(__name__)


class TreeSearchEngine:
    """
    Core engine for multi-agent tree search.

    This engine coordinates multiple agents to collaboratively explore a search space
    using tree-based algorithms. It handles agent selection, coordination, evaluation,
    and result synthesis.
    """

    def __init__(
        self,
        agent_pool: AgentPool,
        search_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the tree search engine.

        Args:
            agent_pool: Pool of available agents
            search_config: Configuration parameters for search
        """
        self.agent_pool = agent_pool
        self.search_config = search_config or {}

        # Default search parameters
        self.max_depth = self.search_config.get("max_depth", 5)
        self.max_branches_per_node = self.search_config.get("max_branches_per_node", 3)
        self.max_active_nodes = self.search_config.get("max_active_nodes", 10)
        self.search_timeout = self.search_config.get("search_timeout", 300.0)
        self.coordination_strategy = self.search_config.get("coordination_strategy", "hierarchical")

        # Performance tracking
        self.total_agent_calls = 0
        self.search_statistics = {}

    async def search(
        self,
        initial_node: TreeNode,
        sampling_params: Dict[str, Any],
        problem_type: str = "general",
        difficulty_level: float = 0.5
    ) -> TreeSearchResult:
        """
        Execute a complete tree search starting from the initial node.

        Args:
            initial_node: Root node to start search from
            sampling_params: Parameters for text generation
            problem_type: Type of problem being solved
            difficulty_level: Estimated difficulty (0.0 to 1.0)

        Returns:
            Complete search results including best solutions and statistics
        """
        # Initialize search context
        context = SearchContext(
            max_depth=self.max_depth,
            max_branches_per_node=self.max_branches_per_node,
            max_active_nodes=self.max_active_nodes,
            search_timeout=self.search_timeout,
            problem_type=problem_type,
            difficulty_level=difficulty_level,
            sampling_params=sampling_params
        )

        context.register_node(initial_node)
        active_nodes = [initial_node]

        logger.info(f"Starting tree search with {len(self.agent_pool)} agents")
        logger.info(f"Search config: depth={self.max_depth}, branches={self.max_branches_per_node}, timeout={self.search_timeout}s")

        try:
            # Main search loop
            while active_nodes and not context.should_terminate_search():
                logger.debug(f"Search iteration: depth={context.current_depth}, active_nodes={len(active_nodes)}")

                # Execute one search iteration
                new_nodes = await self._search_iteration(active_nodes, context)

                if not new_nodes:
                    logger.info("No new nodes generated, terminating search")
                    break

                # Update active nodes and search state
                active_nodes = self._select_nodes_for_next_iteration(new_nodes, context)
                context.current_depth += 1
                context.update_search_metrics(active_nodes)

                logger.debug(f"Generated {len(new_nodes)} new nodes, {len(active_nodes)} remain active")

        except Exception as e:
            logger.error(f"Error during tree search: {e}")
            raise

        # Build final results
        result = self._build_search_result(context)
        logger.info(f"Search completed: {result.total_nodes_explored} nodes explored, best score: {result.best_score:.3f}")

        return result

    async def _search_iteration(
        self,
        active_nodes: List[TreeNode],
        context: SearchContext
    ) -> List[TreeNode]:
        """
        Execute one iteration of the tree search.

        Args:
            active_nodes: Nodes to expand in this iteration
            context: Current search context

        Returns:
            New nodes generated in this iteration
        """
        if self.coordination_strategy == "hierarchical":
            return await self._hierarchical_search_iteration(active_nodes, context)
        elif self.coordination_strategy == "collaborative":
            return await self._collaborative_search_iteration(active_nodes, context)
        elif self.coordination_strategy == "competitive":
            return await self._competitive_search_iteration(active_nodes, context)
        else:
            logger.warning(f"Unknown coordination strategy: {self.coordination_strategy}, using hierarchical")
            return await self._hierarchical_search_iteration(active_nodes, context)

    async def _hierarchical_search_iteration(
        self,
        active_nodes: List[TreeNode],
        context: SearchContext
    ) -> List[TreeNode]:
        """
        Hierarchical search: coordinator plans, explorers execute, evaluators assess.
        """
        # Step 1: Coordination (if coordinator available)
        coordinator = self.agent_pool.select_coordinator_agent(context)
        if coordinator:
            try:
                coordination_plan = await coordinator.coordinate_search(active_nodes, context)
                logger.debug(f"Coordinator plan: {coordination_plan}")
            except Exception as e:
                logger.warning(f"Coordinator failed: {e}")
                coordination_plan = {"action": "continue"}
        else:
            coordination_plan = {"action": "continue"}

        # Step 2: Parallel exploration
        exploration_tasks = []
        agent_assignments = self.agent_pool.select_agents_for_exploration(
            active_nodes, context, max_agents=len(active_nodes)
        )

        for node, agent in agent_assignments.items():
            task = self._explore_node_with_agent(node, agent, context)
            exploration_tasks.append(task)

        if exploration_tasks:
            exploration_results = await asyncio.gather(*exploration_tasks, return_exceptions=True)
            new_nodes = []
            for result in exploration_results:
                if isinstance(result, Exception):
                    logger.warning(f"Exploration failed: {result}")
                else:
                    new_nodes.extend(result)
        else:
            new_nodes = []

        # Step 3: Parallel evaluation
        if new_nodes:
            evaluation_tasks = []
            evaluator_agents = self.agent_pool.select_agents_for_evaluation(new_nodes, context)

            for node in new_nodes:
                for evaluator in evaluator_agents:
                    task = self._evaluate_node_with_agent(node, evaluator, context)
                    evaluation_tasks.append(task)

            if evaluation_tasks:
                evaluation_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
                self._process_evaluations(new_nodes, evaluation_results, context)

        # Step 4: Synthesis (if synthesizer available)
        synthesizer = self.agent_pool.select_synthesizer_agent(context)
        if synthesizer and new_nodes:
            try:
                # Create dummy evaluations for synthesis
                evaluations = [
                    NodeEvaluation(
                        node_id=node.node_id,
                        agent_name="system",
                        quality_score=node.quality_score,
                        confidence=0.8,
                        reasoning="System evaluation"
                    )
                    for node in new_nodes
                ]
                synthesized_nodes = await synthesizer.synthesize_results(evaluations, context)
                if synthesized_nodes:
                    new_nodes = synthesized_nodes
            except Exception as e:
                logger.warning(f"Synthesis failed: {e}")

        return new_nodes

    async def _collaborative_search_iteration(
        self,
        active_nodes: List[TreeNode],
        context: SearchContext
    ) -> List[TreeNode]:
        """
        Collaborative search: all agents work together on each node.
        """
        all_new_nodes = []

        for node in active_nodes:
            # Get diverse agents for this node
            diverse_agents = self.agent_pool.get_diverse_agents(num_agents=3)
            explorer_agents = [agent for agent in diverse_agents if agent.role == AgentRole.EXPLORER]

            if not explorer_agents:
                explorer_agents = self.agent_pool.get_agents_by_role(AgentRole.EXPLORER)[:2]

            # Parallel exploration by multiple agents
            exploration_tasks = [
                self._explore_node_with_agent(node, agent, context)
                for agent in explorer_agents
            ]

            if exploration_tasks:
                exploration_results = await asyncio.gather(*exploration_tasks, return_exceptions=True)
                node_new_nodes = []
                for result in exploration_results:
                    if not isinstance(result, Exception):
                        node_new_nodes.extend(result)

                # Collaborative evaluation
                if node_new_nodes:
                    evaluator_agents = self.agent_pool.get_agents_by_role(AgentRole.EVALUATOR)
                    evaluation_tasks = []
                    for new_node in node_new_nodes:
                        for evaluator in evaluator_agents:
                            task = self._evaluate_node_with_agent(new_node, evaluator, context)
                            evaluation_tasks.append(task)

                    if evaluation_tasks:
                        evaluation_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
                        self._process_evaluations(node_new_nodes, evaluation_results, context)

                all_new_nodes.extend(node_new_nodes)

        return all_new_nodes

    async def _competitive_search_iteration(
        self,
        active_nodes: List[TreeNode],
        context: SearchContext
    ) -> List[TreeNode]:
        """
        Competitive search: agents compete to generate the best solutions.
        """
        all_candidates = []

        # Each agent generates solutions for all nodes
        explorer_agents = self.agent_pool.get_agents_by_role(AgentRole.EXPLORER)

        for agent in explorer_agents:
            agent_tasks = [
                self._explore_node_with_agent(node, agent, context)
                for node in active_nodes
            ]

            if agent_tasks:
                agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
                for result in agent_results:
                    if not isinstance(result, Exception):
                        all_candidates.extend(result)

        # Competitive evaluation - select only the best
        if all_candidates:
            evaluator_agents = self.agent_pool.get_agents_by_role(AgentRole.EVALUATOR)
            evaluation_tasks = []

            for candidate in all_candidates:
                for evaluator in evaluator_agents:
                    task = self._evaluate_node_with_agent(candidate, evaluator, context)
                    evaluation_tasks.append(task)

            if evaluation_tasks:
                evaluation_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
                self._process_evaluations(all_candidates, evaluation_results, context)

            # Select top candidates
            all_candidates.sort(key=lambda x: x.cumulative_score, reverse=True)
            return all_candidates[:self.max_active_nodes]

        return all_candidates

    async def _explore_node_with_agent(
        self,
        node: TreeNode,
        agent: BaseTreeAgent,
        context: SearchContext
    ) -> List[TreeNode]:
        """
        Have a specific agent explore/expand a node.
        """
        start_time = time.time()
        try:
            # Check if agent should expand this node
            should_expand = await agent.should_expand(node, context)
            if not should_expand:
                return []

            # Generate branches
            new_nodes = await agent.generate_branches(
                node, context, num_branches=self.max_branches_per_node
            )

            # Register new nodes in context
            for new_node in new_nodes:
                context.register_node(new_node)
                context.total_nodes_generated += 1

            execution_time = time.time() - start_time
            self.agent_pool.update_agent_performance(agent.name, True, execution_time)
            self.total_agent_calls += 1

            return new_nodes

        except Exception as e:
            execution_time = time.time() - start_time
            logger.warning(f"Agent {agent.name} failed to explore node {node.node_id}: {e}")
            self.agent_pool.update_agent_performance(agent.name, False, execution_time)
            return []

    async def _evaluate_node_with_agent(
        self,
        node: TreeNode,
        agent: BaseTreeAgent,
        context: SearchContext
    ) -> Optional[NodeEvaluation]:
        """
        Have a specific agent evaluate a node.
        """
        start_time = time.time()
        try:
            evaluation = await agent.evaluate_node(node, context)
            execution_time = time.time() - start_time
            self.agent_pool.update_agent_performance(agent.name, True, execution_time)
            self.total_agent_calls += 1
            return evaluation

        except Exception as e:
            execution_time = time.time() - start_time
            logger.warning(f"Agent {agent.name} failed to evaluate node {node.node_id}: {e}")
            self.agent_pool.update_agent_performance(agent.name, False, execution_time)
            return None

    def _process_evaluations(
        self,
        nodes: List[TreeNode],
        evaluation_results: List[Any],
        context: SearchContext
    ):
        """
        Process evaluation results and update node scores.
        """
        evaluations_by_node = {}

        for result in evaluation_results:
            if isinstance(result, NodeEvaluation):
                node_id = result.node_id
                if node_id not in evaluations_by_node:
                    evaluations_by_node[node_id] = []
                evaluations_by_node[node_id].append(result)

        # Update node scores based on aggregated evaluations
        for node in nodes:
            node_evaluations = evaluations_by_node.get(node.node_id, [])
            if node_evaluations:
                # Aggregate multiple evaluations
                quality_scores = [eval.quality_score for eval in node_evaluations]
                confidences = [eval.confidence for eval in node_evaluations]

                # Weighted average by confidence
                total_confidence = sum(confidences)
                if total_confidence > 0:
                    weighted_quality = sum(
                        score * conf for score, conf in zip(quality_scores, confidences)
                    ) / total_confidence
                else:
                    weighted_quality = sum(quality_scores) / len(quality_scores)

                # Calculate uncertainty as variance in evaluations
                uncertainty = 0.0
                if len(quality_scores) > 1:
                    mean_quality = sum(quality_scores) / len(quality_scores)
                    variance = sum((score - mean_quality) ** 2 for score in quality_scores) / len(quality_scores)
                    uncertainty = min(1.0, variance * 4)  # Scale to 0-1

                node.update_scores(
                    quality=weighted_quality,
                    uncertainty=uncertainty,
                    diversity=0.5  # Placeholder
                )

    def _select_nodes_for_next_iteration(
        self,
        nodes: List[TreeNode],
        context: SearchContext
    ) -> List[TreeNode]:
        """
        Select which nodes to keep active for the next iteration.
        """
        if len(nodes) <= self.max_active_nodes:
            return nodes

        # Sort by cumulative score
        nodes.sort(key=lambda x: x.cumulative_score, reverse=True)

        # Keep top performers
        top_nodes = nodes[:self.max_active_nodes // 2]

        # Add diverse nodes
        remaining_nodes = nodes[self.max_active_nodes // 2:]
        diverse_nodes = self._select_diverse_nodes(
            remaining_nodes,
            self.max_active_nodes - len(top_nodes)
        )

        return top_nodes + diverse_nodes

    def _select_diverse_nodes(self, nodes: List[TreeNode], num_nodes: int) -> List[TreeNode]:
        """
        Select diverse nodes to maintain exploration.
        """
        if not nodes or num_nodes <= 0:
            return []

        selected = []
        remaining = nodes.copy()

        # Select first node randomly from top half
        mid_point = len(remaining) // 2
        if remaining[:mid_point]:
            first_node = remaining.pop(remaining.index(remaining[mid_point // 2]))
            selected.append(first_node)

        # Select remaining nodes to maximize diversity
        while len(selected) < num_nodes and remaining:
            # Simple diversity heuristic: prefer nodes from different agents
            best_node = None
            best_diversity_score = -1

            for node in remaining:
                diversity_score = 0
                # Prefer different agents
                if not any(selected_node.agent_name == node.agent_name for selected_node in selected):
                    diversity_score += 0.5
                # Prefer different branch reasons
                if not any(selected_node.branch_reason == node.branch_reason for selected_node in selected):
                    diversity_score += 0.3

                diversity_score += node.cumulative_score * 0.2  # Still consider quality

                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_node = node

            if best_node:
                selected.append(best_node)
                remaining.remove(best_node)
            else:
                break

        return selected

    def _build_search_result(self, context: SearchContext) -> TreeSearchResult:
        """
        Build the final search result from the search context.
        """
        all_nodes = list(context.node_registry.values())
        search_duration = time.time() - context.search_start_time

        # Find best nodes
        leaf_nodes = context.get_leaf_nodes()
        best_nodes = sorted(all_nodes, key=lambda x: x.cumulative_score, reverse=True)[:5]

        # Find best path
        best_path = []
        if best_nodes:
            best_path = best_nodes[0].get_path_from_root(context.node_registry)

        # Calculate agent statistics
        agent_stats = {}
        for agent_name in self.agent_pool._agents.keys():
            agent = self.agent_pool.get_agent_by_name(agent_name)
            agent_stats[agent_name] = {
                "generation_count": agent.generation_count,
                "evaluation_count": agent.evaluation_count,
                "success_rate": agent.success_rate,
                "role": agent.role.value,
                "specialization": agent.specialization
            }

        return TreeSearchResult(
            best_nodes=[node.to_dict() for node in best_nodes],
            best_path=[node.to_dict() for node in best_path],
            total_nodes_explored=len(all_nodes),
            max_depth_reached=max(node.depth for node in all_nodes) if all_nodes else 0,
            search_duration=search_duration,
            total_agent_calls=self.total_agent_calls,
            best_score=context.best_score,
            average_score=context.average_score,
            diversity_score=context.diversity_score,
            agent_statistics=agent_stats
        )