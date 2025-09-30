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
Agent Tree Search Module

This module provides a flexible and extensible framework for multi-agent tree-based rollout
and search in reinforcement learning environments. It allows multiple specialized agents to
collaborate in exploring solution spaces through tree search algorithms.
"""

from .base_agent import BaseTreeAgent, AgentRole
from .tree_structures import (
    TreeNode, SearchContext, NodeEvaluation, TreeSearchResult,
    TreeGRPONode, TreeGRPOConfig, RewardAggregationConfig,
    GroupingConfig, TreeStructureConfig, TreeStructureBuilder
)
from .agent_pool import AgentPool
from .tree_search_engine import TreeSearchEngine
from .specialized_agents import SpecializedTreeAgent
from .agent_factory import AgentFactory
from .agent_tree_loop import AgentTreeLoop
from .config_templates import (
    get_config_template, get_available_templates,
    get_tree_grpo_config_template, get_tree_grpo_full_config,
    get_available_grpo_templates
)
from .grpo_tree import (
    compute_grpo_tree_advantage, create_tree_grpo_config_template,
    TreeRewardPropagator, TreeGRPOGrouper, TreeGRPOAdvantageComputer
)
from .context_management import (
    ContextCompressionConfig, NodeContext, SimpleContextCompressor,
    AgentInformationFusion, DynamicContextSelector, HierarchicalContextManager
)

# Smolagents integration (optional import)
try:
    from .smolagents_integration import (
        SmolagentsTreeCoordinator, MultiProposalGenerator, ComboNodeBuilder,
        AgentOutput, CombinedOutput
    )
    from .smolagents_tree_orchestrator import (
        SmolagentsTreeOrchestrator, SmolagentsTreeConfig
    )
    SMOLAGENTS_INTEGRATION_AVAILABLE = True
except ImportError:
    SMOLAGENTS_INTEGRATION_AVAILABLE = False

__all__ = [
    # Base classes
    "BaseTreeAgent",
    "AgentRole",

    # Tree structures
    "TreeNode",
    "SearchContext",
    "NodeEvaluation",
    "TreeSearchResult",

    # Tree GRPO classes
    "TreeGRPONode",
    "TreeGRPOConfig",
    "RewardAggregationConfig",
    "GroupingConfig",
    "TreeStructureConfig",
    "TreeStructureBuilder",

    # Core components
    "AgentPool",
    "TreeSearchEngine",
    "SpecializedTreeAgent",
    "AgentFactory",
    "AgentTreeLoop",

    # Configuration templates
    "get_config_template",
    "get_available_templates",
    "get_tree_grpo_config_template",
    "get_tree_grpo_full_config",
    "get_available_grpo_templates",

    # Tree GRPO functionality
    "compute_grpo_tree_advantage",
    "create_tree_grpo_config_template",
    "TreeRewardPropagator",
    "TreeGRPOGrouper",
    "TreeGRPOAdvantageComputer",

    # Context management
    "ContextCompressionConfig",
    "NodeContext",
    "SimpleContextCompressor",
    "AgentInformationFusion",
    "DynamicContextSelector",
    "HierarchicalContextManager",
]

# Add smolagents integration to __all__ if available
if SMOLAGENTS_INTEGRATION_AVAILABLE:
    __all__.extend([
        # Smolagents integration
        "SmolagentsTreeCoordinator",
        "MultiProposalGenerator",
        "ComboNodeBuilder",
        "AgentOutput",
        "CombinedOutput",
        "SmolagentsTreeOrchestrator",
        "SmolagentsTreeConfig",
        "SMOLAGENTS_INTEGRATION_AVAILABLE",
    ])