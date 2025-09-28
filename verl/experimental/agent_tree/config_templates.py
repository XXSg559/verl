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
Configuration templates for agent tree search.

This module provides predefined configuration templates for different use cases
and problem types.
"""

from typing import Any, Dict


def get_default_tree_config() -> Dict[str, Any]:
    """Get default tree search configuration."""
    return {
        "enable": True,
        "max_depth": 4,
        "max_branches_per_node": 3,
        "max_active_nodes": 8,
        "search_timeout": 180.0,
        "coordination_strategy": "hierarchical",
        "fallback_to_single_agent": True
    }


def get_math_problem_config() -> Dict[str, Any]:
    """Configuration optimized for mathematical problem solving."""
    return {
        "tree_config": {
            "enable": True,
            "max_depth": 5,
            "max_branches_per_node": 4,
            "max_active_nodes": 12,
            "search_timeout": 300.0,
            "coordination_strategy": "hierarchical"
        },
        "agent_tree_config": {
            "agent_pool": {
                "explorers": [
                    {
                        "name": "algebra_specialist",
                        "type": "specialized",
                        "role": "EXPLORER",
                        "specialization": "mathematical",
                        "temperature": 0.3,
                        "max_tokens": 400,
                        "tools": ["calculator", "symbolic_math"],
                        "weight": 1.5
                    },
                    {
                        "name": "geometry_specialist",
                        "type": "specialized",
                        "role": "EXPLORER",
                        "specialization": "geometric",
                        "temperature": 0.4,
                        "max_tokens": 350,
                        "tools": ["geometric_calculator"],
                        "weight": 1.2
                    },
                    {
                        "name": "logic_verifier",
                        "type": "specialized",
                        "role": "EXPLORER",
                        "specialization": "logical",
                        "temperature": 0.2,
                        "max_tokens": 300,
                        "tools": ["logic_checker"],
                        "weight": 1.0
                    }
                ],
                "evaluators": [
                    {
                        "name": "math_accuracy_judge",
                        "type": "specialized",
                        "role": "EVALUATOR",
                        "specialization": "mathematical",
                        "temperature": 0.2,
                        "max_tokens": 200,
                        "criteria": ["accuracy", "completeness", "clarity"],
                        "weight": 1.0
                    }
                ],
                "coordinators": [
                    {
                        "name": "math_coordinator",
                        "type": "specialized",
                        "role": "COORDINATOR",
                        "specialization": "mathematical",
                        "temperature": 0.3,
                        "max_tokens": 250
                    }
                ]
            }
        }
    }


def get_creative_task_config() -> Dict[str, Any]:
    """Configuration optimized for creative tasks."""
    return {
        "tree_config": {
            "enable": True,
            "max_depth": 4,
            "max_branches_per_node": 5,
            "max_active_nodes": 15,
            "search_timeout": 240.0,
            "coordination_strategy": "collaborative"
        },
        "agent_tree_config": {
            "agent_pool": {
                "explorers": [
                    {
                        "name": "creative_innovator",
                        "type": "specialized",
                        "role": "EXPLORER",
                        "specialization": "creative",
                        "temperature": 1.0,
                        "max_tokens": 350,
                        "tools": ["brainstorming", "ideation"],
                        "weight": 1.5
                    },
                    {
                        "name": "practical_evaluator",
                        "type": "specialized",
                        "role": "EXPLORER",
                        "specialization": "practical",
                        "temperature": 0.6,
                        "max_tokens": 300,
                        "tools": ["feasibility_check"],
                        "weight": 1.0
                    },
                    {
                        "name": "artistic_designer",
                        "type": "specialized",
                        "role": "EXPLORER",
                        "specialization": "artistic",
                        "temperature": 0.9,
                        "max_tokens": 400,
                        "tools": ["visual_design", "aesthetic_evaluation"],
                        "weight": 1.2
                    }
                ],
                "evaluators": [
                    {
                        "name": "creativity_judge",
                        "type": "specialized",
                        "role": "EVALUATOR",
                        "specialization": "creative",
                        "criteria": ["originality", "feasibility", "impact"],
                        "temperature": 0.4,
                        "max_tokens": 200
                    },
                    {
                        "name": "market_evaluator",
                        "type": "specialized",
                        "role": "EVALUATOR",
                        "specialization": "market_analysis",
                        "criteria": ["market_potential", "user_appeal"],
                        "temperature": 0.3,
                        "max_tokens": 180
                    }
                ]
            }
        }
    }


def get_coding_problem_config() -> Dict[str, Any]:
    """Configuration optimized for coding problems."""
    return {
        "tree_config": {
            "enable": True,
            "max_depth": 4,
            "max_branches_per_node": 3,
            "max_active_nodes": 10,
            "search_timeout": 300.0,
            "coordination_strategy": "hierarchical"
        },
        "agent_tree_config": {
            "agent_pool": {
                "explorers": [
                    {
                        "name": "algorithm_designer",
                        "type": "specialized",
                        "role": "EXPLORER",
                        "specialization": "algorithmic",
                        "temperature": 0.4,
                        "max_tokens": 500,
                        "tools": ["code_execution", "algorithm_analysis"],
                        "weight": 1.5
                    },
                    {
                        "name": "code_optimizer",
                        "type": "specialized",
                        "role": "EXPLORER",
                        "specialization": "optimization",
                        "temperature": 0.3,
                        "max_tokens": 400,
                        "tools": ["performance_profiler", "code_analysis"],
                        "weight": 1.2
                    },
                    {
                        "name": "edge_case_tester",
                        "type": "specialized",
                        "role": "EXPLORER",
                        "specialization": "testing",
                        "temperature": 0.5,
                        "max_tokens": 300,
                        "tools": ["test_generator", "edge_case_finder"],
                        "weight": 1.0
                    }
                ],
                "evaluators": [
                    {
                        "name": "code_quality_judge",
                        "type": "specialized",
                        "role": "EVALUATOR",
                        "specialization": "code_quality",
                        "criteria": ["correctness", "efficiency", "readability"],
                        "temperature": 0.2,
                        "max_tokens": 250
                    }
                ]
            }
        }
    }


def get_research_analysis_config() -> Dict[str, Any]:
    """Configuration optimized for research and analysis tasks."""
    return {
        "tree_config": {
            "enable": True,
            "max_depth": 5,
            "max_branches_per_node": 4,
            "max_active_nodes": 16,
            "search_timeout": 400.0,
            "coordination_strategy": "collaborative"
        },
        "agent_tree_config": {
            "agent_pool": {
                "explorers": [
                    {
                        "name": "data_analyst",
                        "type": "specialized",
                        "role": "EXPLORER",
                        "specialization": "data_analysis",
                        "temperature": 0.4,
                        "max_tokens": 400,
                        "tools": ["data_processing", "statistical_analysis"],
                        "weight": 1.3
                    },
                    {
                        "name": "research_synthesizer",
                        "type": "specialized",
                        "role": "EXPLORER",
                        "specialization": "research",
                        "temperature": 0.5,
                        "max_tokens": 450,
                        "tools": ["literature_search", "citation_manager"],
                        "weight": 1.4
                    },
                    {
                        "name": "critical_reviewer",
                        "type": "specialized",
                        "role": "EXPLORER",
                        "specialization": "critical_analysis",
                        "temperature": 0.3,
                        "max_tokens": 350,
                        "tools": ["bias_detector", "fact_checker"],
                        "weight": 1.1
                    }
                ],
                "evaluators": [
                    {
                        "name": "methodology_evaluator",
                        "type": "specialized",
                        "role": "EVALUATOR",
                        "specialization": "methodology",
                        "criteria": ["rigor", "validity", "reproducibility"],
                        "temperature": 0.2,
                        "max_tokens": 300
                    },
                    {
                        "name": "evidence_evaluator",
                        "type": "specialized",
                        "role": "EVALUATOR",
                        "specialization": "evidence",
                        "criteria": ["evidence_quality", "source_reliability"],
                        "temperature": 0.2,
                        "max_tokens": 250
                    }
                ],
                "synthesizers": [
                    {
                        "name": "research_synthesizer",
                        "type": "specialized",
                        "role": "SYNTHESIZER",
                        "specialization": "synthesis",
                        "temperature": 0.4,
                        "max_tokens": 500
                    }
                ]
            }
        }
    }


def get_competitive_search_config() -> Dict[str, Any]:
    """Configuration for competitive search strategy."""
    return {
        "tree_config": {
            "enable": True,
            "max_depth": 3,
            "max_branches_per_node": 6,
            "max_active_nodes": 12,
            "search_timeout": 200.0,
            "coordination_strategy": "competitive"
        },
        "agent_tree_config": {
            "agent_pool": {
                "explorers": [
                    {
                        "name": f"competitor_{i}",
                        "type": "specialized",
                        "role": "EXPLORER",
                        "specialization": ["creative", "logical", "practical", "innovative"][i % 4],
                        "temperature": 0.3 + i * 0.2,
                        "max_tokens": 300,
                        "weight": 1.0
                    }
                    for i in range(6)
                ],
                "evaluators": [
                    {
                        "name": "competition_judge",
                        "type": "specialized",
                        "role": "EVALUATOR",
                        "specialization": "competitive_evaluation",
                        "criteria": ["quality", "innovation", "effectiveness"],
                        "temperature": 0.2,
                        "max_tokens": 200
                    }
                ]
            }
        }
    }


def get_minimal_config() -> Dict[str, Any]:
    """Minimal configuration for basic tree search."""
    return {
        "tree_config": {
            "enable": True,
            "max_depth": 3,
            "max_branches_per_node": 2,
            "max_active_nodes": 4,
            "search_timeout": 120.0,
            "coordination_strategy": "hierarchical"
        },
        "agent_tree_config": {
            "agent_pool": {
                "explorers": [
                    {
                        "name": "general_explorer",
                        "type": "specialized",
                        "role": "EXPLORER",
                        "specialization": "general",
                        "temperature": 0.7,
                        "max_tokens": 200
                    }
                ],
                "evaluators": [
                    {
                        "name": "general_evaluator",
                        "type": "specialized",
                        "role": "EVALUATOR",
                        "specialization": "general",
                        "temperature": 0.3,
                        "max_tokens": 150
                    }
                ]
            }
        }
    }


def get_config_template(template_name: str) -> Dict[str, Any]:
    """
    Get a configuration template by name.

    Args:
        template_name: Name of the template

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If template name is not found
    """
    templates = {
        "default": get_default_tree_config,
        "math": get_math_problem_config,
        "creative": get_creative_task_config,
        "coding": get_coding_problem_config,
        "research": get_research_analysis_config,
        "competitive": get_competitive_search_config,
        "minimal": get_minimal_config
    }

    if template_name not in templates:
        available = list(templates.keys())
        raise ValueError(f"Unknown template: {template_name}. Available templates: {available}")

    return templates[template_name]()


def get_tree_grpo_config_template(template_name: str) -> Dict[str, Any]:
    """
    Get a tree GRPO configuration template.

    Args:
        template_name: Name of the GRPO template

    Returns:
        Tree GRPO configuration dictionary
    """
    if template_name == "binary_tree":
        # Classic 1→2→4→8 pattern
        return {
            "tree_grpo_config": {
                "tree_structure": {
                    "max_depth": 3,
                    "uniform_branching": 2
                },
                "reward_aggregation": {
                    "method": "mean_minus_std",
                    "std_penalty": 0.1
                },
                "grouping": {
                    "method": "by_parent_and_depth",
                    "min_group_size": 2,
                    "max_group_size": 8
                },
                "separate_depth_training": True,
                "norm_adv_by_std": True
            }
        }

    elif template_name == "ternary_tree":
        # 1→3→9→27 pattern
        return {
            "tree_grpo_config": {
                "tree_structure": {
                    "max_depth": 3,
                    "uniform_branching": 3
                },
                "reward_aggregation": {
                    "method": "weighted_stats",
                    "weights": {
                        "mean": 1.0,
                        "std": -0.15,
                        "max": 0.1,
                        "min": -0.05
                    }
                },
                "grouping": {
                    "method": "by_parent_and_depth"
                },
                "separate_depth_training": True
            }
        }

    elif template_name == "asymmetric":
        # Custom asymmetric pattern: 1→3→6→24→24
        return {
            "tree_grpo_config": {
                "tree_structure": {
                    "max_depth": 4,
                    "branching_schedule": [3, 2, 4, 1]
                },
                "reward_aggregation": {
                    "method": "risk_adjusted",
                    "risk_free_rate": 0.0
                },
                "grouping": {
                    "method": "by_depth_only"
                },
                "separate_depth_training": True
            }
        }

    elif template_name == "conservative":
        # Risk-averse configuration
        return {
            "tree_grpo_config": {
                "tree_structure": {
                    "max_depth": 2,
                    "uniform_branching": 2
                },
                "reward_aggregation": {
                    "method": "mean_minus_std",
                    "std_penalty": 0.3  # Heavy penalty for variance
                },
                "grouping": {
                    "method": "by_parent_and_depth"
                },
                "separate_depth_training": True,
                "norm_adv_by_std": True
            }
        }

    elif template_name == "exploratory":
        # Exploration-focused configuration
        return {
            "tree_grpo_config": {
                "tree_structure": {
                    "max_depth": 3,
                    "uniform_branching": 4  # More branches for exploration
                },
                "reward_aggregation": {
                    "method": "mean_plus_std",
                    "std_bonus": 0.2  # Reward diversity
                },
                "grouping": {
                    "method": "by_reward_range",
                    "reward_range_bins": 5
                },
                "separate_depth_training": True
            }
        }

    elif template_name == "math_grpo":
        # Mathematical problem solving with GRPO tree
        return {
            "tree_grpo_config": {
                "tree_structure": {
                    "max_depth": 3,
                    "uniform_branching": 2
                },
                "reward_aggregation": {
                    "method": "mean_minus_std",
                    "std_penalty": 0.05  # Low penalty, focus on accuracy
                },
                "grouping": {
                    "method": "by_parent_and_depth"
                },
                "separate_depth_training": True
            },
            "agent_tree_config": {
                "agent_pool": {
                    "explorers": [
                        {
                            "name": "math_explorer_1",
                            "type": "specialized",
                            "role": "EXPLORER",
                            "specialization": "mathematical",
                            "temperature": 0.3,
                            "max_tokens": 300,
                            "weight": 1.0
                        },
                        {
                            "name": "math_explorer_2",
                            "type": "specialized",
                            "role": "EXPLORER",
                            "specialization": "mathematical",
                            "temperature": 0.5,
                            "max_tokens": 300,
                            "weight": 1.0
                        }
                    ]
                }
            }
        }

    else:
        raise ValueError(f"Unknown tree GRPO template: {template_name}. "
                        f"Available: binary_tree, ternary_tree, asymmetric, conservative, "
                        f"exploratory, math_grpo")


def get_tree_grpo_full_config(template_name: str = "binary_tree") -> Dict[str, Any]:
    """
    Get a complete configuration for tree GRPO training.

    Args:
        template_name: Name of the GRPO template to use

    Returns:
        Complete configuration dictionary for verl training with tree GRPO
    """
    base_config = {
        "actor_rollout_ref": {
            "rollout": {
                "name": "agent_tree",
                "tree_config": {
                    "enable": False,  # Disable regular tree search
                    "fallback_to_single_agent": True
                }
            }
        },
        "algorithm": {
            "name": "grpo",
            "grpo": {
                "kl_coeff": 0.05,
                "temperature": 0.8,
                "advantage_estimator": "grpo_tree"
            }
        },
        "data": {
            "max_prompt_length": 1024,
            "max_response_length": 1024,
            "train_batch_size": 64,
            "return_raw_chat": True,
            "shuffle": False
        }
    }

    # Get tree GRPO specific configuration
    grpo_config = get_tree_grpo_config_template(template_name)

    # Merge configurations
    base_config["actor_rollout_ref"]["rollout"].update(grpo_config)

    return base_config


def get_available_templates() -> list[str]:
    """Get list of available configuration templates."""
    return ["default", "math", "creative", "coding", "research", "competitive", "minimal"]


def get_available_grpo_templates() -> list[str]:
    """Get list of available tree GRPO configuration templates."""
    return ["binary_tree", "ternary_tree", "asymmetric", "conservative", "exploratory", "math_grpo"]