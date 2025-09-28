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
from typing import Any, Callable, Dict, List, Optional

from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager

from .agent_pool import AgentPool
from .base_agent import BaseTreeAgent, AgentRole
from .specialized_agents import SpecializedTreeAgent

logger = logging.getLogger(__name__)


class AgentFactory:
    """
    Factory class for creating and configuring tree search agents.

    This factory provides a flexible way to create different types of agents
    based on configuration, allowing for easy extension and customization.
    """

    # Registry of agent types and their creation functions
    _agent_registry: Dict[str, Callable] = {}

    @classmethod
    def register_agent_type(
        cls,
        agent_type: str,
        creation_function: Callable
    ):
        """
        Register a new agent type with its creation function.

        Args:
            agent_type: Unique identifier for the agent type
            creation_function: Function that creates an agent of this type
        """
        cls._agent_registry[agent_type] = creation_function
        logger.info(f"Registered agent type: {agent_type}")

    @classmethod
    def create_agent(
        cls,
        agent_config: Dict[str, Any],
        server_manager: AsyncLLMServerManager,
        tokenizer: Any,
        **kwargs
    ) -> BaseTreeAgent:
        """
        Create an agent based on the provided configuration.

        Args:
            agent_config: Configuration dictionary for the agent
            server_manager: AsyncLLMServerManager for text generation
            tokenizer: Tokenizer for text processing
            **kwargs: Additional arguments

        Returns:
            Created agent instance

        Raises:
            ValueError: If agent type is not supported
        """
        agent_type = agent_config.get("type", "specialized")

        # Check if agent type is registered
        if agent_type in cls._agent_registry:
            return cls._agent_registry[agent_type](
                agent_config, server_manager, tokenizer, **kwargs
            )

        # Built-in agent types
        if agent_type == "specialized" or agent_type == "tool_agent":
            return cls._create_specialized_agent(agent_config, server_manager, tokenizer)
        elif agent_type == "coordinator":
            return cls._create_coordinator_agent(agent_config, server_manager, tokenizer)
        elif agent_type == "evaluator":
            return cls._create_evaluator_agent(agent_config, server_manager, tokenizer)
        elif agent_type == "synthesizer":
            return cls._create_synthesizer_agent(agent_config, server_manager, tokenizer)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    @classmethod
    def create_agent_pool(
        cls,
        config: Dict[str, Any],
        server_manager: AsyncLLMServerManager,
        tokenizer: Any,
        **kwargs
    ) -> AgentPool:
        """
        Create an agent pool from configuration.

        Args:
            config: Configuration dictionary containing agent definitions
            server_manager: AsyncLLMServerManager for text generation
            tokenizer: Tokenizer for text processing
            **kwargs: Additional arguments

        Returns:
            Configured agent pool
        """
        pool = AgentPool()

        agent_pool_config = config.get("agent_pool", {})

        # Create agents for each role category
        for role_name, agent_configs in agent_pool_config.items():
            if not isinstance(agent_configs, list):
                continue

            for agent_config in agent_configs:
                try:
                    # Ensure agent has a role
                    if "role" not in agent_config:
                        agent_config["role"] = role_name.rstrip('s').upper()  # explorers -> EXPLORER

                    agent = cls.create_agent(
                        agent_config,
                        server_manager,
                        tokenizer,
                        **kwargs
                    )

                    weight = agent_config.get("weight", 1.0)
                    pool.register_agent(agent, weight)

                except Exception as e:
                    logger.error(f"Failed to create agent from config {agent_config}: {e}")
                    continue

        logger.info(f"Created agent pool with {len(pool)} agents")
        return pool

    @classmethod
    def _create_specialized_agent(
        cls,
        config: Dict[str, Any],
        server_manager: AsyncLLMServerManager,
        tokenizer: Any
    ) -> SpecializedTreeAgent:
        """Create a specialized agent."""
        name = config.get("name", "specialized_agent")
        role_str = config.get("role", "EXPLORER")
        role = AgentRole(role_str.upper()) if isinstance(role_str, str) else role_str
        specialization = config.get("specialization", "general")
        model_path = config.get("model", config.get("model_path"))
        tools = config.get("tools", [])

        return SpecializedTreeAgent(
            name=name,
            role=role,
            specialization=specialization,
            server_manager=server_manager,
            tokenizer=tokenizer,
            model_path=model_path,
            tools=tools,
            config=config
        )

    @classmethod
    def _create_coordinator_agent(
        cls,
        config: Dict[str, Any],
        server_manager: AsyncLLMServerManager,
        tokenizer: Any
    ) -> SpecializedTreeAgent:
        """Create a coordinator agent."""
        config_copy = config.copy()
        config_copy.update({
            "role": "COORDINATOR",
            "specialization": config.get("specialization", "coordination")
        })
        return cls._create_specialized_agent(config_copy, server_manager, tokenizer)

    @classmethod
    def _create_evaluator_agent(
        cls,
        config: Dict[str, Any],
        server_manager: AsyncLLMServerManager,
        tokenizer: Any
    ) -> SpecializedTreeAgent:
        """Create an evaluator agent."""
        config_copy = config.copy()
        config_copy.update({
            "role": "EVALUATOR",
            "specialization": config.get("specialization", "evaluation")
        })
        return cls._create_specialized_agent(config_copy, server_manager, tokenizer)

    @classmethod
    def _create_synthesizer_agent(
        cls,
        config: Dict[str, Any],
        server_manager: AsyncLLMServerManager,
        tokenizer: Any
    ) -> SpecializedTreeAgent:
        """Create a synthesizer agent."""
        config_copy = config.copy()
        config_copy.update({
            "role": "SYNTHESIZER",
            "specialization": config.get("specialization", "synthesis")
        })
        return cls._create_specialized_agent(config_copy, server_manager, tokenizer)

    @classmethod
    def create_predefined_agents(
        cls,
        server_manager: AsyncLLMServerManager,
        tokenizer: Any,
        specializations: Optional[List[str]] = None
    ) -> AgentPool:
        """
        Create a set of predefined agents for common use cases.

        Args:
            server_manager: AsyncLLMServerManager for text generation
            tokenizer: Tokenizer for text processing
            specializations: List of specializations to create

        Returns:
            Agent pool with predefined agents
        """
        if specializations is None:
            specializations = ["mathematical", "creative", "logical", "general"]

        pool = AgentPool()

        # Create explorer agents for each specialization
        for spec in specializations:
            explorer_config = {
                "name": f"{spec}_explorer",
                "type": "specialized",
                "role": "EXPLORER",
                "specialization": spec,
                "temperature": 0.8 if spec == "creative" else 0.5,
                "max_tokens": 300
            }

            explorer = cls.create_agent(explorer_config, server_manager, tokenizer)
            pool.register_agent(explorer, weight=1.0)

        # Create evaluator agents
        evaluator_configs = [
            {
                "name": "quality_evaluator",
                "type": "specialized",
                "role": "EVALUATOR",
                "specialization": "quality_assessment",
                "temperature": 0.3,
                "max_tokens": 150
            },
            {
                "name": "accuracy_evaluator",
                "type": "specialized",
                "role": "EVALUATOR",
                "specialization": "accuracy_check",
                "temperature": 0.2,
                "max_tokens": 100
            }
        ]

        for eval_config in evaluator_configs:
            evaluator = cls.create_agent(eval_config, server_manager, tokenizer)
            pool.register_agent(evaluator, weight=1.0)

        # Create a coordinator agent
        coordinator_config = {
            "name": "search_coordinator",
            "type": "specialized",
            "role": "COORDINATOR",
            "specialization": "coordination",
            "temperature": 0.4,
            "max_tokens": 200
        }

        coordinator = cls.create_agent(coordinator_config, server_manager, tokenizer)
        pool.register_agent(coordinator, weight=1.0)

        # Create a synthesizer agent
        synthesizer_config = {
            "name": "result_synthesizer",
            "type": "specialized",
            "role": "SYNTHESIZER",
            "specialization": "synthesis",
            "temperature": 0.4,
            "max_tokens": 250
        }

        synthesizer = cls.create_agent(synthesizer_config, server_manager, tokenizer)
        pool.register_agent(synthesizer, weight=1.0)

        logger.info(f"Created predefined agent pool with {len(pool)} agents")
        return pool

    @classmethod
    def get_available_agent_types(cls) -> List[str]:
        """Get list of all available agent types."""
        built_in_types = ["specialized", "tool_agent", "coordinator", "evaluator", "synthesizer"]
        registered_types = list(cls._agent_registry.keys())
        return built_in_types + registered_types

    @classmethod
    def validate_agent_config(cls, agent_config: Dict[str, Any]) -> bool:
        """
        Validate an agent configuration.

        Args:
            agent_config: Configuration to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        required_fields = ["name", "type"]

        for field in required_fields:
            if field not in agent_config:
                logger.error(f"Missing required field in agent config: {field}")
                return False

        agent_type = agent_config["type"]
        available_types = cls.get_available_agent_types()

        if agent_type not in available_types:
            logger.error(f"Unknown agent type: {agent_type}. Available types: {available_types}")
            return False

        # Validate role if specified
        if "role" in agent_config:
            role_str = agent_config["role"]
            try:
                AgentRole(role_str.upper())
            except ValueError:
                logger.error(f"Invalid agent role: {role_str}")
                return False

        return True

    @classmethod
    def create_agent_from_template(
        cls,
        template_name: str,
        name: str,
        server_manager: AsyncLLMServerManager,
        tokenizer: Any,
        **overrides
    ) -> BaseTreeAgent:
        """
        Create an agent from a predefined template.

        Args:
            template_name: Name of the template to use
            name: Name for the new agent
            server_manager: AsyncLLMServerManager for text generation
            tokenizer: Tokenizer for text processing
            **overrides: Configuration overrides

        Returns:
            Created agent instance
        """
        templates = {
            "math_explorer": {
                "type": "specialized",
                "role": "EXPLORER",
                "specialization": "mathematical",
                "temperature": 0.3,
                "max_tokens": 400,
                "tools": ["calculator", "symbolic_math"]
            },
            "creative_explorer": {
                "type": "specialized",
                "role": "EXPLORER",
                "specialization": "creative",
                "temperature": 1.0,
                "max_tokens": 300,
                "tools": ["brainstorming", "ideation"]
            },
            "logical_analyzer": {
                "type": "specialized",
                "role": "EVALUATOR",
                "specialization": "logical",
                "temperature": 0.2,
                "max_tokens": 200,
                "tools": ["logic_checker", "argument_validator"]
            },
            "quality_judge": {
                "type": "specialized",
                "role": "EVALUATOR",
                "specialization": "quality_assessment",
                "temperature": 0.3,
                "max_tokens": 150,
                "tools": ["quality_metrics", "completeness_checker"]
            }
        }

        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")

        config = templates[template_name].copy()
        config["name"] = name
        config.update(overrides)

        return cls.create_agent(config, server_manager, tokenizer)