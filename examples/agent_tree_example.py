#!/usr/bin/env python3
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
Example script demonstrating Agent Tree Rollout functionality.

This script shows how to:
1. Set up multi-agent tree search
2. Configure different agent specializations
3. Run tree search on various problem types
4. Analyze results and performance
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add verl to path
sys.path.append(str(Path(__file__).parent.parent))

from verl.experimental.agent_tree import (
    AgentFactory,
    AgentPool,
    AgentRole,
    TreeSearchEngine,
    TreeNode,
    SearchContext,
    SpecializedTreeAgent
)
from verl.experimental.agent_tree.config_templates import get_config_template

# Mock components for the example (in real usage, these come from verl)
class MockServerManager:
    """Mock server manager for demonstration."""

    async def generate(self, request_id, prompt_ids, sampling_params, image_data=None):
        """Mock generation that returns different responses based on agent type."""
        import random

        # Simulate different agent responses
        if "mathematical" in request_id:
            responses = [
                "Let me solve this step by step: First, I'll identify the variables...",
                "Using algebraic approach: We can set up the equation as...",
                "Applying mathematical principles: The solution involves..."
            ]
        elif "creative" in request_id:
            responses = [
                "Let me think creatively about this: What if we approach it from...",
                "Here's an innovative perspective: We could consider...",
                "Thinking outside the box: An unconventional solution might be..."
            ]
        else:
            responses = [
                "Let me analyze this systematically...",
                "Considering the logical structure...",
                "Breaking down the problem..."
            ]

        # Mock TokenOutput
        class MockTokenOutput:
            def __init__(self, text):
                self.token_ids = list(range(len(text.split())))  # Mock token IDs
                self.text = text

        return MockTokenOutput(random.choice(responses))

class MockTokenizer:
    """Mock tokenizer for demonstration."""

    def encode(self, text, add_special_tokens=True):
        """Mock encoding."""
        return list(range(len(text.split())))

    def decode(self, token_ids, skip_special_tokens=True):
        """Mock decoding."""
        return f"Generated response with {len(token_ids)} tokens"


async def demonstrate_basic_tree_search():
    """Demonstrate basic tree search functionality."""
    print("=== Basic Tree Search Demo ===")

    # Create mock components
    server_manager = MockServerManager()
    tokenizer = MockTokenizer()

    # Create agent pool
    agent_pool = AgentPool()

    # Add specialized agents
    math_agent = SpecializedTreeAgent(
        name="math_specialist",
        role=AgentRole.EXPLORER,
        specialization="mathematical",
        server_manager=server_manager,
        tokenizer=tokenizer
    )

    creative_agent = SpecializedTreeAgent(
        name="creative_thinker",
        role=AgentRole.EXPLORER,
        specialization="creative",
        server_manager=server_manager,
        tokenizer=tokenizer
    )

    evaluator_agent = SpecializedTreeAgent(
        name="quality_evaluator",
        role=AgentRole.EVALUATOR,
        specialization="quality_assessment",
        server_manager=server_manager,
        tokenizer=tokenizer
    )

    # Register agents
    agent_pool.register_agent(math_agent, weight=1.5)
    agent_pool.register_agent(creative_agent, weight=1.2)
    agent_pool.register_agent(evaluator_agent, weight=1.0)

    print(f"Created agent pool with {len(agent_pool)} agents")
    print(f"Pool statistics: {agent_pool.get_pool_statistics()}")

    # Create tree search engine
    search_config = {
        "max_depth": 3,
        "max_branches_per_node": 2,
        "max_active_nodes": 6,
        "search_timeout": 60.0,
        "coordination_strategy": "hierarchical"
    }

    tree_engine = TreeSearchEngine(agent_pool, search_config)

    # Create initial problem
    initial_messages = [
        {"role": "user", "content": "Solve this math problem: If a train travels at 80 mph for 2.5 hours, how far does it travel? Show multiple solution approaches."}
    ]

    initial_node = TreeNode.from_prompt(initial_messages)

    # Run tree search
    print("\nStarting tree search...")

    try:
        result = await tree_engine.search(
            initial_node=initial_node,
            sampling_params={"temperature": 0.7, "max_tokens": 200},
            problem_type="mathematical",
            difficulty_level=0.3
        )

        print(f"\nSearch completed!")
        print(f"Nodes explored: {result.total_nodes_explored}")
        print(f"Max depth reached: {result.max_depth_reached}")
        print(f"Search duration: {result.search_duration:.2f}s")
        print(f"Best score: {result.best_score:.3f}")
        print(f"Best nodes found: {len(result.best_nodes)}")

        # Show agent performance
        print("\nAgent Statistics:")
        for agent_name, stats in result.agent_statistics.items():
            print(f"  {agent_name}: {stats}")

    except Exception as e:
        print(f"Search failed: {e}")


async def demonstrate_config_based_setup():
    """Demonstrate setting up agents from configuration."""
    print("\n=== Configuration-Based Setup Demo ===")

    # Get a predefined configuration template
    config = get_config_template("math")
    print(f"Using math problem configuration template")

    # Create mock components
    server_manager = MockServerManager()
    tokenizer = MockTokenizer()

    # Create agent pool from configuration
    try:
        agent_pool = AgentFactory.create_agent_pool(
            config["agent_tree_config"],
            server_manager,
            tokenizer
        )

        print(f"Created {len(agent_pool)} agents from configuration")
        print(f"Agent pool details: {agent_pool}")

        # Show agent roles
        for role in AgentRole:
            agents = agent_pool.get_agents_by_role(role)
            if agents:
                print(f"  {role.value}: {[agent.name for agent in agents]}")

    except Exception as e:
        print(f"Failed to create agent pool from config: {e}")


def demonstrate_configuration_templates():
    """Show available configuration templates."""
    print("\n=== Available Configuration Templates ===")

    from verl.experimental.agent_tree.config_templates import get_available_templates

    templates = get_available_templates()
    print(f"Available templates: {templates}")

    # Show details of each template
    for template_name in templates:
        try:
            config = get_config_template(template_name)
            tree_config = config.get("tree_config", {})

            print(f"\n{template_name.upper()} template:")
            print(f"  Max depth: {tree_config.get('max_depth', 'N/A')}")
            print(f"  Branches per node: {tree_config.get('max_branches_per_node', 'N/A')}")
            print(f"  Coordination: {tree_config.get('coordination_strategy', 'N/A')}")
            print(f"  Timeout: {tree_config.get('search_timeout', 'N/A')}s")

            # Count agents if available
            agent_config = config.get("agent_tree_config", {}).get("agent_pool", {})
            total_agents = sum(len(agents) for agents in agent_config.values() if isinstance(agents, list))
            if total_agents > 0:
                print(f"  Total agents: {total_agents}")

        except Exception as e:
            print(f"  Error loading {template_name}: {e}")


async def demonstrate_agent_specializations():
    """Demonstrate different agent specializations."""
    print("\n=== Agent Specialization Demo ===")

    server_manager = MockServerManager()
    tokenizer = MockTokenizer()

    # Test different specializations
    specializations = [
        ("mathematical", "Solve complex equations"),
        ("creative", "Design innovative solutions"),
        ("logical", "Verify argument validity"),
        ("analytical", "Break down complex problems")
    ]

    for spec, task_description in specializations:
        print(f"\n{spec.upper()} Agent:")

        agent = SpecializedTreeAgent(
            name=f"{spec}_agent",
            role=AgentRole.EXPLORER,
            specialization=spec,
            server_manager=server_manager,
            tokenizer=tokenizer
        )

        # Create a test node
        test_messages = [
            {"role": "user", "content": f"Task: {task_description}"}
        ]
        test_node = TreeNode.from_prompt(test_messages)

        # Test the agent's prompt template
        prompt_template = agent.get_specialization_prompt("branch")
        print(f"  Prompt template: {prompt_template[:100]}...")

        # Test expansion decision
        context = SearchContext(problem_type=spec, difficulty_level=0.5)
        try:
            should_expand = await agent.should_expand(test_node, context)
            print(f"  Should expand: {should_expand}")
        except Exception as e:
            print(f"  Expansion check failed: {e}")


def main():
    """Main demonstration function."""
    print("Agent Tree Rollout Demonstration")
    print("=" * 50)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    async def run_demos():
        """Run all demonstrations."""
        await demonstrate_basic_tree_search()
        await demonstrate_config_based_setup()
        demonstrate_configuration_templates()
        await demonstrate_agent_specializations()

        print("\n" + "=" * 50)
        print("Demonstration completed!")
        print("\nTo use Agent Tree Rollout in your verl training:")
        print("1. Use the configuration example in examples/agent_tree_config_example.yaml")
        print("2. Set rollout.name to 'agent_tree' in your config")
        print("3. Configure the agent_tree_config section with your desired agents")
        print("4. Run your training as usual - tree search will be used automatically")

    # Run the async demonstrations
    asyncio.run(run_demos())


if __name__ == "__main__":
    main()