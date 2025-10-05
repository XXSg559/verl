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
Simple async agent coordination utilities.

This module provides minimal utilities for coordinating AsyncCodeAgent instances
without complex wrapper classes or configuration management.
"""

import asyncio
import logging
from typing import List

logger = logging.getLogger(__name__)


async def coordinate_async_agents(
    agents: List,
    task: str,
    max_steps: int = 5
) -> List[str]:
    """
    Simple function to coordinate multiple AsyncCodeAgent instances.

    Args:
        agents: List of AsyncCodeAgent objects from smolagents
        task: Task description
        max_steps: Maximum steps for coordination

    Returns:
        List of results from each agent
    """
    logger.info(f"Coordinating {len(agents)} agents for task: {task[:50]}...")

    # Execute all agents concurrently
    tasks = []
    for agent in agents:
        # Create agent task
        agent_task = agent.execute_sequence_async([task], max_steps=max_steps)
        tasks.append(agent_task)

    # Run all agents concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Extract responses
    responses = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            responses.append(f"Agent {agents[i].agent_id} failed: {str(result)}")
        elif hasattr(result, 'content'):
            responses.append(result.content)
        else:
            responses.append(str(result))

    return responses


async def run_demo():
    """Simple demo of async agent coordination."""
    print("🚀 Simple Async Agent Coordination Demo")
    print("=" * 50)

    # This would require actual agents to be created
    print("Create AsyncCodeAgent instances and use coordinate_async_agents() to coordinate them!")
    print("Example:")
    print("""
    from smolagents import AsyncCodeAgent
    from verl.experimental.agent_tree.async_batch import coordinate_async_agents

    # Create agents with different capabilities
    agent1 = AsyncCodeAgent(tools=[review_tool], model=model, agent_id="reviewer")
    agent2 = AsyncCodeAgent(tools=[test_tool], model=model, agent_id="tester")

    # Coordinate them
    results = await coordinate_async_agents([agent1, agent2], "Review this code")
    """)