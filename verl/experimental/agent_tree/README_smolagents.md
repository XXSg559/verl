# Smolagents + Tree GRPO Integration

This document describes the integration between [smolagents](https://github.com/huggingface/smolagents) and our Tree GRPO system, enabling powerful multi-agent coordination while maintaining node-level training semantics.

## Overview

The integration combines:
- **smolagents**: Mature multi-agent coordination with managed_agents architecture
- **Tree GRPO**: Node-level training with hierarchical reward propagation
- **Combination nodes**: Cartesian product of agent outputs for rich credit assignment

## Architecture

### Three-Layer Design

1. **Coordination Layer**: smolagents CodeAgent as tree coordinator/planner
2. **Generation Layer**: smolagents managed_agents generate multiple proposals
3. **Training Layer**: Combination results form TreeGRPONode instances

### Key Components

#### SmolagentsTreeCoordinator
```python
coordinator = SmolagentsTreeCoordinator(
    smolagents_agent=code_agent_with_managed_agents,
    coordination_strategy="hierarchical",
    enable_silent_mode=True
)
```

Analyzes tree context and creates coordination plans for managed agents.

#### MultiProposalGenerator
```python
generator = MultiProposalGenerator(managed_agent)
generator.set_base_memory(parent_context)
proposals = generator.generate_multiple_proposals(task, n=2)
```

Generates multiple isolated proposals with memory isolation.

#### ComboNodeBuilder
```python
builder = ComboNodeBuilder(fusion_strategy="structured_combination")
combo_nodes = builder.create_combo_nodes(parent_node, agent_outputs, search_context)
```

Creates TreeGRPONode instances from cartesian product of agent outputs.

#### SmolagentsTreeOrchestrator
```python
orchestrator = SmolagentsTreeOrchestrator(coordinator_agent, config)
child_nodes = orchestrator.expand_tree_node(parent_node, search_context)
```

Main orchestrator coordinating the complete workflow.

## Core Workflow

### 1. Tree Node Expansion
```python
def expand_tree_node(parent_node, search_context):
    # 1. Coordinator analyzes context and creates plan
    plan = coordinator.plan_expansion(parent_node, search_context)

    # 2. Each managed agent generates multiple proposals (memory isolated)
    agent_outputs = {}
    for agent_name, task in plan.items():
        outputs = generate_multiple_proposals(agent, task, n=2)
        agent_outputs[agent_name] = outputs

    # 3. Cartesian product combination
    combo_nodes = []
    for combo in itertools.product(*agent_outputs.values()):
        combined_output = fuse_agent_outputs(combo)
        node = TreeGRPONode.from_combined_output(
            parent=parent_node,
            combined_output=combined_output,
            agent_contributions=dict(zip(agent_outputs.keys(), combo))
        )
        combo_nodes.append(node)

    return combo_nodes
```

### 2. Information Sharing
- **Unified memory**: Coordinator maintains complete conversation history
- **Natural attention**: LLM automatically attends to relevant information
- **No explicit messaging**: Information flows through coordinator's memory

### 3. Credit Assignment
```python
# Each combo node stores agent contributions
combo_node.agent_contributions = {
    "research_agent": AgentOutput(...),
    "coding_agent": AgentOutput(...),
    "reasoning_agent": AgentOutput(...)
}

# Reward decomposition
def decompose_node_reward(node_reward):
    agent_rewards = {}
    for agent_name, contribution in node.agent_contributions.items():
        agent_rewards[agent_name] = calculate_contribution_reward(
            agent_name, contribution, node_reward
        )
    return agent_rewards
```

## Example Usage

### Basic Setup
```python
from verl.experimental.agent_tree import (
    SmolagentsTreeOrchestrator, SmolagentsTreeConfig
)
from smolagents import CodeAgent, ToolCallingAgent, LiteLLMModel

# Create model
model = LiteLLMModel(model_id="openai/gpt-4")

# Create specialized managed agents
research_agent = ToolCallingAgent(
    tools=[WebSearchTool()],
    model=model,
    name="research_agent",
    description="Specializes in research and information gathering",
    verbosity_level=0  # Silent mode
)

coding_agent = ToolCallingAgent(
    tools=[PythonInterpreterTool()],
    model=model,
    name="coding_agent",
    description="Specializes in code generation and analysis",
    verbosity_level=0
)

# Create coordinator with managed agents
coordinator = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[research_agent, coding_agent],
    name="coordinator"
)

# Setup orchestrator
config = SmolagentsTreeConfig(
    proposals_per_agent=2,
    fusion_strategy="structured_combination"
)

orchestrator = SmolagentsTreeOrchestrator(coordinator, config)
```

### Tree Expansion
```python
# Create initial tree structure
root_node = TreeGRPONode.from_prompt(messages=initial_messages)
search_context = SearchContext(max_depth=4, max_branches_per_node=4)

# Expand tree
child_nodes = orchestrator.expand_tree_node(root_node, search_context)

# Results: If 2 agents each generate 2 proposals → 4 combination nodes
# combo_1: (research_output_1, coding_output_1)
# combo_2: (research_output_1, coding_output_2)
# combo_3: (research_output_2, coding_output_1)
# combo_4: (research_output_2, coding_output_2)
```

### Training Integration
```python
# After tree expansion, compute rewards
final_rewards = {node.node_id: evaluate_node(node) for node in leaf_nodes}

# Propagate rewards up the tree (existing tree GRPO mechanism)
tree_rewards = propagate_rewards(leaf_rewards, tree_structure)

# Decompose combination node rewards to individual agents
agent_rewards = orchestrator.decompose_rewards_for_tree(
    tree_nodes=all_combo_nodes,
    final_rewards=tree_rewards
)

# Use for GRPO training
train_grpo_with_agent_rewards(agent_rewards)
```

## Configuration

See `examples/smolagents_tree_grpo_config.yaml` for complete configuration options.

Key settings:
- `proposals_per_agent`: Number of proposals each agent generates
- `fusion_strategy`: How to combine agent outputs ("simple_concat", "structured_combination")
- `coordination_strategy`: How coordinator plans agent deployment
- `enable_memory_isolation`: Ensure independent proposal generation

## Benefits

### 1. Rich Credit Assignment
Each agent's proposal is tested in multiple combinations, providing rich learning signals:
```
research_agent_proposal_1 appears in combos with:
- coding_agent_proposal_1 → reward_1
- coding_agent_proposal_2 → reward_2

This enables learning: "research proposal 1 works better with coding approach 2"
```

### 2. Collaboration Learning
The system learns effective agent collaboration patterns:
- Which agent combinations work well together
- When to use different agents for different contexts
- How to coordinate agents for complex tasks

### 3. Natural Information Flow
- No complex messaging protocols needed
- LLM naturally attends to relevant information
- Leverages smolagents' proven coordination mechanisms

### 4. Training Compatibility
- Maintains existing tree GRPO training semantics
- Each combination node has clear generation boundaries
- Supports hierarchical reward propagation

## Running Examples

```bash
# Basic example
python examples/smolagents_tree_grpo_example.py

# With custom configuration
python examples/smolagents_tree_grpo_example.py --config examples/smolagents_tree_grpo_config.yaml
```

## Requirements

- `smolagents` package
- Access to LLM API (OpenAI, Anthropic, etc.)
- Required tools for managed agents (search APIs, code execution, etc.)

## Limitations

- Requires smolagents dependency
- Increases computational cost due to multiple agent calls
- Memory usage scales with number of agents and proposals
- Coordination quality depends on coordinator agent capability

## Future Enhancements

- Dynamic proposal count based on task complexity
- Learned fusion strategies beyond template-based approaches
- Agent specialization optimization based on performance
- Integration with more sophisticated coordination strategies