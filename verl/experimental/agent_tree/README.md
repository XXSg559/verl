# Agent Tree Rollout

A flexible and extensible framework for multi-agent tree-based rollout and search in reinforcement learning environments.

## Overview

Agent Tree Rollout extends verl's existing agent loop infrastructure with powerful multi-agent tree search capabilities. It allows multiple specialized agents to collaborate in exploring solution spaces through tree search algorithms, enabling more sophisticated problem-solving and exploration strategies.

## Key Features

- **Multi-Agent Collaboration**: Multiple specialized agents work together to explore different aspects of the solution space
- **Tree-Based Search**: Systematic exploration using tree search algorithms with intelligent pruning
- **Flexible Agent Roles**: Configurable agents with different roles (Explorer, Evaluator, Coordinator, Synthesizer)
- **Specialization Support**: Agents can specialize in different domains (mathematical, creative, logical, etc.)
- **Multiple Coordination Strategies**: Support for hierarchical, collaborative, and competitive coordination
- **Configuration-Driven**: Easy setup through YAML configuration files
- **vLLM Integration**: Leverages vLLM's multi-candidate generation and beam search capabilities
- **Backwards Compatibility**: Seamlessly integrates with existing verl workflows

## Architecture

### Core Components

1. **BaseTreeAgent**: Abstract interface for all tree search agents
2. **AgentPool**: Manages collections of agents with load balancing and selection
3. **TreeSearchEngine**: Core search engine coordinating multi-agent exploration
4. **TreeNode**: Represents nodes in the search tree with conversation state
5. **AgentFactory**: Creates agents from configuration templates
6. **AgentTreeLoop**: Integrates with verl's agent loop system

### Agent Roles

- **Explorer**: Generates new branches and solutions
- **Evaluator**: Assesses quality and promise of solutions
- **Coordinator**: Plans and manages the search process
- **Synthesizer**: Combines and ranks results from multiple agents
- **Validator**: Validates solutions for correctness
- **Critic**: Provides critical analysis and feedback

## Quick Start

### 1. Basic Configuration

Create a configuration file (e.g., `my_agent_tree_config.yaml`):

```yaml
defaults:
  - ppo_trainer
  - _self_

actor_rollout_ref:
  rollout:
    name: agent_tree

    tree_config:
      enable: true
      max_depth: 4
      max_branches_per_node: 3
      coordination_strategy: hierarchical

    agent_tree_config:
      agent_pool:
        explorers:
          - name: "math_explorer"
            type: "specialized"
            role: "EXPLORER"
            specialization: "mathematical"
            temperature: 0.3

          - name: "creative_explorer"
            type: "specialized"
            role: "EXPLORER"
            specialization: "creative"
            temperature: 0.9

        evaluators:
          - name: "quality_judge"
            type: "specialized"
            role: "EVALUATOR"
            specialization: "quality_assessment"
            temperature: 0.2
```

### 2. Run Training

```bash
# Use your configuration file with verl training
python -m verl.trainer.main_ppo config=my_agent_tree_config.yaml
```

### 3. Programmatic Usage

```python
from verl.experimental.agent_tree import (
    AgentFactory, TreeSearchEngine, AgentTreeLoop
)

# Create agent pool from configuration
agent_pool = AgentFactory.create_agent_pool(
    config["agent_tree_config"],
    server_manager,
    tokenizer
)

# Create tree search engine
tree_engine = TreeSearchEngine(agent_pool, search_config)

# Run search
result = await tree_engine.search(
    initial_node=initial_node,
    sampling_params=sampling_params,
    problem_type="mathematical"
)
```

## Configuration Templates

Pre-built configuration templates are available for common use cases:

- **Math Problems**: `get_config_template("math")`
- **Creative Tasks**: `get_config_template("creative")`
- **Coding Problems**: `get_config_template("coding")`
- **Research Analysis**: `get_config_template("research")`
- **Minimal Setup**: `get_config_template("minimal")`

```python
from verl.experimental.agent_tree import get_config_template

# Get a predefined configuration
config = get_config_template("math")
```

## Advanced Features

### Custom Agent Specializations

Create custom specialized agents:

```python
class CustomDomainAgent(SpecializedTreeAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(
            specialization="custom_domain",
            *args, **kwargs
        )

    async def generate_branches(self, node, context, num_branches=3):
        # Custom branch generation logic
        pass
```

### Coordination Strategies

- **Hierarchical**: Coordinator plans → Explorers execute → Evaluators assess → Synthesizer combines
- **Collaborative**: All agents work together on each node
- **Competitive**: Agents compete to generate the best solutions

### Dynamic Agent Selection

The system automatically selects appropriate agents based on:
- Problem type classification
- Agent specialization matching
- Performance history
- Current workload
- Diversity requirements

## Configuration Reference

### Tree Search Configuration

```yaml
tree_config:
  enable: true                    # Enable tree search
  max_depth: 4                    # Maximum tree depth
  max_branches_per_node: 3        # Branches per node
  max_active_nodes: 8             # Max simultaneous nodes
  search_timeout: 180.0           # Timeout in seconds
  coordination_strategy: hierarchical  # Strategy type
  fallback_to_single_agent: true  # Fallback behavior
```

### Agent Configuration

```yaml
agent_pool:
  explorers:
    - name: "agent_name"
      type: "specialized"           # Agent type
      role: "EXPLORER"             # Agent role
      specialization: "mathematical"  # Domain specialization
      model: "model/path"          # Model to use
      temperature: 0.7             # Generation temperature
      max_tokens: 300              # Max tokens per generation
      tools: ["tool1", "tool2"]    # Available tools
      weight: 1.0                  # Selection weight
```

## Performance Tuning

### Search Parameters

- **max_depth**: Deeper search for complex problems, shallower for simple ones
- **max_branches_per_node**: More branches increase diversity but cost computation
- **max_active_nodes**: Controls memory usage and computational load
- **search_timeout**: Prevents runaway searches

### Agent Selection

- **weight**: Higher weights make agents more likely to be selected
- **specialization**: Match agent specializations to problem types
- **temperature**: Lower for accuracy, higher for creativity

### Coordination Strategy

- **hierarchical**: Best for structured problems requiring systematic exploration
- **collaborative**: Good for problems benefiting from diverse perspectives
- **competitive**: Useful when you want only the best solutions

## Examples

See `examples/agent_tree_example.py` for a complete demonstration including:
- Basic tree search setup
- Configuration-based agent creation
- Different specialization examples
- Performance analysis

## Integration with verl

Agent Tree Rollout seamlessly integrates with verl's existing components:

- **ToolAgentLoop**: Extends existing tool-calling capabilities
- **AsyncLLMServerManager**: Uses existing server management
- **vLLM/SGLang**: Leverages existing inference backends
- **Ray**: Uses existing distributed infrastructure
- **Hydra**: Compatible with existing configuration system

## Best Practices

1. **Agent Diversity**: Use agents with complementary specializations
2. **Problem Classification**: Accurately classify problems for better agent selection
3. **Resource Management**: Monitor and tune search parameters for your hardware
4. **Fallback Strategy**: Always enable fallback to single-agent mode for reliability
5. **Performance Monitoring**: Track agent performance and adjust weights accordingly

## Troubleshooting

### Common Issues

1. **Tree search too slow**: Reduce max_depth, max_branches_per_node, or search_timeout
2. **Poor quality results**: Add more evaluator agents or adjust specializations
3. **Memory issues**: Reduce max_active_nodes or enable more aggressive pruning
4. **Agent conflicts**: Use hierarchical coordination instead of competitive

### Debug Information

Enable debug logging to see detailed search progress:

```python
import logging
logging.getLogger('verl.experimental.agent_tree').setLevel(logging.DEBUG)
```

## Contributing

To add new agent types or coordination strategies:

1. Implement the `BaseTreeAgent` interface
2. Register with `AgentFactory.register_agent_type()`
3. Add configuration templates in `config_templates.py`
4. Update documentation and examples

## License

This code is licensed under the Apache License 2.0. See the LICENSE file for details.