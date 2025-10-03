# Agent Tree Examples

This directory contains examples demonstrating multi-agent tree-based GRPO training using smolagents integration.

## Examples

### 1. `smolagents_tree_grpo_example.py`
**General-purpose multi-agent collaboration with tree GRPO**

Demonstrates the core smolagents + tree GRPO integration with three general agents:
- **Research Agent**: Web search and information gathering
- **Coding Agent**: Code generation and execution
- **Reasoning Agent**: Logical analysis and strategic thinking

Features:
- Tree-based rollout with cartesian product combinations
- Multi-agent coordination and proposal generation
- Execution-based reward calculation
- Credit assignment and reward decomposition

### 2. `math_solving_example.py`
**Specialized mathematical problem solving**

Shows how to adapt the multi-agent system for mathematical problem solving with three specialized agents:
- **Math Analyzer Agent**: Problem analysis and strategy identification
- **Math Solver Agent**: Solution generation and calculation
- **Math Verifier Agent**: Answer verification and validation

Features:
- Domain-specific agent specialization
- Mathematical reasoning workflow (analyze → solve → verify)
- Quadratic equation solving demonstration
- Solution quality evaluation

### 3. `smolagents_tree_grpo_config.yaml`
**Configuration file for smolagents tree GRPO**

Comprehensive configuration showing all available options:
- Agent coordination strategies
- Tree search parameters
- Reward calculation settings
- Performance optimization options

## Usage

Run the examples directly:

```bash
# General multi-agent collaboration
python examples/agent_tree/smolagents_tree_grpo_example.py

# Math problem solving
python examples/agent_tree/math_solving_example.py

# With custom configuration
python examples/agent_tree/smolagents_tree_grpo_example.py --config examples/agent_tree/smolagents_tree_grpo_config.yaml
```

## Requirements

- `smolagents` package
- Access to LLM API (OpenAI, Anthropic, etc.)
- Python environment with verl installed

## Key Concepts

### Tree GRPO Training
- **Tree-based rollout**: Multiple decision paths explored simultaneously
- **Node-level training**: Each tree node represents a training sample
- **Hierarchical rewards**: Reward propagation from leaves to root

### Multi-Agent Coordination
- **Coordinator agent**: Plans and manages agent deployment
- **Managed agents**: Specialized agents for specific tasks
- **Cartesian product**: All combinations of agent outputs explored

### Credit Assignment
- **Agent contributions**: Track individual agent's role in each combination
- **Reward decomposition**: Distribute node rewards to contributing agents
- **Collaborative learning**: Learn optimal agent coordination patterns

## Architecture

```
Coordinator (Planner)
├── Agent 1 (Specialist A)
├── Agent 2 (Specialist B)
└── Agent 3 (Specialist C)

Tree Expansion:
Root → [Combo1, Combo2, Combo3, ...] → [Further combinations...]

Each Combo = (Agent1_output_i, Agent2_output_j, Agent3_output_k)
```

This architecture enables rich exploration of collaborative strategies while maintaining clear training boundaries for GRPO optimization.