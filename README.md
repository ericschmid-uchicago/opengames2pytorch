# Open Game Engine to PyTorch Compiler

## 🎮 Project Overview

### Background
The Open Game Engine to PyTorch Compiler is an innovative automated translation tool that bridges the gap between domain-specific game-theoretic modeling languages and modern machine learning frameworks. By systematically converting complex game-theoretic models into executable PyTorch implementations, this compiler enables researchers and developers to leverage advanced computational techniques in game theory and strategic modeling.

### Key Objectives
- 🔄 Automatic DSL to PyTorch Translation
- 🧩 Preservation of Game-Theoretic Semantics
- 🚀 Enhanced Computational Efficiency
- 🔬 Support for Advanced Game Theory Concepts

## 🛠 Technical Architecture

### Compilation Process
The compiler follows a multi-stage transformation pipeline:

1. **DSL Parsing**
   - Analyze source domain-specific language files
   - Extract type definitions, game structures, and semantic information
   - Build an intermediate representation of game-theoretic constructs

2. **Semantic Mapping**
   - Map DSL concepts to PyTorch-compatible abstractions
   - Transform category-theoretical game representations
   - Generate type-safe tensor-based game implementations

3. **Code Generation**
   - Produce modular PyTorch modules
   - Create comprehensive test suites
   - Generate package management files

### Core Modules Generated

| Module | Functionality | Key Features |
|--------|--------------|--------------|
| `core.py` | Fundamental Abstractions | - Game Composition Operators<br>- Tensor Transformations<br>- Category-Theoretical Foundations |
| `game.py` | Strategic Game Implementations | - Nash Equilibrium Computation<br>- Mixed Strategy Modeling<br>- Multi-Player Game Support |
| `sequential.py` | Sequential Game Models | - Turn-Based Game Mechanics<br>- State Transition Modeling<br>- Backward Induction<br>- Subgame Perfect Equilibrium |
| `bayesian.py` | Incomplete Information Games | - Type-Conditional Payoff Modeling<br>- Probabilistic Strategy Computation<br>- Bayesian Nash Equilibrium |
| `category.py` | Categorical Type Abstractions | - Functor Implementations<br>- Monad Transformations<br>- Higher-Order Game Composition |
| `utils.py` | Utility Functions | - Probability Simplex Projection<br>- Expected Utility Calculation<br>- Best Response Computation |

> **Note:** While the repository includes extensive support for sequential games with backward induction and subgame perfect equilibrium, it currently does not include explicit support for extensive-form game representations.

## 📋 Command-Line Interface

### Detailed Argument Specification

| Argument | Type | Description | Constraints | Example |
|----------|------|-------------|-------------|---------|
| `--repo` | Optional[str] | Local path to Open Game Engine repository | Must be valid directory | `/home/user/open-game-engine` |
| `--url` | Optional[str] | Remote repository URL to clone | Must be valid Git repository URL | `https://github.com/organization/open-game-engine` |
| `--output` | Required[str] | Destination directory for generated PyTorch implementation | Must be writable path | `/path/to/generated/pytorch-oge` |

### Usage Patterns

```bash
# Compile from local repository
python compiler.py --repo /path/to/local/open-game-engine --output ./pytorch_oge

# Compile by cloning remote repository
python compiler.py --url https://github.com/CyberCat-Institute/open-game-engine.git --output ./pytorch_oge

# Advanced: Specify custom compilation options
python compiler.py --repo /path/to/open-game-engine --output ./pytorch_oge --verbose
```

## 🧪 Comprehensive Test Suite

### Test Categories

1. **Core Functionality Tests**
   - Player Strategy Validation
   - Lens Transformation Mechanics
   - Game Composition Verification
   - Tensor Operation Correctness

2. **Strategic Game Tests**
   - Nash Equilibrium Computation
   - Best Response Strategies
   - Payoff Matrix Validation
   - Multi-Player Game Scenarios

3. **Sequential Game Tests**
   - Turn-Based Game Mechanics
   - State Transition Validation
   - Player Interaction Modeling
   - Backward Induction Analysis
   - Market Entry Deterrence Games
   - Market for Lemons with Information Asymmetry
   - Signaling Games and Belief Updating

4. **Bayesian Game Tests**
   - Type-Conditional Payoff Verification
   - Probabilistic Strategy Computation
   - Incomplete Information Game Modeling

5. **Composition Tests**
   - Ten 3-Player Game Composition
   - Associativity Testing
   - Performance Evaluation

## 🧮 Mathematical Foundations

### Theoretical Underpinnings

The compiler implements advanced mathematical concepts:

- **Open Game Composition** (∘): Allows complex game structures to be built from simpler components
- **Nash Equilibrium Computation**: Identifies stable strategy profiles
- **Backward Induction**: Solves sequential games using dynamic programming
- **Subgame Perfect Equilibrium**: Refines Nash equilibrium for sequential games
- **Categorical Game Theory**: Provides higher-order game transformations
- **Tensor-Based Strategic Modeling**: Enables efficient computational representations

## 💻 Example Implementations

### Prisoners' Dilemma (Strategic Game)

```python
import torch
from pytorch_oge.game import StrategicGame

# Define payoff matrices
player1_payoffs = torch.tensor([
    [-1, -5],  # Cooperation vs. Opponent's Actions
    [0, -3]    # Defection vs. Opponent's Actions
], dtype=torch.float32)

player2_payoffs = torch.tensor([
    [-1, 0],   # Cooperation vs. Opponent's Actions
    [-5, -3]   # Defection vs. Opponent's Actions
], dtype=torch.float32)

# Create Strategic Game
game = StrategicGame(
    num_players=2,
    action_spaces=[2, 2],
    payoff_tensors=[player1_payoffs, player2_payoffs]
)

# Compute Nash Equilibrium
equilibrium_strategies = game.nash_equilibrium(iterations=1000)
print("Equilibrium Strategies:", equilibrium_strategies)
```

### Entry Deterrence Game (Sequential Game)

```python
from pytorch_oge.core import Player
from pytorch_oge.sequential import SequentialGame
import torch

# Define player strategies
def entrant_strategy(observation):
    # Enter (1) if market size >= 8, otherwise Stay Out (0)
    market_size = observation["market_size"]
    return 1 if market_size >= 8 else 0

def incumbent_strategy(observation):
    # If entrant enters, fight (1) in small markets, accommodate (0) in large markets
    if "entrant_action" not in observation:
        return 0
    if observation["entrant_action"] == 0:  # Entrant stayed out
        return 0
    else:  # Entrant entered
        market_size = observation["market_size"]
        return 1 if market_size < 8 else 0

# Create players
entrant = Player("Entrant", entrant_strategy)
incumbent = Player("Incumbent", incumbent_strategy)

# Define state transition function
def state_transition(state, action, player_idx):
    if player_idx == 0:  # Entrant's turn
        return {"market_size": state["market_size"], "entrant_action": action}
    else:  # Incumbent's turn
        return {
            "market_size": state["market_size"],
            "entrant_action": state["entrant_action"],
            "incumbent_action": action
        }

# Define payoff functions
def entrant_payoff(final_state):
    if final_state["entrant_action"] == 0:  # Stayed out
        return torch.tensor(2.0)  # Safe outside option
    else:  # Entered
        if final_state["incumbent_action"] == 1:  # Incumbent fought
            return torch.tensor(-1.0)
        else:  # Incumbent accommodated
            return torch.tensor(final_state["market_size"] / 2)

def incumbent_payoff(final_state):
    if final_state["entrant_action"] == 0:  # Entrant stayed out
        return torch.tensor(final_state["market_size"])  # Monopoly profits
    else:  # Entrant entered
        if final_state["incumbent_action"] == 1:  # Fought
            return torch.tensor(3.0)
        else:  # Accommodated
            return torch.tensor(final_state["market_size"] / 3)

# Create the game
game = SequentialGame(
    players=[entrant, incumbent],
    initial_state={"market_size": 6},
    state_transition=state_transition,
    payoff_functions=[entrant_payoff, incumbent_payoff]
)

# Play the game
actions, final_state, payoffs = game()
print(f"Actions: {actions}")
print(f"Final State: {final_state}")
print(f"Payoffs: {[p.item() for p in payoffs]}")

# Find subgame perfect equilibrium
equilibrium = game.backward_induction()
print("Subgame Perfect Equilibrium:", equilibrium)
```

## 🚧 Limitations and Considerations

### Potential Constraints

- Requires well-structured input Domain-Specific Language
- Performance varies with game complexity
- Limited to PyTorch-compatible computational models
- Assumes canonical game-theoretic representational patterns
- Does not currently support extensive-form game representations

### Compatibility Requirements

- Python 3.7+
- PyTorch 1.9.0+
- Robust type annotations
- Minimal external dependencies

## 🔬 Advanced Features

### Extensibility Points

- Custom game transformation hooks
- User-definable compilation strategies
- Pluggable game composition mechanisms
- Advanced tensor operation support
- Sequential game modeling with backward induction

## 📦 Installation & Setup

### Prerequisites

```bash
# Recommended Python Environment
python3 -m venv oge_env
source oge_env/bin/activate

# Install Core Dependencies
pip install torch numpy argparse typing

# Clone Compiler Repository
git clone https://github.com/ericschmid-uchicago/opengames2pytorch.git
cd opengames2pytorch

# Install Compiler
pip install -e .
```

### Compilation Workflow

```bash
# Basic Compilation from Remote Repository
python compiler.py --url https://github.com/CyberCat-Institute/open-game-engine.git --output ./generated_oge

# Compilation from Local Repository
python compiler.py --repo ./open-game-engine --output ./generated_oge

# Verbose Compilation with Logging
python compiler.py --repo ./open-game-engine --output ./generated_oge --verbose

# Run Generated Test Suite
cd generated_oge
python -m unittest discover tests
```

## 🤝 Contributing

### Contribution Guidelines

1. Fork the Repository
2. Create Feature Branch
3. Implement Changes
4. Write Comprehensive Tests
5. Submit Pull Request

### Reporting Issues

- Use GitHub Issues
- Provide Minimal Reproducible Example
- Describe Expected vs. Actual Behavior

## 📞 Contact & Support

- **Project Maintainer**: Eric Schmid
- **Email**: schmideric@pm.me
- **Compiler Repository**: [opengames2pytorch on GitHub](https://github.com/ericschmid-uchicago/opengames2pytorch)
- **Open Game Engine Repository**: [CyberCat-Institute/open-game-engine](https://github.com/CyberCat-Institute/open-game-engine)
- **Issue Tracker**: 
  - Compiler Issues: [opengames2pytorch Issues](https://github.com/ericschmid-uchicago/opengames2pytorch/issues)

## 🌟 Acknowledgments

- CyberCat Institute
- Open Game Engine Community
- PyTorch Development Team
- Game Theory Researchers

---

**Disclaimer**: This is an automated compilation tool developed by Eric Schmid for translating the Open Game Engine to PyTorch. Always verify generated implementations for your specific use case.
