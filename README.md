# Open Game Engine to PyTorch Compiler

## Overview

This project provides an automatic compiler that transforms the Open Game Engine's domain-specific language (DSL) into a fully functional PyTorch implementation. The compiler analyzes the source code of the Open Game Engine and generates a PyTorch-based library that preserves the original game-theoretic semantics.

## Features

- **Comprehensive Game Theory Support**
  - Strategic-form (Normal-form) games
  - Sequential games
  - Extensive-form games 
  - Bayesian games with incomplete information
  - Simultaneous play games

- **Deep DSL Analysis**
  - Automatically detects game structures and patterns
  - Converts complex type systems to PyTorch tensors
  - Preserves original semantic intent

- **Extensive Testing**
  - Automatically generated unit tests
  - Covers core game theory concepts
  - Validates translation correctness

- **Advanced Game Algorithms**
  - Nash equilibrium computation
  - Best response calculations
  - Fictitious play
  - Bayesian Nash equilibrium

## Comprehensive Unit Testing Framework

The unit testing approach provides rigorous validation of the PyTorch-based Open Game Engine implementation, ensuring correctness, reliability, and adherence to game-theoretic principles.

### Test Suites

1. **Core Functionality Tests** (`test_core.py`)
   - Player class validation
   - Lens (state manipulation) tests
   - OpenGame infrastructure tests

2. **Strategic Game Tests** (`test_strategic_games.py`)
   - Prisoner's Dilemma game analysis
   - Matching Pennies game verification
   - Nash equilibrium computation

3. **Simultaneous Play Tests** (`test_simultaneous_games.py`)
   - Chicken Game (Hawk-Dove) scenario
   - Simultaneous action payoff calculations
   - Best response computations
   - Nash equilibrium for games with mixed strategies

4. **Sequential Game Tests** (`test_sequential_games.py`)
   - Sequential gameplay mechanics
   - Extensive form game structure
   - Player interaction sequences

5. **Bayesian Game Tests** (`test_bayesian_games.py`)
   - Incomplete information game testing
   - Type distribution validation
   - Bayesian Nash equilibrium computation

### Key Test Scenarios

- Player creation and strategy management
- State observation and modification
- Game play and payoff calculations
- Best response computations
- Equilibrium strategy detection
- Probabilistic strategy generation
- Simultaneous and sequential interaction models

### Running Tests

```bash
# Run entire test suite
python -m unittest discover tests

# Run specific test modules
python -m unittest tests.test_strategic_games
python -m unittest tests.test_simultaneous_games
```

## Prerequisites

- Python 3.7+
- PyTorch 1.9.0+
- Git (for cloning repositories)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/opengames2pytorch.git
cd opengames2pytorch

# Install the package
pip install -e .
```

## Usage

### Compiling an Open Game Engine Repository

```bash
python compiler.py --url https://github.com/example/open-game-engine.git --output ./pytorch-oge
```

### Example Game Implementation

```python
import torch
from pytorch_oge.game import StrategicGame, create_strategic_game

# Create a Prisoner's Dilemma game
player1_payoffs = [
    [-1, -5],  # Player 1 cooperates: [Player 2 cooperates, Player 2 defects]
    [0, -3]    # Player 1 defects: [Player 2 cooperates, Player 2 defects]
]

player2_payoffs = [
    [-1, 0],   
    [-5, -3]   
]

payoff_tensors = [
    torch.tensor(player1_payoffs, dtype=torch.float32),
    torch.tensor(player2_payoffs, dtype=torch.float32)
]

# Instantiate the game
game = StrategicGame(2, [2, 2], payoff_tensors)

# Calculate payoffs for a specific action profile
actions = [torch.tensor(0), torch.tensor(0)]  # Both cooperate
payoffs = game(actions)
print(f"Payoffs when both cooperate: {payoffs}")

# Find best response
p1_best_response = game.best_response(0, [None, torch.tensor(0)])
print(f"Player 1's best response when Player 2 cooperates: {p1_best_response}")

# Calculate Nash equilibrium
equilibrium = game.nash_equilibrium(iterations=1000)
print(f"Nash equilibrium strategies: {equilibrium}")
```

## Limitations

- Currently supports a subset of Open Game Engine's DSL
- Requires manual verification for complex game structures
- Performance may vary compared to native implementations

## Acknowledgments

- [Open Game Engine](https://github.com/CyberCat-Institute/open-game-engine/) Project
- PyTorch Team
- Game Theory Research Community
- Category Theory Research Community
- Functional Programming Research Community

## Contact

Eric Schmid - schmideric@pm.me
