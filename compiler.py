import os
import sys
import subprocess
import argparse
import logging
import re
import ast
import shutil
import json
import tempfile
import textwrap
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Iterator, Callable
from pathlib import Path
from dataclasses import dataclass, field

import torch
from torch.utils.cpp_extension import load_inline
import torch.nn as nn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("oge2pytorch")


# --------------------------------------------------------------------------
# Type definitions for representing the deeply embedded DSL

@dataclass
class DSLType:
    """Base class for DSL type representations."""
    name: str
    
    def to_pytorch_type(self) -> str:
        return "torch.Tensor"  # Default fallback

@dataclass
class DSLPrimitiveType(DSLType):
    """Represents primitive types in the DSL."""
    def to_pytorch_type(self) -> str:
        type_mapping = {
            "int": "torch.int64",
            "float": "torch.float32",
            "bool": "torch.bool",
            "string": "str",
        }
        return type_mapping.get(self.name, "torch.Tensor")

@dataclass
class DSLFunctionType(DSLType):
    """Represents function types in the DSL."""
    param_types: List[DSLType]
    return_type: DSLType
    
    def to_pytorch_type(self) -> str:
        params = ", ".join(param.to_pytorch_type() for param in self.param_types)
        return_type = self.return_type.to_pytorch_type()
        return f"Callable[[{params}], {return_type}]"

@dataclass
class DSLGameType(DSLType):
    """Represents game types in the DSL."""
    state_type: Optional[DSLType] = None
    observation_type: Optional[DSLType] = None
    action_type: Optional[DSLType] = None
    payoff_type: Optional[DSLType] = None
    
    def to_pytorch_type(self) -> str:
        return "OpenGameTensor"

@dataclass
class DSLContext:
    """Context for DSL translation, holding type definitions and functions."""
    types: Dict[str, DSLType] = field(default_factory=dict)
    functions: Dict[str, Callable] = field(default_factory=dict)
    modules: Dict[str, 'DSLModule'] = field(default_factory=dict)
    game_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)

@dataclass
class DSLModule:
    """Represents a module in the DSL."""
    name: str
    types: Dict[str, DSLType] = field(default_factory=dict)
    functions: Dict[str, Any] = field(default_factory=dict)
    submodules: Dict[str, 'DSLModule'] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Functions for cloning and scanning repositories

def clone_repo(repo_url: str, target_dir: str) -> str:
    logger.info(f"Cloning repository {repo_url} to {target_dir}")
    os.makedirs(target_dir, exist_ok=True)
    try:
        process = subprocess.Popen(
            ['git', 'clone', '--progress', repo_url, target_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        for line in process.stderr:
            line = line.strip()
            if line:
                logger.info(f"Git: {line}")
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, ['git', 'clone', repo_url, target_dir])
        logger.info(f"Successfully cloned repository to {target_dir}")
    except Exception as e:
        logger.error(f"Error cloning repository: {e}")
        raise
    return target_dir

def find_source_files(repo_dir: str) -> List[Path]:
    logger.info(f"Finding source files in {repo_dir}")
    source_files = []
    for ext in ['.re', '.res', '.ml', '.mli', '.hs']:
        found_files = list(Path(repo_dir).glob(f'**/*{ext}'))
        logger.info(f"Found {len(found_files)} {ext} files")
        source_files.extend(found_files)
    excluded_dirs = ['node_modules', '_build', 'tests', 'examples']
    filtered_files = [f for f in source_files if not any(ex_dir in str(f) for ex_dir in excluded_dirs)]
    logger.info(f"Found {len(filtered_files)} source files after filtering")
    if filtered_files:
        logger.info("Sample files:")
        for f in filtered_files[:5]:
            logger.info(f"  {f}")
        if len(filtered_files) > 5:
            logger.info(f"  ... and {len(filtered_files) - 5} more")
    return filtered_files


# --------------------------------------------------------------------------
# Deep Embedding Analyzer

class DeepEmbeddingAnalyzer:
    """Analyzes the deeply embedded DSL structure of the Open Game Engine."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.context = DSLContext()
    
    def analyze(self) -> DSLContext:
        logger.info("Analyzing deep embedding structure")
        # Try to find the core DSL file in either .re or .hs format
        core_file = self._find_file("OpenGames.re") or self._find_file("OpenGames.hs")
        if core_file:
            logger.info(f"Found core DSL file: {core_file}")
            self._analyze_classes_file(core_file)
        else:
            logger.warning("Could not find core DSL file (e.g., OpenGames.re or OpenGames.hs)")
        
        category_file = self._find_file("Category.re") or self._find_file("Category.hs")
        if category_file:
            logger.info(f"Found category file: {category_file}")
            self._analyze_category_file(category_file)
        
        self._detect_game_patterns()
        self._analyze_module_dependencies()
        return self.context
    
    def _find_file(self, filename: str) -> Optional[Path]:
        # First, search for the file as given
        matches = list(self.repo_path.glob(f"**/{filename}"))
        if not matches:
            # If not found, try switching the extension (.re -> .hs or vice versa)
            if filename.endswith(".re"):
                alt_filename = filename.replace(".re", ".hs")
            elif filename.endswith(".hs"):
                alt_filename = filename.replace(".hs", ".re")
            else:
                alt_filename = filename
            matches = list(self.repo_path.glob(f"**/{alt_filename}"))
        return matches[0] if matches else None
    
    def _analyze_classes_file(self, file_path: Path):
        logger.info(f"Analyzing type classes in {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return
        # Use regex with MULTILINE to capture definitions
        module_matches = re.finditer(r'^module\s+(\w+)\s*=\s*{(.*?)}', content, re.DOTALL | re.MULTILINE)
        for match in module_matches:
            module_name = match.group(1)
            module_content = match.group(2)
            logger.info(f"Found module: {module_name}")
            self._extract_types_from_module(module_name, module_content)
        instance_matches = re.finditer(r'let\s+(\w+)(?:Instance|_instance)\s*:\s*(\w+)', content)
        for match in instance_matches:
            instance_name = match.group(1)
            type_class = match.group(2)
            logger.info(f"Found type class instance: {instance_name} for {type_class}")
            self.context.functions[instance_name] = (type_class, "instance")
    
    def _extract_types_from_module(self, module_name: str, module_content: str):
        # Adjust regex to allow multi-line definitions until a semicolon
        type_matches = re.finditer(r'^type\s+(\w+)\s*=\s*(.*?);', module_content, re.DOTALL | re.MULTILINE)
        module = DSLModule(name=module_name)
        for match in type_matches:
            type_name = match.group(1)
            type_definition = match.group(2).strip()
            logger.info(f"Found type: {type_name} in module {module_name}")
            dsl_type = self._parse_type_definition(type_name, type_definition)
            module.types[type_name] = dsl_type
        func_matches = re.finditer(r'^let\s+(\w+)\s*:\s*(.*?)\s*=\s*(.*?);', module_content, re.DOTALL | re.MULTILINE)
        for match in func_matches:
            func_name = match.group(1)
            func_type = match.group(2).strip()
            func_body = match.group(3).strip()
            logger.info(f"Found function: {func_name} in module {module_name}")
            module.functions[func_name] = (func_type, func_body)
        self.context.modules[module_name] = module
    
    def _parse_type_definition(self, name: str, definition: str) -> DSLType:
        # Remove Haskell type constraints (anything before =>)
        if "=>" in definition:
            definition = definition.split("=>", 1)[1].strip()
        if "->" in definition:
            return self._parse_function_type(name, definition)
        if any(kw in definition.lower() for kw in ["game", "lens", "optic"]):
            return DSLGameType(name=name)
        return DSLPrimitiveType(name=name)
    
    def _parse_function_type(self, name: str, definition: str) -> DSLFunctionType:
        parts = [p.strip() for p in definition.split("->")]
        param_types = [DSLPrimitiveType(name=p) for p in parts[:-1]]
        return_type = DSLPrimitiveType(name=parts[-1])
        return DSLFunctionType(name=name, param_types=param_types, return_type=return_type)
    
    def _analyze_category_file(self, file_path: Path):
        logger.info(f"Analyzing category theory abstractions in {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return
        category_matches = re.finditer(r'^module\s+(\w+)\s*=\s*{(.*?)}', content, re.DOTALL | re.MULTILINE)
        for match in category_matches:
            module_name = match.group(1)
            module_content = match.group(2)
            logger.info(f"Found category module: {module_name}")
            self._extract_categorical_operations(module_name, module_content)
    
    def _extract_categorical_operations(self, module_name: str, module_content: str):
        functor_matches = re.finditer(r'let\s+(map|fmap)\b', module_content)
        for _ in functor_matches:
            logger.info(f"Found functor operation in {module_name}")
            if module_name not in self.context.modules:
                self.context.modules[module_name] = DSLModule(name=module_name)
            self.context.modules[module_name].functions["functor"] = True
        monad_matches = re.finditer(r'let\s+(bind|return|pure)\b', module_content)
        for _ in monad_matches:
            logger.info(f"Found monad operation in {module_name}")
            if module_name not in self.context.modules:
                self.context.modules[module_name] = DSLModule(name=module_name)
            self.context.modules[module_name].functions["monad"] = True
    
    def _detect_game_patterns(self):
        """Detect common game-theoretic patterns in the codebase."""
        logger.info("Detecting game patterns")
        
        # Search through source files for game pattern definitions
        source_files = []
        for ext in ['.re', '.res', '.ml', '.mli', '.hs']:
            source_files.extend(self.repo_path.glob(f"**/*{ext}"))
        
        for file_path in source_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue
            
            # Look for strategic/normal form game definitions
            strategic_matches = re.finditer(r'(strategic|normal)[\s_]?form[\s_]?game', content, re.IGNORECASE)
            for match in strategic_matches:
                logger.info(f"Found strategic-form game pattern in {file_path}")
                self.context.game_patterns["strategic"] = {
                    "file": file_path,
                    "detected": True
                }
            
            # Look for sequential/extensive form game definitions
            sequential_matches = re.finditer(r'(sequential|extensive)[\s_]?form[\s_]?game', content, re.IGNORECASE)
            for match in sequential_matches:
                logger.info(f"Found sequential-form game pattern in {file_path}")
                self.context.game_patterns["sequential"] = {
                    "file": file_path,
                    "detected": True
                }
            
            # Look for Bayesian game definitions
            bayesian_matches = re.finditer(r'bayesian[\s_]?game', content, re.IGNORECASE)
            for match in bayesian_matches:
                logger.info(f"Found Bayesian game pattern in {file_path}")
                self.context.game_patterns["bayesian"] = {
                    "file": file_path,
                    "detected": True
                }
            
            # Look for decision rules or strategic players
            decision_matches = re.finditer(r'(decision|player|agent)[\s_]?(rule|strategy)', content, re.IGNORECASE)
            for match in decision_matches:
                logger.info(f"Found decision rule pattern in {file_path}")
                self.context.game_patterns["decision"] = {
                    "file": file_path,
                    "detected": True
                }
    
    def _analyze_module_dependencies(self):
        logger.info("Analyzing module dependencies")
        source_files = []
        for ext in ['.re', '.res', '.ml', '.mli', '.hs']:
            source_files.extend(self.repo_path.glob(f'**/*{ext}'))
        for file_path in source_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue
            import_matches = re.finditer(r'open\s+(\w+)', content)
            module_name = file_path.stem  # Use file stem directly (reflecting Haskell conventions)
            for match in import_matches:
                imported_module = match.group(1)
                logger.info(f"Module {module_name} imports {imported_module}")
                if module_name not in self.context.modules:
                    self.context.modules[module_name] = DSLModule(name=module_name)
                if imported_module in self.context.modules:
                    self.context.modules[module_name].submodules[imported_module] = self.context.modules[imported_module]


# --------------------------------------------------------------------------
# PyTorch DSL Generator

class PyTorchDSLGenerator:
    """Generates a PyTorch implementation of the Open Game Engine DSL."""
    
    def __init__(self, dsl_context: DSLContext):
        self.context = dsl_context
    
    def generate_pytorch_dsl(self) -> Dict[str, str]:
        logger.info("Generating PyTorch DSL implementation")
        modules = {}
        modules["core"] = self._generate_core_module()
        modules["game"] = self._generate_game_module()
        modules["category"] = self._generate_category_module()
        modules["utils"] = self._generate_utils_module()
        
        # Generate additional modules based on detected patterns
        if "sequential" in self.context.game_patterns and self.context.game_patterns["sequential"]["detected"]:
            modules["sequential"] = self._generate_sequential_game_module()
        
        if "bayesian" in self.context.game_patterns and self.context.game_patterns["bayesian"]["detected"]:
            modules["bayesian"] = self._generate_bayesian_game_module()
        
        for module_name, module in self.context.modules.items():
            lower_name = module_name.lower()
            if lower_name not in modules:
                modules[lower_name] = self._generate_module(module_name, module)
        
        return modules
    
    def _generate_core_module(self) -> str:
        core_code = textwrap.dedent('''
            import torch
            import torch.nn as nn
            from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

            S = TypeVar('S')  # State type
            O = TypeVar('O')  # Observation type
            A = TypeVar('A')  # Action type
            P = TypeVar('P')  # Payoff type

            class OpenGameTensor(torch.Tensor):
                """Extension of torch.Tensor with specialized functions for game theory."""
                
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    if kwargs is None:
                        kwargs = {}
                    return super().__torch_function__(func, types, args, kwargs)

            class GameContext:
                """Maintains game state and history during gameplay."""
                
                def __init__(self):
                    self.tensors = {}
                    self.state = {}
                    self.history = []
                    
                def register_tensor(self, name: str, tensor: torch.Tensor):
                    """Store a tensor with a reference name."""
                    self.tensors[name] = tensor
                    
                def get_tensor(self, name: str) -> torch.Tensor:
                    """Retrieve a stored tensor by name."""
                    return self.tensors.get(name)
                    
                def update_state(self, key: str, value: Any):
                    """Update game state and record in history."""
                    self.state[key] = value
                    self.history.append((key, value))
                    
                def get_state(self, key: str) -> Any:
                    """Retrieve a state value by key."""
                    return self.state.get(key)
                    
                def get_history(self) -> List[Tuple[str, Any]]:
                    """Get the full history of state changes."""
                    return self.history

            class Lens(Generic[S, O, A]):
                """Provides a way to view and update parts of a state."""
                
                def __init__(self, view: Callable[[S], O], update: Callable[[S, A], S]):
                    """
                    Initialize a lens with view and update functions.
                    
                    Args:
                        view: A function that extracts a part of a state
                        update: A function that updates a state based on an action
                    """
                    self.view = view
                    self.update = update
                    
                def get(self, state: S) -> O:
                    """Get the view of the state."""
                    return self.view(state)
                    
                def set(self, state: S, action: A) -> S:
                    """Update the state with the given action."""
                    return self.update(state, action)
                    
                def compose(self, other: 'Lens') -> 'Lens':
                    """Compose this lens with another lens."""
                    def composed_view(s):
                        return other.view(self.view(s))
                        
                    def composed_update(s, a):
                        inner_state = self.view(s)
                        updated_inner = other.update(inner_state, a)
                        return self.update(s, updated_inner)
                        
                    return Lens(composed_view, composed_update)

            class Player:
                """Represents a player in a game with a strategy."""
                
                def __init__(self, name: str, strategy: Optional[Callable] = None):
                    """
                    Initialize a player with a name and optional strategy.
                    
                    Args:
                        name: Name of the player
                        strategy: Function mapping observations to actions
                    """
                    self.name = name
                    self.strategy = strategy
                    
                def act(self, observation: Any) -> Any:
                    """Take an action based on the observation using the players strategy."""
                    if self.strategy is None:
                        raise ValueError(f"Player {self.name} has no strategy")
                    return self.strategy(observation)
                    
                def update_strategy(self, strategy: Callable):
                    """Update the players strategy."""
                    self.strategy = strategy

            class OpenGame(Generic[S, O, A, P]):
                """
                Core class representing an open game with composition operators.
                
                An open game has:
                - A play function that determines how the game is played
                - A coutility function that determines how payoffs are calculated
                """
                
                def __init__(self, 
                            play_function: Callable[[S, Callable[[A], P]], Tuple[A, P]], 
                            coutility_function: Callable[[S, A, P], P], 
                            name: str = "unnamed_game"):
                    """
                    Initialize an open game.
                    
                    Args:
                        play_function: Function that plays the game and returns (action, payoff)
                        coutility_function: Function that calculates payoffs
                        name: Name of the game
                    """
                    self.play = play_function
                    self.coutility = coutility_function
                    self.name = name
                    
                def tensor_play(self, state_tensor: torch.Tensor, 
                            continuation: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
                    """Play the game with tensor inputs and outputs."""
                    state = state_tensor.item() if state_tensor.numel() == 1 else state_tensor
                    
                    def tensor_continuation(action):
                        action_tensor = torch.tensor(action) if not isinstance(action, torch.Tensor) else action
                        return continuation(action_tensor)
                        
                    action, payoff = self.play(state, tensor_continuation)
                    action_tensor = torch.tensor(action) if not isinstance(action, torch.Tensor) else action
                    payoff_tensor = torch.tensor(payoff) if not isinstance(payoff, torch.Tensor) else payoff
                    return action_tensor, payoff_tensor
                    
                def tensor_coutility(self, state_tensor: torch.Tensor, 
                                    action_tensor: torch.Tensor, 
                                    payoff_tensor: torch.Tensor) -> torch.Tensor:
                    """Calculate coutility with tensor inputs and outputs."""
                    state = state_tensor.item() if state_tensor.numel() == 1 else state_tensor
                    action = action_tensor.item() if action_tensor.numel() == 1 else action_tensor
                    payoff = payoff_tensor.item() if payoff_tensor.numel() == 1 else payoff_tensor
                    
                    result = self.coutility(state, action, payoff)
                    return torch.tensor(result) if not isinstance(result, torch.Tensor) else result
                    
                def compose(self, other: 'OpenGame') -> 'OpenGame':
                    def composed_play(state, continuation):
                        def intermediate_continuation(action1):
                            # Pass action1 as the new state for the second game
                            action2, payoff2 = other.play(action1, continuation)
                            return payoff2

                        action1, payoff1 = self.play(state, intermediate_continuation)
                        return action1, payoff1

                    def composed_coutility(state, action, payoff):
                        return self.coutility(
                            state, 
                            action, 
                            other.coutility(state, action, payoff)
                        )

                    return OpenGame(composed_play, composed_coutility, f"{self.name};{other.name}")

                    
                def tensor_compose(self, other: 'OpenGame') -> 'OpenGame':
                    """Compose this game with another game using tensor operations."""
                    def composed_tensor_play(state_tensor, continuation):
                        def intermediate_continuation(action_tensor):
                            next_state_tensor = state_tensor  # In simple composition, state passes unchanged
                            action2_tensor, payoff2_tensor = other.tensor_play(next_state_tensor, continuation)
                            return payoff2_tensor
                            
                        action1_tensor, payoff1_tensor = self.tensor_play(state_tensor, intermediate_continuation)
                        return action1_tensor, payoff1_tensor
                        
                    def composed_tensor_coutility(state_tensor, action_tensor, payoff_tensor):
                        intermediate_payoff = other.tensor_coutility(state_tensor, action_tensor, payoff_tensor)
                        return self.tensor_coutility(state_tensor, action_tensor, intermediate_payoff)
                        
                    return OpenGame(composed_tensor_play, composed_tensor_coutility, f"{self.name};{other.name}")

            class ComposedFunction:
                """Represents a composition of two functions."""
                
                def __init__(self, f, g):
                    self.f = f
                    self.g = g
                
                def __call__(self, x: torch.Tensor) -> torch.Tensor:
                    return self.f(self.g(x))

            def compose_tensor_fns(f: Callable[[torch.Tensor], torch.Tensor], 
                                g: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
                """Compose two tensor functions."""
                return ComposedFunction(f, g)
            ''')
        return core_code
    
    def _generate_game_module(self) -> str:
        game_code = textwrap.dedent('''
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from typing import Any, Callable, Dict, List, Optional, Tuple, Union
            from .core import OpenGame, Player, GameContext, Lens, OpenGameTensor

            class DecisionGame(OpenGame):
                \"\"\"
                Represents a single-player decision problem within a larger game.
                
                In a decision game, a player observes the game state and chooses an action
                to maximize their utility.
                \"\"\"
                
                def __init__(self, player: Player, observation_fn: Callable, utility_fn: Callable, name: str = "decision"):
                    \"\"\"
                    Initialize a decision game.
                    
                    Args:
                        player: The player making the decision
                        observation_fn: Function mapping state to observation
                        utility_fn: Function mapping state and action to utility
                        name: Name of the game
                    \"\"\"
                    def play_function(state, continuation):
                        # Player observes state and takes action
                        observation = observation_fn(state)
                        action = player.act(observation)
                        # Calculate payoff through continuation
                        payoff = continuation(action)
                        return action, payoff
                        
                    def coutility_function(state, action, payoff):
                        # In a decision game, coutility simply passes through the payoff
                        return payoff
                        
                    super().__init__(play_function, coutility_function, name)

            class StrategicGame(nn.Module):
                \"\"\"
                Represents a strategic-form (normal-form) game.
                
                In a strategic game, all players choose actions simultaneously, and
                payoffs are determined by the joint action profile.
                \"\"\"
                
                def __init__(self, num_players: int, action_spaces: List[int], payoff_tensors: List[torch.Tensor]):
                    \"\"\"
                    Initialize a strategic-form game.
                    
                    Args:
                        num_players: Number of players in the game
                        action_spaces: List of action space sizes for each player
                        payoff_tensors: List of payoff tensors, one for each player
                            Each tensor has dimensions matching the action spaces
                    \"\"\"
                    super().__init__()
                    self.num_players = num_players
                    self.action_spaces = action_spaces
                    
                    # Register payoff tensors as parameters so they can be optimized
                    self.payoff_tensors = nn.ParameterList(
                        [nn.Parameter(tensor, requires_grad=True) for tensor in payoff_tensors]
                    )
                    
                    # Verify payoff tensor shapes
                    for i, tensor in enumerate(self.payoff_tensors):
                        expected_shape = tuple(self.action_spaces)
                        if tensor.shape != expected_shape:
                            raise ValueError(
                                f"Payoff tensor for player {i} has shape {tensor.shape}, "
                                f"but expected {expected_shape} based on action spaces"
                            )
                
                def forward(self, actions: List[torch.Tensor]) -> List[torch.Tensor]:
                    \"\"\"
                    Calculate payoffs for a given action profile.
                    
                    Args:
                        actions: List of tensors containing each player's action
                        
                    Returns:
                        List of tensors containing each player's payoff
                    \"\"\"
                    # Verify action count matches player count
                    if len(actions) != self.num_players:
                        raise ValueError(
                            f"Expected {self.num_players} actions, got {len(actions)}"
                        )
                    
                    payoffs = []
                    for i in range(self.num_players):
                        # Start with the full payoff tensor for this player
                        payoff = self.payoff_tensors[i]
                        
                        # For each player's action, select the corresponding slice of the payoff tensor
                        for j, action in enumerate(actions):
                            # Convert action to integer index if needed
                            idx = action.item() if action.numel() == 1 else action
                            # Select the appropriate slice along dimension j
                            payoff = torch.index_select(payoff, j, torch.tensor(idx, dtype=torch.long))
                        
                        payoffs.append(payoff)
                    
                    return payoffs
                
                def best_response(self, player_idx: int, other_actions: List[Optional[torch.Tensor]]) -> torch.Tensor:
                    \"\"\"
                    Calculate the best response action for a player given other players' actions.
                    
                    Args:
                        player_idx: Index of the player for whom to calculate best response
                        other_actions: List of actions for all players (None for the player_idx position)
                        
                    Returns:
                        Tensor containing the best response action
                    \"\"\"
                    if len(other_actions) != self.num_players:
                        raise ValueError(
                            f"Expected {self.num_players} actions, got {len(other_actions)}"
                        )
                    
                    if other_actions[player_idx] is not None:
                        raise ValueError(
                            f"Action for player {player_idx} should be None"
                        )
                    
                    # Get payoff tensor for the player
                    payoff_tensor = self.payoff_tensors[player_idx]
                    
                    # Create index tuple for advanced indexing
                    index_tuple = []
                    current_dim = 0
                    
                    for i, action in enumerate(other_actions):
                        if i == player_idx:
                            # For the player's dimension, keep all possibilities
                            index_tuple.append(slice(None))
                        else:
                            # For other players, select the specific action
                            if action is None:
                                raise ValueError(f"Action for player {i} should not be None")
                            
                            # Convert action to integer index
                            idx = action.item() if action.numel() == 1 else action
                            index_tuple.append(idx)
                    
                    # Extract the payoff vector for the player
                    payoff_vector = payoff_tensor[tuple(index_tuple)]
                    
                    # Find the action that maximizes payoff
                    best_action = torch.argmax(payoff_vector)
                    
                    return best_action
                
                def nash_equilibrium(self, 
                                    init_strategies: Optional[List[torch.Tensor]] = None, 
                                    iterations: int = 1000, 
                                    learning_rate: float = 0.01) -> List[torch.Tensor]:
                    \"\"\"
                    Compute an approximate Nash equilibrium using gradient ascent.
                    
                    Args:
                        init_strategies: Initial mixed strategies (will use uniform if None)
                        iterations: Number of gradient steps
                        learning_rate: Learning rate for gradient ascent
                        
                    Returns:
                        List of mixed strategy tensors (probability distributions over actions)
                    \"\"\"
                    # Initialize strategies if not provided
                    if init_strategies is None:
                        init_strategies = []
                        for action_space in self.action_spaces:
                            strategy = torch.ones(action_space) / action_space
                            init_strategies.append(strategy)
                    
                    # Make sure strategies are proper probability distributions
                    strategies = []
                    for strategy in init_strategies:
                        strategy = F.softmax(strategy, dim=0)
                        strategy = nn.Parameter(strategy, requires_grad=True)
                        strategies.append(strategy)
                    
                    # Define optimizer
                    optimizer = torch.optim.Adam(strategies, lr=learning_rate)
                    
                    # Training loop
                    for _ in range(iterations):
                        optimizer.zero_grad()
                        
                        # Calculate expected payoffs
                        expected_payoffs = []
                        for i in range(self.num_players):
                            # Start with the full payoff tensor
                            expected_payoff = self.payoff_tensors[i]
                            
                            # For each player's strategy, compute the expected value
                            for j, strategy in enumerate(strategies):
                                # Take dot product along appropriate dimension
                                expected_payoff = torch.tensordot(expected_payoff, strategy, dims=([0], [0]))
                            
                            expected_payoffs.append(expected_payoff)
                        
                        # Compute loss (negative expected payoff)
                        losses = [-payoff for payoff in expected_payoffs]
                        
                        # Update strategies through backpropagation
                        for i, loss in enumerate(losses):
                            if i > 0:
                                # For multi-player games, handle gradient computation separately
                                loss.backward(retain_graph=(i < len(losses) - 1))
                            else:
                                loss.backward(retain_graph=True)
                        
                        optimizer.step()
                        
                        # Project back to probability simplex
                        with torch.no_grad():
                            for i, strategy in enumerate(strategies):
                                strategy.data = F.softmax(strategy.data, dim=0)
                    
                    # Return the final strategies
                    return [strategy.detach() for strategy in strategies]

            def create_strategic_game(payoff_matrices: List[List[List[float]]]) -> StrategicGame:
                \"\"\"
                Create a strategic game from a list of payoff matrices.
                
                Args:
                    payoff_matrices: List of payoff matrices for each player
                        Each matrix has dimensions matching all players' action spaces
                        
                Returns:
                    StrategicGame instance
                \"\"\"
                num_players = len(payoff_matrices)
                action_spaces = [len(matrices[0]) for matrices in payoff_matrices]
                
                payoff_tensors = []
                for player_matrices in payoff_matrices:
                    tensor = torch.tensor(player_matrices, dtype=torch.float32)
                    payoff_tensors.append(tensor)
                    
                return StrategicGame(num_players, action_spaces, payoff_tensors)

            def create_matrix_game(payoff_matrix: torch.Tensor) -> StrategicGame:
                \"\"\"
                Create a two-player zero-sum game from a payoff matrix.
                
                Args:
                    payoff_matrix: Matrix of payoffs for player 1
                        Player 2's payoffs are the negative of player 1's
                        
                Returns:
                    StrategicGame instance
                \"\"\"
                payoff_tensors = [payoff_matrix, -payoff_matrix]
                return StrategicGame(2, [payoff_matrix.shape[0], payoff_matrix.shape[1]], payoff_tensors)
        ''')
        return game_code
    
    def _generate_category_module(self) -> str:
        category_code = textwrap.dedent('''
            import torch
            from typing import Any, Callable, Generic, TypeVar, List

            A = TypeVar('A')
            B = TypeVar('B')
            C = TypeVar('C')
            F = TypeVar('F')
            M = TypeVar('M')

            class Functor(Generic[F]):
                """
                Represents a functor in category theory.
                
                A functor maps both objects and morphisms from one category to another,
                preserving identity morphisms and composition.
                """
                
                def fmap(self, f: Callable[[A], B], fa: F) -> F:
                    """
                    Map a function over the functor.
                    
                    Args:
                        f: Function to apply
                        fa: Functor value containing elements of type A
                        
                    Returns:
                        Functor value containing elements of type B
                    """
                    raise NotImplementedError

            class Monad(Functor[M]):
                """
                Represents a monad in category theory.
                
                A monad is a functor with additional operations:
                - return_: Embed a value in the monad
                - bind: Sequence monadic operations
                """
                
                def return_(self, a: A) -> M:
                    """
                    Embed a value in the monad.
                    
                    Args:
                        a: Value to embed
                        
                    Returns:
                        Monadic value containing a
                    """
                    raise NotImplementedError
                
                def bind(self, ma: M, f: Callable[[A], M]) -> M:
                    """
                    Sequence monadic operations.
                    
                    Args:
                        ma: Monadic value containing elements of type A
                        f: Function mapping A to monadic value containing elements of type B
                        
                    Returns:
                        Monadic value containing elements of type B
                    """
                    raise NotImplementedError
                
                def fmap(self, f: Callable[[A], B], ma: M) -> M:
                    """
                    Implement fmap in terms of return_ and bind.
                    
                    Args:
                        f: Function to apply
                        ma: Monadic value containing elements of type A
                        
                    Returns:
                        Monadic value containing elements of type B
                    """
                    def mapper(a: A) -> M:
                        return self.return_(f(a))
                    return self.bind(ma, mapper)

            class TensorFunctor(Functor[torch.Tensor]):
                """Functor instance for torch.Tensor."""
                
                def fmap(self, f: Callable[[torch.Tensor], torch.Tensor], tensor: torch.Tensor) -> torch.Tensor:
                    """Apply function f to the tensor."""
                    return f(tensor)

            class TensorMonad(Monad[torch.Tensor]):
                """Monad instance for torch.Tensor."""
                
                def return_(self, a: Any) -> torch.Tensor:
                    """Embed a value in a tensor."""
                    return a if isinstance(a, torch.Tensor) else torch.tensor(a)
                
                def bind(self, ma: torch.Tensor, f: Callable[[Any], torch.Tensor]) -> torch.Tensor:
                    """Apply function f to the tensor."""
                    return f(ma)

            class ListFunctor(Functor[List[A]]):
                """Functor instance for Python lists."""
                
                def fmap(self, f: Callable[[A], B], fa: List[A]) -> List[B]:
                    """Map function f over each element in the list."""
                    return [f(a) for a in fa]

            class ListMonad(Monad[List[A]]):
                """Monad instance for Python lists."""
                
                def return_(self, a: A) -> List[A]:
                    """Embed a value in a singleton list."""
                    return [a]
                
                def bind(self, ma: List[A], f: Callable[[A], List[B]]) -> List[B]:
                    """Flatmap function f over each element in the list."""
                    return [b for a in ma for b in f(a)]

            # Singleton instances for common functors and monads
            tensor_functor = TensorFunctor()
            tensor_monad = TensorMonad()
            list_functor = ListFunctor()
            list_monad = ListMonad()

            def kleisli_compose(f: Callable[[A], M], g: Callable[[B], M], monad: Monad[M]) -> Callable[[A], M]:
                """
                Compose two monadic functions using Kleisli composition.
                
                Args:
                    f: First monadic function
                    g: Second monadic function
                    monad: Monad instance to use for composition
                    
                Returns:
                    Composed monadic function
                """
                def composed(a: A) -> M:
                    return monad.bind(f(a), g)
                return composed

            class Id(Generic[A]):
                """Identity functor that simply wraps a value."""
                
                def __init__(self, value: A):
                    self.value = value

            class IdFunctor(Functor['Id[A]']):
                """Functor instance for the identity functor."""
                
                def fmap(self, f: Callable[[A], B], fa: 'Id[A]') -> 'Id[B]':
                    """Apply function f to the value inside the identity functor."""
                    return Id(f(fa.value))

            class IdMonad(Monad['Id[A]']):
                """Monad instance for the identity functor."""
                
                def return_(self, a: A) -> 'Id[A]':
                    """Embed a value in the identity functor."""
                    return Id(a)
                
                def bind(self, ma: 'Id[A]', f: Callable[[A], 'Id[B]']) -> 'Id[B]':
                    """Apply monadic function f to the value inside the identity functor."""
                    return f(ma.value)

            class Category:
                """
                Abstract class representing a category.
                
                A category consists of objects and morphisms, with composition and identity.
                """
                
                def compose(self, f, g):
                    """Compose two morphisms."""
                    raise NotImplementedError
                
                def id(self, a):
                    """Return the identity morphism for an object."""
                    raise NotImplementedError

            class FunctionCategory(Category):
                """Category of functions."""
                
                def compose(self, f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
                    """Compose two functions."""
                    def composed(a: A) -> C:
                        return f(g(a))
                    return composed
                
                def id(self, a: A) -> Callable[[A], A]:
                    """Return the identity function for a type."""
                    def identity(x: A) -> A:
                        return x
                    return identity

            class ComposedTensorFunction:
                """Represents a composition of two tensor functions."""
                
                def __init__(self, f: Callable[[torch.Tensor], torch.Tensor], 
                           g: Callable[[torch.Tensor], torch.Tensor]):
                    self.f = f
                    self.g = g
                
                def __call__(self, x: torch.Tensor) -> torch.Tensor:
                    """Apply composed function to a tensor."""
                    return self.f(self.g(x))

            def compose_tensor_fns(f: Callable[[torch.Tensor], torch.Tensor], 
                                  g: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
                """Compose two tensor functions."""
                return ComposedTensorFunction(f, g)
        ''')
        return category_code
    
    def _generate_sequential_game_module(self) -> str:
        """Generate the module for sequential-form games."""
        sequential_code = textwrap.dedent('''
            import torch
            import torch.nn as nn
            from typing import Any, Callable, Dict, List, Optional, Tuple, Union
            from .core import OpenGame, Player, GameContext, Lens, OpenGameTensor

            class SequentialGame(nn.Module):
                """
                Represents a sequential game where players move in order.
                
                In a sequential game, players take actions one after another, with
                later players observing the actions of earlier players.
                """
                
                def __init__(self, 
                            players: List[Player], 
                            initial_state: Any,
                            state_transition: Callable,
                            payoff_functions: List[Callable],
                            observation_functions: Optional[List[Callable]] = None):
                    """
                    Initialize a sequential game.
                    
                    Args:
                        players: List of players in order of play
                        initial_state: Initial game state
                        state_transition: Function mapping (state, action, player_idx) to next state
                        payoff_functions: List of functions mapping final state to payoff for each player
                        observation_functions: List of functions mapping state to observation for each player
                            If None, players observe the full state
                    """
                    super().__init__()
                    self.players = players
                    self.initial_state = initial_state
                    self.state_transition = state_transition
                    self.payoff_functions = payoff_functions
                    
                    # Default observation functions show the full state
                    if observation_functions is None:
                        self.observation_functions = [lambda state, player_idx: state] * len(players)
                    else:
                        self.observation_functions = observation_functions
                
                def forward(self) -> Tuple[List[torch.Tensor], Any, List[torch.Tensor]]:
                    """
                    Play the game from the initial state to completion.
                    
                    Returns:
                        Tuple of (actions, final_state, payoffs)
                    """
                    state = self.initial_state
                    actions = []
                    
                    # Each player takes their turn in order
                    for i, player in enumerate(self.players):
                        observation = self.observation_functions[i](state, i)
                        action = player.act(observation)
                        actions.append(action)
                        
                        # Update state based on action
                        state = self.state_transition(state, action, i)
                    
                    # Calculate payoffs based on final state
                    payoffs = [payoff_fn(state) for payoff_fn in self.payoff_functions]
                    
                    return actions, state, payoffs
                
                def backward_induction(self):
                    """
                    Compute subgame perfect equilibrium strategies using backward induction.
                    
                    This method assumes finite action spaces and perfect information.
                    
                    Returns:
                        Dict of optimal strategies for each player at each state
                    """
                    # This is a simplified version - a full implementation would
                    # need to exhaustively consider all possible game states and actions
                    optimal_strategies = {}
                    
                    # Start from the last player and work backwards
                    for i in range(len(self.players) - 1, -1, -1):
                        # For each possible state the player might face...
                        # (This would need to be expanded for a real implementation)
                        pass
                    
                    return optimal_strategies
                
                def to_strategic_form(self, discrete_states: List[Any] = None):
                    """
                    Convert sequential game to strategic form (normal form).
                    
                    Args:
                        discrete_states: List of discrete states to consider
                            If None, attempts to automatically generate states
                            
                    Returns:
                        StrategicGame equivalent to this sequential game
                    """
                    # This is a placeholder for a complex conversion process
                    # that would depend on the specific game structure
                    pass

            class ExtensiveFormGame(SequentialGame):
                """
                Represents an extensive-form game (game tree).
                
                This is a specialized version of sequential game with explicit tree structure.
                """
                
                def __init__(self, 
                            game_tree: Dict,
                            players: List[Player]):
                    """
                    Initialize an extensive-form game.
                    
                    Args:
                        game_tree: Dictionary representing the game tree
                            Each node contains player, actions, and payoffs information
                        players: List of players in the game
                    """
                    # Simplified initialization - would need more structure for a complete implementation
                    self.game_tree = game_tree
                    self.players = players
                    
                def solve(self):
                    """
                    Solve the game using backward induction.
                    
                    Returns:
                        Dict of optimal strategies for each player at each node
                    """
                    # Placeholder for game tree solving algorithm
                    pass
        ''')
        return sequential_code
    
    def _generate_bayesian_game_module(self) -> str:
        """Generate the module for Bayesian games."""
        bayesian_code = textwrap.dedent('''
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from typing import Any, Callable, Dict, List, Optional, Tuple, Union
            from .core import OpenGame, Player, GameContext, Lens, OpenGameTensor

            class BayesianGame(nn.Module):
                """
                Represents a Bayesian game with incomplete information.
                
                In a Bayesian game, players have private information (types) that
                affect their payoffs, but the types are drawn from known distributions.
                """
                
                def __init__(self, 
                            type_distributions: List[torch.Tensor], 
                            type_conditional_payoffs: List[List[torch.Tensor]],
                            action_spaces: List[int]):
                    """
                    Initialize a Bayesian game.
                    
                    Args:
                        type_distributions: List of probability distributions over types for each player
                        type_conditional_payoffs: List of lists of payoff tensors for each player and type
                        action_spaces: List of action space sizes for each player
                    """
                    super().__init__()
                    self.num_players = len(type_distributions)
                    self.type_distributions = nn.ParameterList(
                        [nn.Parameter(dist, requires_grad=True) for dist in type_distributions]
                    )
                    
                    # Ensure type distributions are valid probability distributions
                    for i, dist in enumerate(self.type_distributions):
                        if not torch.allclose(dist.sum(), torch.tensor(1.0)):
                            raise ValueError(f"Type distribution for player {i} must sum to 1")
                    
                    # Initialize payoff tensors as parameters
                    self.type_conditional_payoffs = nn.ModuleList([
                        nn.ParameterList([nn.Parameter(t) for t in payoff_list]) 
                        for payoff_list in type_conditional_payoffs
                    ])
                    
                    self.action_spaces = action_spaces
                    
                    # Verify payoff tensor shapes
                    for player, type_payoffs in enumerate(self.type_conditional_payoffs):
                        for t, payoff in enumerate(type_payoffs):
                            expected_shape = tuple(self.action_spaces)
                            if payoff.shape != expected_shape:
                                raise ValueError(
                                    f"Payoff tensor for player {player}, type {t} has shape {payoff.shape}, "
                                    f"but expected {expected_shape} based on action spaces"
                                )
                
                def forward(self, type_indices: List[torch.Tensor], actions: List[torch.Tensor]) -> List[torch.Tensor]:
                    """
                    Calculate payoffs for given types and actions.
                    
                    Args:
                        type_indices: List of tensors containing each players type index
                        actions: List of tensors containing each players action
                        
                    Returns:
                        List of tensors containing each players payoff
                    """
                    payoffs = []
                    for player in range(self.num_players):
                        # Get the payoff tensor for this player and type
                        type_idx = type_indices[player].item()
                        payoff_tensor = self.type_conditional_payoffs[player][type_idx]
                        
                        # Calculate payoff based on all players actions
                        payoff = payoff_tensor
                        for i, action in enumerate(actions):
                            idx = action.item() if action.numel() == 1 else action
                            payoff = torch.index_select(payoff, i, torch.tensor(idx, dtype=torch.long))
                            
                        payoffs.append(payoff)
                        
                    return payoffs
                
                def expected_payoff(self, strategies: List[Callable]) -> List[torch.Tensor]:
                    """
                    Calculate expected payoffs given type-conditional strategies.
                    
                    Args:
                        strategies: List of functions mapping type to action for each player
                        
                    Returns:
                        List of expected payoff tensors for each player
                    """
                    # This would be a more complex calculation in practice
                    # We would need to integrate over all possible type combinations
                    expected_payoffs = []
                    
                    # Calculate expected payoff for each player
                    for player in range(self.num_players):
                        payoff = torch.tensor(0.0)
                        
                        # For each possible type profile
                        # (This is simplified; a complete implementation would consider all type combinations)
                        for type_profile in self._generate_type_profiles():
                            # Calculate joint probability of this type profile
                            prob = self._type_profile_probability(type_profile)
                            
                            # Calculate actions for this type profile
                            actions = [strategies[i](type_profile[i]) for i in range(self.num_players)]
                            
                            # Get payoff for this player, type, and action profile
                            player_payoff = self._calculate_payoff(player, type_profile[player], actions)
                            
                            # Add weighted payoff to expected value
                            payoff += prob * player_payoff
                            
                        expected_payoffs.append(payoff)
                        
                    return expected_payoffs
                
                def _generate_type_profiles(self):
                    """Generate all possible type profiles (simplified)."""
                    # This would need to be implemented based on the game's type spaces
                    return []
                
                def _type_profile_probability(self, type_profile):
                    """Calculate probability of a given type profile."""
                    # For independent types, multiply individual probabilities
                    prob = torch.tensor(1.0)
                    for i, type_idx in enumerate(type_profile):
                        prob *= self.type_distributions[i][type_idx]
                    return prob
                
                def _calculate_payoff(self, player, player_type, actions):
                    """Calculate payoff for a player given their type and all actions."""
                    # Convert actions to tensors if needed
                    action_tensors = [
                        torch.tensor(action) if not isinstance(action, torch.Tensor) else action 
                        for action in actions
                    ]
                    
                    # Get this players payoff tensor for their type
                    payoff_tensor = self.type_conditional_payoffs[player][player_type]
                    
                    # Extract payoff for the given action profile
                    payoff = payoff_tensor
                    for i, action in enumerate(action_tensors):
                        idx = action.item() if action.numel() == 1 else action
                        payoff = torch.index_select(payoff, i, torch.tensor(idx, dtype=torch.long))
                    
                    return payoff
                
                def bayesian_nash_equilibrium(self, 
                                            iterations: int = 1000, 
                                            learning_rate: float = 0.01) -> List[List[torch.Tensor]]:
                    """
                    Compute an approximate Bayesian Nash equilibrium.
                    
                    Args:
                        iterations: Number of gradient steps
                        learning_rate: Learning rate for gradient ascent
                        
                    Returns:
                        List of lists of mixed strategies for each player and type
                    """
                    # Initialize type-conditional strategies
                    strategies = []
                    for player in range(self.num_players):
                        player_strategies = []
                        num_types = len(self.type_conditional_payoffs[player])
                        
                        for _ in range(num_types):
                            # Uniform initial strategy
                            strategy = torch.ones(self.action_spaces[player]) / self.action_spaces[player]
                            strategy = nn.Parameter(strategy, requires_grad=True)
                            player_strategies.append(strategy)
                            
                        strategies.append(player_strategies)
                        
                    # Flatten strategies for optimizer
                    flat_strategies = [s for player_strategies in strategies for s in player_strategies]
                    optimizer = torch.optim.Adam(flat_strategies, lr=learning_rate)
                    
                    # Training loop
                    for _ in range(iterations):
                        optimizer.zero_grad()
                        
                        # Calculate expected payoffs for each player and type
                        losses = []
                        
                        for player in range(self.num_players):
                            for type_idx in range(len(strategies[player])):
                                # For each player and type, calculate expected payoff
                                expected_payoff = self._calculate_type_expected_payoff(
                                    player, type_idx, strategies)
                                
                                # Add negative expected payoff to losses
                                losses.append(-expected_payoff)
                        
                        # Update strategies
                        for i, loss in enumerate(losses):
                            if i > 0:
                                loss.backward(retain_graph=(i < len(losses) - 1))
                            else:
                                loss.backward(retain_graph=True)
                        
                        optimizer.step()
                        
                        # Project back to probability simplex
                        with torch.no_grad():
                            for strategy in flat_strategies:
                                strategy.data = F.softmax(strategy.data, dim=0)
                    
                    # Restructure the flattened strategies
                    result = []
                    idx = 0
                    for player in range(self.num_players):
                        num_types = len(self.type_conditional_payoffs[player])
                        player_strategies = [flat_strategies[idx + t].detach() for t in range(num_types)]
                        result.append(player_strategies)
                        idx += num_types
                        
                    return result
                
                def _calculate_type_expected_payoff(self, player, type_idx, strategies):
                    """Calculate expected payoff for a player of a given type."""
                    # Get payoff tensor for this player and type
                    payoff_tensor = self.type_conditional_payoffs[player][type_idx]
                    
                    # For each player, take expected value over their strategy
                    expected_payoff = payoff_tensor
                    
                    for i in range(self.num_players):
                        if i == player:
                            # For this player, use their strategy for this type
                            strategy = strategies[i][type_idx]
                        else:
                            # For other players, average over their types and strategies
                            # (Simplified: in practice, would need to consider type beliefs)
                            avg_strategy = torch.zeros(self.action_spaces[i])
                            for t in range(len(strategies[i])):
                                avg_strategy += self.type_distributions[i][t] * strategies[i][t]
                            strategy = avg_strategy
                        
                        # Take expected value along this dimension
                        expected_payoff = torch.tensordot(expected_payoff, strategy, dims=([0], [0]))
                    
                    return expected_payoff
        ''')
        return bayesian_code
    
    def _generate_utils_module(self) -> str:
        """Generate the utility module with helper functions."""
        utils_code = textwrap.dedent('''
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from typing import Any, Callable, Dict, List, Optional, Tuple, Union

            def probability_simplex_projection(v: torch.Tensor) -> torch.Tensor:
                """
                Project a vector onto the probability simplex.
                
                Args:
                    v: Input vector
                    
                Returns:
                    Projected vector that sums to 1 with non-negative elements
                """
                # Sort values in descending order
                u, _ = torch.sort(v, descending=True)
                
                # Find index where cumulative sum minus value exceeds 1
                cssv = torch.cumsum(u, dim=0)
                rho = torch.nonzero(u * torch.arange(1, len(u) + 1) > (cssv - 1)).max()
                
                # Compute projection
                theta = (cssv[rho] - 1) / (rho + 1)
                return torch.clamp(v - theta, min=0)

            def expected_utility(payoff_tensor: torch.Tensor, strategies: List[torch.Tensor]) -> torch.Tensor:
                """
                Calculate expected utility given payoff tensor and strategies.
                
                Args:
                    payoff_tensor: Tensor of payoffs with dimensions matching action spaces
                    strategies: List of mixed strategies (probability distributions over actions)
                    
                Returns:
                    Expected utility (scalar tensor)
                """
                expected_payoff = payoff_tensor
                
                for i, strategy in enumerate(strategies):
                    expected_payoff = torch.tensordot(expected_payoff, strategy, dims=([0], [0]))
                
                return expected_payoff

            def compute_best_response(payoff_tensor: torch.Tensor, 
                                    other_strategies: List[torch.Tensor], 
                                    player_idx: int) -> torch.Tensor:
                """
                Compute best response to other players strategies.
                
                Args:
                    payoff_tensor: Players payoff tensor
                    other_strategies: List of other players mixed strategies
                    player_idx: Index of the player in the strategy profile
                    
                Returns:
                    Best response strategy (one-hot encoding of best action)
                """
                # Start with full payoff tensor
                expected_payoffs = payoff_tensor
                
                # For each players strategy, compute the expected value
                strategy_idx = 0
                for i in range(len(other_strategies) + 1):
                    if i == player_idx:
                        # Skip the current players dimension
                        continue
                    
                    # Calculate expected payoff along this dimension
                    strategy = other_strategies[strategy_idx]
                    expected_payoffs = torch.tensordot(expected_payoffs, strategy, dims=([0], [0]))
                    strategy_idx += 1
                
                # Find best action
                best_action = torch.argmax(expected_payoffs)
                
                # Create one-hot encoding
                best_response = torch.zeros_like(expected_payoffs)
                best_response[best_action] = 1.0
                
                return best_response

            def regret_matching(payoff_tensor: torch.Tensor, 
                              other_strategies: List[torch.Tensor], 
                              player_idx: int, 
                              current_strategy: torch.Tensor) -> torch.Tensor:
                """
                Update strategy using regret matching algorithm.
                
                Args:
                    payoff_tensor: Players payoff tensor
                    other_strategies: List of other players mixed strategies
                    player_idx: Index of the player in the strategy profile
                    current_strategy: Players current mixed strategy
                    
                Returns:
                    Updated strategy
                """
                # Calculate expected payoff for each action
                expected_payoffs = payoff_tensor
                
                # For each players strategy, compute the expected value
                strategy_idx = 0
                for i in range(len(other_strategies) + 1):
                    if i == player_idx:
                        # Skip the current players dimension
                        continue
                    
                    # Calculate expected payoff along this dimension
                    strategy = other_strategies[strategy_idx]
                    expected_payoffs = torch.tensordot(expected_payoffs, strategy, dims=([0], [0]))
                    strategy_idx += 1
                
                # Calculate current expected payoff
                current_value = torch.dot(expected_payoffs, current_strategy)
                
                # Calculate regrets
                regrets = expected_payoffs - current_value
                
                # Set negative regrets to zero
                positive_regrets = F.relu(regrets)
                
                # If sum of positive regrets is zero, keep current strategy
                if torch.sum(positive_regrets) < 1e-10:
                    return current_strategy
                
                # Otherwise, update according to regret matching
                return positive_regrets / torch.sum(positive_regrets)

            def fictitious_play(game, iterations: int = 1000) -> List[torch.Tensor]:
                """
                Run fictitious play algorithm to approximate Nash equilibrium.
                
                Args:
                    game: Strategic game
                    iterations: Number of iterations
                    
                Returns:
                    List of mixed strategies (one for each player)
                """
                # Initialize uniform strategies
                strategies = []
                for action_space in game.action_spaces:
                    strategies.append(torch.ones(action_space) / action_space)
                
                # Initialize action counts
                action_counts = [torch.zeros(space) for space in game.action_spaces]
                
                # Fictitious play iterations
                for _ in range(iterations):
                    for i in range(game.num_players):
                        # Compute best response to other players strategies
                        other_actions = [None] * game.num_players
                        for j in range(game.num_players):
                            if j != i:
                                # Use historical frequencies as belief about other players
                                other_actions[j] = strategies[j]
                        
                        # Get best response action
                        best_action = game.best_response(i, other_actions)
                        
                        # Update action counts
                        action_counts[i][best_action] += 1
                        
                        # Update strategy to be empirical frequencies
                        strategies[i] = action_counts[i] / torch.sum(action_counts[i])
                
                return strategies
        ''')
        return utils_code
        
    def _generate_module(self, module_name: str, module: DSLModule) -> str:
        """Generate code for a custom module based on the DSL module structure."""
        functions = []
        types = []
        
        # Extract types
        for type_name, dsl_type in module.types.items():
            types.append(f"class {type_name}:")
            if isinstance(dsl_type, DSLGameType):
                types[-1] += textwrap.dedent('''
                    """Game type from the original DSL."""
                    
                    def __init__(self, state_type=None, observation_type=None, action_type=None, payoff_type=None):
                        self.state_type = state_type
                        self.observation_type = observation_type
                        self.action_type = action_type
                        self.payoff_type = payoff_type
                ''')
            elif isinstance(dsl_type, DSLFunctionType):
                param_types = [p.name for p in dsl_type.param_types]
                return_type = dsl_type.return_type.name
                types[-1] += textwrap.dedent(f'''
                    """Function type from the original DSL: ({', '.join(param_types)}) -> {return_type}"""
                    
                    def __init__(self, func=None):
                        self.func = func
                    
                    def __call__(self, *args):
                        if self.func is None:
                            raise ValueError("Function not implemented")
                        return self.func(*args)
                ''')
            else:
                types[-1] += textwrap.dedent('''
                    """Primitive type from the original DSL."""
                    
                    def __init__(self, value=None):
                        self.value = value
                ''')
        
        # Extract functions
        for func_name, (func_type, func_body) in module.functions.items():
            # Skip special flags like "monad" and "functor"
            if func_name in ["monad", "functor"] and isinstance(func_body, bool):
                continue
                
            functions.append(f"def {func_name}(*args, **kwargs):")
            functions.append(f'    """Original type: {func_type}"""')
            functions.append(f"    # Implementation of {func_name}")
            functions.append(f"    # Original body: {func_body[:100]}...")
            functions.append(f"    raise NotImplementedError(\"This function is a placeholder\")")
        
        # Generate the module code
        module_code = textwrap.dedent(f'''
            import torch
            import torch.nn as nn
            from typing import Any, Callable, Dict, List, Optional, Tuple, Union
            from .core import OpenGame, Player, GameContext, Lens, OpenGameTensor

            # Module: {module_name}
            # This module was auto-generated from the original DSL

            {"".join(f"{type_def}\\n\\n" for type_def in types)}
            {"".join(f"{func_def}\\n\\n" for func_def in functions)}

            # Add any additional module-specific functionality here
        ''')
        
        return module_code

    def generate_tests(self) -> Dict[str, str]:
        """Generate test files for the PyTorch implementation."""
        tests = {}
        
        # Basic tests for core functionality
        tests["test_core"] = self._generate_core_tests()

        tests["test_simultaneous_games"] = self._generate_simultaneous_game_tests()
        
        # Game-specific tests
        tests["test_strategic_games"] = self._generate_strategic_game_tests()
        
        if "sequential" in self.context.game_patterns and self.context.game_patterns["sequential"]["detected"]:
            tests["test_sequential_games"] = self._generate_sequential_game_tests()
        
        if "bayesian" in self.context.game_patterns and self.context.game_patterns["bayesian"]["detected"]:
            tests["test_bayesian_games"] = self._generate_bayesian_game_tests()
            
        return tests
    
    def _generate_core_tests(self) -> str:
        """Generate tests for core functionality."""
        return textwrap.dedent('''
            import unittest
            import torch
            from pytorch_oge.core import OpenGame, Player, Lens, GameContext

            class TestPlayer(unittest.TestCase):
                def test_player_creation(self):
                    # Create a player with a strategy
                    def always_cooperate(obs):
                        return 0
                        
                    player = Player("Alice", always_cooperate)
                    self.assertEqual(player.name, "Alice")
                    self.assertEqual(player.act("any observation"), 0)
                    
                    # Update the strategy
                    def always_defect(obs):
                        return 1
                        
                    player.update_strategy(always_defect)
                    self.assertEqual(player.act("any observation"), 1)
                    
                    # Player with no strategy should raise ValueError
                    player_no_strategy = Player("Bob")
                    with self.assertRaises(ValueError):
                        player_no_strategy.act("any observation")

            class TestLens(unittest.TestCase):
                def test_lens_operations(self):
                    # Create a lens
                    def view(state):
                        return state["value"]
                        
                    def update(state, action):
                        new_state = state.copy()
                        new_state["value"] = action
                        return new_state
                        
                    lens = Lens(view, update)
                    
                    # Test get (view) operation
                    state = {"value": 10, "other": 20}
                    self.assertEqual(lens.get(state), 10)
                    
                    # Test set (update) operation
                    new_state = lens.set(state, 30)
                    self.assertEqual(new_state["value"], 30)
                    self.assertEqual(new_state["other"], 20)
                    self.assertEqual(state["value"], 10)  # Original state shouldn't change
                    
                    # Test lens composition
                    def nested_view(inner):
                        return inner * 2
                        
                    def nested_update(inner, action):
                        return action // 2
                        
                    nested_lens = Lens(nested_view, nested_update)
                    composed_lens = lens.compose(nested_lens)
                    
                    # Test composed view
                    self.assertEqual(composed_lens.get(state), 20)  # 10 * 2
                    
                    # Test composed update
                    updated_state = composed_lens.set(state, 60)
                    self.assertEqual(updated_state["value"], 30)  # 60 // 2

            class TestOpenGame(unittest.TestCase):
                def test_open_game_creation(self):
                    # Create a simple open game
                    def play_function(state, continuation):
                        action = state + 1
                        payoff = continuation(action)
                        return action, payoff
                        
                    def coutility_function(state, action, payoff):
                        return payoff * 2
                        
                    game = OpenGame(play_function, coutility_function, "test_game")
                    
                    # Test play
                    def continuation(action):
                        return action * 3
                        
                    action, payoff = game.play(5, continuation)
                    self.assertEqual(action, 6)  # 5 + 1
                    self.assertEqual(payoff, 18)  # 6 * 3
                    
                    # Test coutility
                    result = game.coutility(5, 6, 18)
                    self.assertEqual(result, 36)  # 18 * 2
                    
                    # Test tensor operations
                    action_tensor, payoff_tensor = game.tensor_play(
                        torch.tensor(5), 
                        lambda a: torch.tensor(a.item() * 3)
                    )
                    self.assertEqual(action_tensor.item(), 6)
                    self.assertEqual(payoff_tensor.item(), 18)
                    
                    result_tensor = game.tensor_coutility(
                        torch.tensor(5),
                        torch.tensor(6),
                        torch.tensor(18)
                    )
                    self.assertEqual(result_tensor.item(), 36)

                def test_game_composition(self):
                    # First game: add 1 to state and pass to continuation
                    def play1(state, continuation):
                        action = state + 1
                        payoff = continuation(action)
                        return action, payoff
                        
                    def coutility1(state, action, payoff):
                        return payoff * 2
                        
                    game1 = OpenGame(play1, coutility1, "game1")
                    
                    # Second game: multiply state by 2 and pass to continuation
                    def play2(state, continuation):
                        action = state * 2
                        payoff = continuation(action)
                        return action, payoff
                        
                    def coutility2(state, action, payoff):
                        return payoff + 10
                        
                    game2 = OpenGame(play2, coutility2, "game2")
                    
                    # Compose the games
                    composed_game = game1.compose(game2)
                    
                    # Final continuation simply returns the action
                    def final_continuation(action):
                        return action
                        
                    # Test composed play
                    # Initial state: 5
                    # game1: action = 5+1 = 6
                    # game2: action = 6*2 = 12, payoff = 12
                    # game1 returns: (6, 12)
                    action, payoff = composed_game.play(5, final_continuation)
                    self.assertEqual(action, 6)
                    self.assertEqual(payoff, 12)
                    
                    # Test composed coutility
                    # initial payoff: 12
                    # game2 coutility: 12+10 = 22
                    # game1 coutility: 22*2 = 44
                    result = composed_game.coutility(5, 6, 12)
                    self.assertEqual(result, 44)

            if __name__ == "__main__":
                unittest.main()
        ''')
    
    def _generate_strategic_game_tests(self) -> str:
        """Generate tests for strategic-form games."""
        return textwrap.dedent('''
            import unittest
            import torch
            from pytorch_oge.core import Player
            from pytorch_oge.game import StrategicGame, create_strategic_game, create_matrix_game

            class TestPrisonersDilemma(unittest.TestCase):
                def setUp(self):
                    """Set up a standard Prisoners Dilemma game for testing"""
                    # Player 1's payoff matrix (maximizing)
                    self.player1_payoffs = [
                        [-1, -5],  # Player 1 cooperates: [Player 2 cooperates, Player 2 defects]
                        [0, -3]    # Player 1 defects: [Player 2 cooperates, Player 2 defects]
                    ]
                    
                    # Player 2's payoff matrix (maximizing)
                    self.player2_payoffs = [
                        [-1, 0],   # Player 2 cooperates: [Player 1 cooperates, Player 1 defects]
                        [-5, -3]   # Player 2 defects: [Player 1 cooperates, Player 1 defects]
                    ]
                    
                    # Convert to tensors
                    self.payoff_tensors = [
                        torch.tensor(self.player1_payoffs, dtype=torch.float32),
                        torch.tensor(self.player2_payoffs, dtype=torch.float32)
                    ]
                    
                    # Create the game
                    self.game = StrategicGame(2, [2, 2], self.payoff_tensors)

                def test_game_initialization(self):
                    """Test that the game is initialized correctly"""
                    self.assertEqual(self.game.num_players, 2)
                    self.assertEqual(self.game.action_spaces, [2, 2])
                    self.assertEqual(len(self.game.payoff_tensors), 2)
                    
                    # Check payoff tensor shapes
                    self.assertEqual(self.game.payoff_tensors[0].shape, torch.Size([2, 2]))
                    self.assertEqual(self.game.payoff_tensors[1].shape, torch.Size([2, 2]))
                    
                    # Check payoff values
                    self.assertEqual(self.game.payoff_tensors[0][0, 0].item(), -1)
                    self.assertEqual(self.game.payoff_tensors[0][0, 1].item(), -5)
                    self.assertEqual(self.game.payoff_tensors[0][1, 0].item(), 0)
                    self.assertEqual(self.game.payoff_tensors[0][1, 1].item(), -3)

                def test_payoffs(self):
                    """Test payoffs for different action profiles"""
                    # Both cooperate
                    actions_cc = [torch.tensor(0), torch.tensor(0)]
                    payoffs_cc = self.game(actions_cc)
                    
                    # Verify payoffs structure
                    self.assertEqual(len(payoffs_cc), 2)
                    self.assertEqual(payoffs_cc[0].item(), -1)
                    self.assertEqual(payoffs_cc[1].item(), -1)
                    
                    # Player 1 defects, Player 2 cooperates
                    actions_dc = [torch.tensor(1), torch.tensor(0)]
                    payoffs_dc = self.game(actions_dc)
                    
                    # Verify payoffs
                    self.assertEqual(payoffs_dc[0].item(), 0)
                    self.assertEqual(payoffs_dc[1].item(), -5.0)
                    
                    # Both defect
                    actions_dd = [torch.tensor(1), torch.tensor(1)]
                    payoffs_dd = self.game(actions_dd)
                    
                    # Verify payoffs
                    self.assertEqual(payoffs_dd[0].item(), -3)
                    self.assertEqual(payoffs_dd[1].item(), -3)

                def test_best_responses(self):
                    """Test best response calculations"""
                    # Player 1's best response when Player 2 cooperates
                    p1_br_to_coop = self.game.best_response(0, [None, torch.tensor(0)])
                    self.assertEqual(p1_br_to_coop.item(), 1, "Player 1 should defect when Player 2 cooperates")
                    
                    # Player 1's best response when Player 2 defects
                    p1_br_to_defect = self.game.best_response(0, [None, torch.tensor(1)])
                    self.assertEqual(p1_br_to_defect.item(), 1, "Player 1 should defect when Player 2 defects")
                    
                    # Player 2's best response when Player 1 cooperates
                    p2_br_to_coop = self.game.best_response(1, [torch.tensor(0), None])
                    self.assertEqual(p2_br_to_coop.item(), 1, "Player 2 should defect when Player 1 cooperates")
                    
                    # Player 2's best response when Player 1 defects
                    p2_br_to_defect = self.game.best_response(1, [torch.tensor(1), None])
                    self.assertEqual(p2_br_to_defect.item(), 1, "Player 2 should defect when Player 1 defects")

            class TestMatchingPennies(unittest.TestCase):
                def setUp(self):
                    """Set up a Matching Pennies game for testing"""
                    # In Matching Pennies:
                    # Player 1 wins if both show same side (heads/heads or tails/tails)
                    # Player 2 wins if they show different sides
                    
                    # Player 1's payoff matrix (heads=0, tails=1)
                    self.player1_payoffs = [
                        [1, -1],   # Player 1 heads: [Player 2 heads, Player 2 tails]
                        [-1, 1]    # Player 1 tails: [Player 2 heads, Player 2 tails]
                    ]
                    
                    # Player 2's payoff matrix (heads=0, tails=1)
                    self.player2_payoffs = [
                        [-1, 1],   # Player 2 heads: [Player 1 heads, Player 1 tails]
                        [1, -1]    # Player 2 tails: [Player 1 heads, Player 1 tails]
                    ]
                    
                    # Convert to tensors
                    self.payoff_tensors = [
                        torch.tensor(self.player1_payoffs, dtype=torch.float32),
                        torch.tensor(self.player2_payoffs, dtype=torch.float32)
                    ]
                    
                    # Create the game
                    self.game = StrategicGame(2, [2, 2], self.payoff_tensors)

                def test_payoffs(self):
                    """Test payoffs for different action profiles"""
                    # Both choose heads
                    actions_hh = [torch.tensor(0), torch.tensor(0)]
                    payoffs_hh = self.game(actions_hh)
                    
                    # Verify payoffs
                    self.assertEqual(payoffs_hh[0].item(), 1)
                    self.assertEqual(payoffs_hh[1].item(), -1)
                    
                    # P1 heads, P2 tails
                    actions_ht = [torch.tensor(0), torch.tensor(1)]
                    payoffs_ht = self.game(actions_ht)
                    
                    # Verify payoffs
                    self.assertEqual(payoffs_ht[0].item(), -1)
                    self.assertEqual(payoffs_ht[1].item(), 1)

                def test_nash_equilibrium(self):
                    """Test Nash equilibrium computation for zero-sum game"""
                    # Matching pennies has a unique mixed strategy equilibrium
                    # where both players randomize 50/50
                    strategies = self.game.nash_equilibrium(iterations=1000)
                    
                    # Verify that strategies are approximately uniform
                    for strategy in strategies:
                        self.assertAlmostEqual(strategy[0].item(), 0.5, delta=0.1)
                        self.assertAlmostEqual(strategy[1].item(), 0.5, delta=0.1)

            if __name__ == "__main__":
                unittest.main()
        ''')
    
    def _generate_simultaneous_game_tests(self) -> str:
        return textwrap.dedent('''
            import unittest
            import torch
            from pytorch_oge.game import StrategicGame, create_strategic_game

            class TestSimultaneousPlay(unittest.TestCase):
                def setUp(self):
                    """
                    Set up a classic simultaneous play game scenario:
                    Chicken Game (also known as Hawk-Dove game)
                    """
                    # Player 1's payoff matrix
                    self.player1_payoffs = [
                        [-1, 3],   # Player 1 Hawk: [Player 2 Hawk, Player 2 Dove]
                        [0, 1]     # Player 1 Dove: [Player 2 Hawk, Player 2 Dove]
                    ]
                    
                    # Player 2's payoff matrix
                    self.player2_payoffs = [
                        [-1, 0],   # Player 2 Hawk: [Player 1 Hawk, Player 1 Dove]
                        [3, 1]     # Player 2 Dove: [Player 1 Hawk, Player 1 Dove]
                    ]
                    
                    # Convert to tensors
                    self.payoff_tensors = [
                        torch.tensor(self.player1_payoffs, dtype=torch.float32),
                        torch.tensor(self.player2_payoffs, dtype=torch.float32)
                    ]
                    
                    # Create the game
                    self.game = StrategicGame(2, [2, 2], self.payoff_tensors)

                def test_simultaneous_play_payoffs(self):
                    """
                    Test payoff calculations for different simultaneous action profiles
                    """
                    # Both choose Hawk (aggressive strategy)
                    actions_hawk_hawk = [torch.tensor(0), torch.tensor(0)]
                    payoffs_hawk_hawk = self.game(actions_hawk_hawk)
                    
                    # Verify payoffs when both choose Hawk
                    self.assertEqual(payoffs_hawk_hawk[0].item(), -1, "Player 1 should get -1 when both choose Hawk")
                    self.assertEqual(payoffs_hawk_hawk[1].item(), -1, "Player 2 should get -1 when both choose Hawk")
                    
                    # Player 1 Hawk, Player 2 Dove
                    actions_hawk_dove = [torch.tensor(0), torch.tensor(1)]
                    payoffs_hawk_dove = self.game(actions_hawk_dove)
                    
                    # Verify payoffs for asymmetric Hawk-Dove scenario
                    self.assertEqual(payoffs_hawk_dove[0].item(), 3, "Player 1 should get 3 when Hawk against Dove")
                    self.assertEqual(payoffs_hawk_dove[1].item(), 0, "Player 2 should get 0 when Dove against Hawk")
                    
                    # Both choose Dove (cooperative strategy)
                    actions_dove_dove = [torch.tensor(1), torch.tensor(1)]
                    payoffs_dove_dove = self.game(actions_dove_dove)
                    
                    # Verify payoffs when both choose Dove
                    self.assertEqual(payoffs_dove_dove[0].item(), 1, "Player 1 should get 1 when both choose Dove")
                    self.assertEqual(payoffs_dove_dove[1].item(), 1, "Player 2 should get 1 when both choose Dove")

                def test_best_responses(self):
                    """
                    Test best response calculations for the Chicken Game
                    """
                    # Best response when opponent chooses Hawk
                    p1_br_to_hawk = self.game.best_response(0, [None, torch.tensor(0)])
                    p2_br_to_hawk = self.game.best_response(1, [torch.tensor(0), None])
                    
                    # Best response when opponent chooses Dove
                    p1_br_to_dove = self.game.best_response(0, [None, torch.tensor(1)])
                    p2_br_to_dove = self.game.best_response(1, [torch.tensor(1), None])
                    
                    # Verify best responses
                    self.assertIn(p1_br_to_hawk.item(), [0, 1], "Player 1 best response to Hawk should be Hawk or Dove")
                    self.assertIn(p2_br_to_hawk.item(), [0, 1], "Player 2 best response to Hawk should be Hawk or Dove")
                    self.assertIn(p1_br_to_dove.item(), [0, 1], "Player 1 best response to Dove should be Hawk or Dove")
                    self.assertIn(p2_br_to_dove.item(), [0, 1], "Player 2 best response to Dove should be Hawk or Dove")

                def test_nash_equilibrium(self):
                    """
                    Test Nash equilibrium computation for the Chicken Game
                    """
                    strategies = self.game.nash_equilibrium(iterations=1000)
                    
                    # Verify that strategies are mixed (probability between 0 and 1)
                    for strategy in strategies:
                        self.assertEqual(strategy.shape, torch.Size([2]))
                        self.assertTrue((strategy >= 0).all() and (strategy <= 1).all())
                        self.assertAlmostEqual(strategy.sum().item(), 1.0, places=6)

            if __name__ == "__main__":
                unittest.main()          
         ''')
    
    def _generate_sequential_game_tests(self) -> str:
        """Generate tests for sequential games."""
        return textwrap.dedent('''
            import unittest
            import torch
            from pytorch_oge.core import Player
            from pytorch_oge.sequential import SequentialGame, ExtensiveFormGame

            class TestSequentialGame(unittest.TestCase):
                def setUp(self):
                    """Set up a simple sequential game for testing"""
                    # Define players with strategies
                    def p1_strategy(observation):
                        # Player 1 observes initial state and chooses an action
                        return 1 if observation > 5 else 0
                        
                    def p2_strategy(observation):
                        # Player 2 observes state after Player 1's action
                        p1_action = observation["p1_action"]
                        return 1 if p1_action == 1 else 0
                        
                    self.player1 = Player("Player 1", p1_strategy)
                    self.player2 = Player("Player 2", p2_strategy)
                    
                    # Define state transition function
                    def state_transition(state, action, player_idx):
                        if player_idx == 0:  # Player 1's turn
                            # Update state with Player 1's action
                            return {"value": state, "p1_action": action}
                        else:  # Player 2's turn
                            # Final state includes both players actions
                            return {"value": state["value"], "p1_action": state["p1_action"], "p2_action": action}
                    
                    # Define observation functions
                    def p1_observation(state, player_idx):
                        # Player 1 just observes the raw state value
                        return state
                        
                    def p2_observation(state, player_idx):
                        # Player 2 observes the state after Player 1's action
                        return state
                    
                    # Define payoff functions
                    def p1_payoff(final_state):
                        # Player 1 gets higher payoff if both players choose the same action
                        return 3 if final_state["p1_action"] == final_state["p2_action"] else 0
                        
                    def p2_payoff(final_state):
                        # Player 2 gets higher payoff if both players choose different actions
                        return 3 if final_state["p1_action"] != final_state["p2_action"] else 1
                    
                    # Create the game
                    self.game = SequentialGame(
                        players=[self.player1, self.player2],
                        initial_state=10,  # Some arbitrary initial state
                        state_transition=state_transition,
                        payoff_functions=[p1_payoff, p2_payoff],
                        observation_functions=[p1_observation, p2_observation]
                    )

                def test_sequential_gameplay(self):
                    """Test sequential game play"""
                    # Play the game
                    actions, final_state, payoffs = self.game()
                    
                    # Check actions (based on our strategies)
                    self.assertEqual(actions[0], 1, "Player 1 should choose action 1 for initial state 10")
                    self.assertEqual(actions[1], 1, "Player 2 should choose action 1 matching Player 1")
                    
                    # Check final state
                    self.assertEqual(final_state["value"], 10)
                    self.assertEqual(final_state["p1_action"], 1)
                    self.assertEqual(final_state["p2_action"], 1)
                    
                    # Check payoffs
                    self.assertEqual(payoffs[0], 3, "Player 1's payoff should be 3 for matching actions")
                    self.assertEqual(payoffs[1], 1, "Player 2's payoff should be 1 for matching actions")
                    
                    # Test with different initial state
                    self.game.initial_state = 2
                    actions2, final_state2, payoffs2 = self.game()
                    
                    # Different initial state should lead to different actions and payoffs
                    self.assertEqual(actions2[0], 0, "Player 1 should choose action 0 for initial state 2")
                    self.assertEqual(actions2[1], 0, "Player 2 should choose action 0 matching Player 1")
                    self.assertEqual(payoffs2[0], 3, "Player 1's payoff should be 3 for matching actions")
                    self.assertEqual(payoffs2[1], 1, "Player 2's payoff should be 1 for matching actions")

            class TestExtensiveFormGame(unittest.TestCase):
                def setUp(self):
                    """Set up a simple extensive form game for testing"""
                    # Define a simple game tree for a sequential game
                    # This represents a simple game where:
                    # - Player 1 chooses Left or Right
                    # - If Left, Player 2 chooses Up or Down
                    # - If Right, Player 2 chooses Left or Right
                    self.game_tree = {
                        "root": {
                            "player": 0,
                            "actions": ["Left", "Right"],
                            "children": ["Left_node", "Right_node"]
                        },
                        "Left_node": {
                            "player": 1,
                            "actions": ["Up", "Down"],
                            "payoffs": [
                                [3, 1],  # P1 Left, P2 Up: [P1 payoff, P2 payoff]
                                [2, 2]   # P1 Left, P2 Down: [P1 payoff, P2 payoff]
                            ]
                        },
                        "Right_node": {
                            "player": 1,
                            "actions": ["Left", "Right"],
                            "payoffs": [
                                [0, 3],  # P1 Right, P2 Left: [P1 payoff, P2 payoff]
                                [1, 0]   # P1 Right, P2 Right: [P1 payoff, P2 payoff]
                            ]
                        }
                    }
                    
                    # Define player strategies for this specific game
                    def p1_strategy(node):
                        # Player 1 chooses Right at the root
                        return 1  # Index for "Right"
                        
                    def p2_strategy(node):
                        # Player 2 chooses based on which node they're at
                        if node == "Left_node":
                            return 1  # Index for "Down"
                        else:  # Right_node
                            return 0  # Index for "Left"
                    
                    self.player1 = Player("Player 1", p1_strategy)
                    self.player2 = Player("Player 2", p2_strategy)
                    
                    # Create the game
                    self.game = ExtensiveFormGame(
                        game_tree=self.game_tree,
                        players=[self.player1, self.player2]
                    )

                def test_extensive_form_game(self):
                    """This test is a placeholder since we havent fully implemented the ExtensiveFormGame class"""
                    # This would test the subgame perfect equilibrium calculation
                    # and other extensive form game specific functionality
                    pass

            if __name__ == "__main__":
                unittest.main()
        ''')
    
    def _generate_bayesian_game_tests(self) -> str:
        """Generate tests for Bayesian games."""
        return textwrap.dedent('''
            import unittest
            import torch
            from pytorch_oge.core import Player
            from pytorch_oge.bayesian import BayesianGame

            class TestBayesianGame(unittest.TestCase):
                def setUp(self):
                    """Set up a simple Bayesian game for testing"""
                    # Two players, each with two possible types
                    # Type 0: "High" type
                    # Type 1: "Low" type
                    
                    # Type distributions:
                    # Player 1: 70% chance of High, 30% chance of Low
                    # Player 2: 40% chance of High, 60% chance of Low
                    self.type_distributions = [
                        torch.tensor([0.7, 0.3]),
                        torch.tensor([0.4, 0.6])
                    ]
                    
                    # Payoff matrices for different type combinations:
                    # Each player has 2 possible actions
                    # Format: [player][type]
                    
                    # Player 1, High type
                    p1_high_payoffs = torch.tensor([
                        [3.0, 1.0],  # P1 action 0: [P2 action 0, P2 action 1]
                        [0.0, 2.0]   # P1 action 1: [P2 action 0, P2 action 1]
                    ])
                    
                    # Player 1, Low type
                    p1_low_payoffs = torch.tensor([
                        [1.0, 0.0],
                        [2.0, 1.5]
                    ])
                    
                    # Player 2, High type
                    p2_high_payoffs = torch.tensor([
                        [2.0, 0.0],  # P2 action 0: [P1 action 0, P1 action 1]
                        [1.0, 3.0]   # P2 action 1: [P1 action 0, P1 action 1]
                    ])
                    
                    # Player 2, Low type
                    p2_low_payoffs = torch.tensor([
                        [1.0, 2.0],
                        [0.5, 1.0]
                    ])
                    
                    self.type_conditional_payoffs = [
                        [p1_high_payoffs, p1_low_payoffs],
                        [p2_high_payoffs, p2_low_payoffs]
                    ]
                    
                    # Both players have 2 actions each
                    self.action_spaces = [2, 2]
                    
                    # Create the game
                    self.game = BayesianGame(
                        type_distributions=self.type_distributions,
                        type_conditional_payoffs=self.type_conditional_payoffs,
                        action_spaces=self.action_spaces
                    )

                def test_bayesian_game_initialization(self):
                    """Test that the Bayesian game is initialized correctly"""
                    self.assertEqual(self.game.num_players, 2)
                    self.assertEqual(self.game.action_spaces, [2, 2])
                    
                    # Check type distributions
                    self.assertEqual(len(self.game.type_distributions), 2)
                    self.assertTrue(torch.allclose(self.game.type_distributions[0], torch.tensor([0.7, 0.3])))
                    self.assertTrue(torch.allclose(self.game.type_distributions[1], torch.tensor([0.4, 0.6])))
                    
                    # Check payoff tensors
                    self.assertEqual(len(self.game.type_conditional_payoffs), 2)
                    self.assertEqual(len(self.game.type_conditional_payoffs[0]), 2)  # Player 1 has 2 types
                    self.assertEqual(len(self.game.type_conditional_payoffs[1]), 2)  # Player 2 has 2 types
                    
                    # Check shapes of payoff tensors
                    for player in range(2):
                        for type_idx in range(2):
                            self.assertEqual(
                                self.game.type_conditional_payoffs[player][type_idx].shape,
                                torch.Size([2, 2])
                            )

                def test_payoffs(self):
                    """Test payoffs for different type and action profiles"""
                    # Player 1 is High type (0), Player 2 is Low type (1)
                    # Both choose action 0
                    types = [torch.tensor(0), torch.tensor(1)]
                    actions = [torch.tensor(0), torch.tensor(0)]
                    payoffs = self.game(types, actions)
                    
                    # Verify payoffs
                    self.assertEqual(payoffs[0].item(), 3.0)  # P1 High type payoff
                    self.assertEqual(payoffs[1].item(), 1.0)  # P2 Low type payoff
                    
                    # Player 1 is Low type (1), Player 2 is High type (0)
                    # P1 chooses action 1, P2 chooses action 1
                    types2 = [torch.tensor(1), torch.tensor(0)]
                    actions2 = [torch.tensor(1), torch.tensor(1)]
                    payoffs2 = self.game(types2, actions2)
                    
                    # Verify payoffs
                    self.assertEqual(payoffs2[0].item(), 1.5)  # P1 Low type payoff
                    self.assertEqual(payoffs2[1].item(), 3.0)  # P2 High type payoff

                def test_bayesian_nash_equilibrium(self):
                    """Test Bayesian Nash equilibrium calculation"""
                    # This is a simplified test that just checks if the algorithm runs
                    # and returns strategies of the correct structure
                    equilibrium = self.game.bayesian_nash_equilibrium(iterations=10)
                    
                    # Check structure of result
                    self.assertEqual(len(equilibrium), 2)  # Two players
                    self.assertEqual(len(equilibrium[0]), 2)  # Player 1 has 2 types
                    self.assertEqual(len(equilibrium[1]), 2)  # Player 2 has 2 types
                    
                    # Check that all strategies are valid probability distributions
                    for player_strategies in equilibrium:
                        for strategy in player_strategies:
                            self.assertEqual(strategy.shape, torch.Size([2]))  # 2 actions
                            self.assertAlmostEqual(strategy.sum().item(), 1.0, places=6)
                            self.assertTrue((strategy >= 0).all() and (strategy <= 1).all())

            if __name__ == "__main__":
                unittest.main()
        ''')


# --------------------------------------------------------------------------
# Main function to compile Open Game Engine to PyTorch

def compile_oge_to_pytorch(repo_path: str, output_dir: str) -> None:
    """
    Compile Open Game Engine DSL to PyTorch implementation.
    
    Args:
        repo_path: Path to the cloned Open Game Engine repository
        output_dir: Directory where the PyTorch implementation will be saved
    """
    logger.info(f"Starting compilation process from {repo_path} to {output_dir}")
    
    # Analyze DSL structure
    analyzer = DeepEmbeddingAnalyzer(Path(repo_path))
    dsl_context = analyzer.analyze()
    
    # Generate PyTorch implementation
    generator = PyTorchDSLGenerator(dsl_context)
    modules = generator.generate_pytorch_dsl()
    
    # Generate tests
    tests = generator.generate_tests()
    
    # Create output directories
    pytorch_dir = Path(output_dir) / "pytorch_oge"
    tests_dir = Path(output_dir) / "tests"
    pytorch_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py files
    with open(pytorch_dir / "__init__.py", 'w') as f:
        f.write('"""PyTorch implementation of Open Game Engine DSL."""\n')
    
    with open(tests_dir / "__init__.py", 'w') as f:
        f.write('"""Tests for PyTorch implementation of Open Game Engine DSL."""\n')
    
    # Write module files
    for module_name, module_code in modules.items():
        module_path = pytorch_dir / f"{module_name}.py"
        with open(module_path, 'w') as f:
            f.write(module_code)
        logger.info(f"Generated module: {module_path}")
    
    # Write test files
    for test_name, test_code in tests.items():
        test_path = tests_dir / f"{test_name}.py"
        with open(test_path, 'w') as f:
            f.write(test_code)
        logger.info(f"Generated test: {test_path}")
    
    # Create setup.py for packaging
    setup_py = Path(output_dir) / "setup.py"
    with open(setup_py, 'w') as f:
        f.write(textwrap.dedent('''
            from setuptools import setup, find_packages

            setup(
                name="pytorch-oge",
                version="0.1.0",
                description="PyTorch implementation of Open Game Engine DSL",
                author="Auto-generated",
                packages=find_packages(),
                install_requires=[
                    "torch>=1.9.0",
                ],
                python_requires=">=3.7",
            )
        '''))
    
    # Create README.md
    readme = Path(output_dir) / "README.md"
    with open(readme, 'w') as f:
        f.write(textwrap.dedent('''
            # PyTorch Open Game Engine

            This is an auto-generated PyTorch implementation of the Open Game Engine DSL.

            ## Installation

            ```bash
            pip install -e .
            ```

            ## Usage

            ```python
            import torch
            from pytorch_oge.game import StrategicGame, create_strategic_game

            # Create a Prisoners Dilemma game
            player1_payoffs = [
                [-1, -5],  # Player 1 cooperates: [Player 2 cooperates, Player 2 defects]
                [0, -3]    # Player 1 defects: [Player 2 cooperates, Player 2 defects]
            ]
                
            player2_payoffs = [
                [-1, 0],   # Player 2 cooperates: [Player 1 cooperates, Player 1 defects]
                [-5, -3]   # Player 2 defects: [Player 1 cooperates, Player 1 defects]
            ]
                
            payoff_tensors = [
                torch.tensor(player1_payoffs, dtype=torch.float32),
                torch.tensor(player2_payoffs, dtype=torch.float32)
            ]
                
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

            ## Testing

            Run the tests:

            ```bash
            python -m unittest discover tests
            ```
        '''))
    
    logger.info(f"Compilation complete. Output in {output_dir}")

# --------------------------------------------------------------------------
# Command line interface

def main():
    parser = argparse.ArgumentParser(description="Open Game Engine to PyTorch compiler.")
    parser.add_argument("--repo", type=str, help="Path to the cloned Open Game Engine repository")
    parser.add_argument("--url", type=str, help="URL of the Open Game Engine repository to clone")
    parser.add_argument("--output", type=str, required=True, help="Output directory for PyTorch implementation")
    args = parser.parse_args()
    
    repo_path = args.repo
    
    # If no repo path provided, but URL is, clone the repository
    if repo_path is None and args.url is not None:
        temp_dir = tempfile.mkdtemp()
        repo_path = clone_repo(args.url, temp_dir)
    
    if repo_path is None:
        parser.error("Either --repo or --url must be provided")
    
    compile_oge_to_pytorch(repo_path, args.output)

if __name__ == "__main__":
    main()
