import tetris
import gymnasium as gym
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import time
from tqdm import tqdm  # Import tqdm for progress bars
import os
import json
from datetime import datetime

# Define the Tetris Gymnasium Environment
class TetrisEnv(gym.Env):
    """Tetris environment following the gym interface"""
    
    def __init__(self, width=10, height=20, buffer_zone=2):
        super().__init__()
        
        # Initialize the Tetris engine
        self.tetris_engine = tetris.TetrisEngine(width, height, buffer_zone)
        self.width = width
        self.height = height
        self.buffer_zone = buffer_zone
        
        # Define action space: left, right, rotate left, rotate right, soft drop, hard drop, hold
        self.action_space = gym.spaces.Discrete(7)
        
        # Define observation space: The board plus some additional features
        self.observation_space = gym.spaces.Dict({
            'board': gym.spaces.Box(low=0, high=8, shape=(height, width), dtype=np.int32),
            'current_piece': gym.spaces.Discrete(7),  # 7 different tetrominos
            'next_piece': gym.spaces.Discrete(7),
            'held_piece': gym.spaces.Discrete(8),  # 7 pieces + 1 for no held piece
        })
        
        # Mapping from piece name to index
        self.piece_to_idx = {
            'I': 0, 'O': 1, 'T': 2, 'S': 3, 'Z': 4, 'J': 5, 'L': 6, None: 7
        }
        self.idx_to_piece = {v: k for k, v in self.piece_to_idx.items() if k is not None}
        
        # Initialize metrics
        self.total_lines_cleared = 0
        self.total_garbage_sent = 0
        self.max_height = 0
        self.steps_survived = 0
        
    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state"""
        super().reset(seed=seed)
        self.tetris_engine = tetris.TetrisEngine(self.width, self.height, self.buffer_zone)
        
        self.total_lines_cleared = 0
        self.total_garbage_sent = 0
        self.max_height = 0
        self.steps_survived = 0
        
        state = self._get_obs()
        info = self._get_info()
        
        return state, info
    
    def step(self, action):
        """Execute an action in the environment"""
        self.steps_survived += 1
        
        # Map actions to Tetris engine commands
        # 0: left, 1: right, 2: rotate left, 3: rotate right, 4: soft drop, 5: hard drop, 6: hold
        moved = False
        
        if action == 0:  # Left
            moved = self.tetris_engine.move_left()
        elif action == 1:  # Right
            moved = self.tetris_engine.move_right()
        elif action == 2:  # Rotate Left
            moved = self.tetris_engine.rotate_left()
        elif action == 3:  # Rotate Right
            moved = self.tetris_engine.rotate_right()
        elif action == 4:  # Soft Drop
            moved = self.tetris_engine.soft_drop()
        elif action == 5:  # Hard Drop
            self.tetris_engine.hard_drop()
            moved = True
        elif action == 6:  # Hold
            moved = self.tetris_engine.hold()
        
        # Get current state
        state = self._get_obs()
        
        # Check if game is over
        terminated = self.tetris_engine.game_over
        
        # Calculate reward
        reward = self._calculate_reward(moved)
        
        # Additional info
        info = self._get_info()
        
        return state, reward, terminated, False, info
    
    def _calculate_reward(self, moved):
        """Calculate rewards based on game state changes"""
        game_state = self.tetris_engine.get_state()
        
        # Start with a small negative reward for each step (encourages efficiency)
        reward = -0.01
        
        # Add reward for lines cleared
        lines_cleared = game_state['lines'] - self.total_lines_cleared
        if lines_cleared > 0:
            # Reward scales quadratically with lines cleared (encourages combos)
            reward += lines_cleared ** 2 * 1.0
            self.total_lines_cleared = game_state['lines']
        
        # Add reward for garbage sent
        garbage_sent = game_state['garbage_sent'] - self.total_garbage_sent
        if garbage_sent > 0:
            reward += garbage_sent * 0.5
            self.total_garbage_sent = game_state['garbage_sent']
            
        # Add reward for combos
        if game_state['combo'] > 0:
            reward += min(game_state['combo'] * 0.5, 5.0)
            
        # Add reward for back-to-back
        if game_state['back_to_back']:
            reward += 1.0
            
        # Penalize invalid moves
        if not moved:
            reward -= 0.1
            
        # Heavy penalty for game over
        if self.tetris_engine.game_over:
            reward -= 10.0
            
        # Calculate board metrics for additional rewards
        board_metrics = self._calculate_board_metrics(game_state['board'])
        
        # Penalize high stacks (encourages keeping the board low)
        reward -= board_metrics['max_height'] * 0.02
        
        # Penalize holes and bumpiness
        reward -= board_metrics['holes'] * 0.1
        reward -= board_metrics['bumpiness'] * 0.01
        
        # Reward for maintaining a clear playing field
        reward += (1.0 - board_metrics['avg_height'] / self.height) * 0.05
        
        return reward
    
    def _calculate_board_metrics(self, board):
        """Calculate various metrics about the board state"""
        heights = np.zeros(self.width, dtype=int)
        for col in range(self.width):
            for row in range(self.height):
                if board[row, col] > 0:
                    heights[col] = self.height - row
                    break
                    
        # Calculate maximum height
        max_height = np.max(heights) if heights.size > 0 else 0
        self.max_height = max(self.max_height, max_height)
        
        # Calculate average height
        avg_height = np.mean(heights)
        
        # Calculate "bumpiness" (sum of differences between adjacent columns)
        bumpiness = np.sum(np.abs(np.diff(heights)))
        
        # Calculate number of holes (empty spaces with filled cells above them)
        holes = 0
        for col in range(self.width):
            found_block = False
            for row in range(self.height):
                if board[row, col] > 0:
                    found_block = True
                elif found_block and board[row, col] == 0:
                    holes += 1
                    
        return {
            'max_height': max_height,
            'avg_height': avg_height,
            'bumpiness': bumpiness,
            'holes': holes
        }
    
    def _get_obs(self):
        """Get the current observation from the engine"""
        state = self.tetris_engine.get_state()
        
        # Convert piece names to indices
        current_piece_idx = self.piece_to_idx[self.tetris_engine.current_piece_name]
        next_piece_idx = self.piece_to_idx[state['next_piece']]
        held_piece_idx = self.piece_to_idx[state['held_piece']]
        
        return {
            'board': state['board'].astype(np.int32),
            'current_piece': current_piece_idx,
            'next_piece': next_piece_idx,
            'held_piece': held_piece_idx
        }
    
    def _get_info(self):
        """Get additional info from the engine"""
        state = self.tetris_engine.get_state()
        
        return {
            'lines_cleared': self.total_lines_cleared,
            'garbage_sent': self.total_garbage_sent,
            'level': state['level'],
            'max_height': self.max_height,
            'steps_survived': self.steps_survived,
            'combo': state['combo'],
            'back_to_back': state['back_to_back'],
        }
    
    def render(self):
        """Render the current board state (useful for debugging)"""
        board = self.tetris_engine.get_render_board()
        
        # Simple ASCII rendering
        print("-" * (self.width * 2 + 2))
        for row in board:
            print("|", end="")
            for cell in row:
                if cell == 0:
                    print("  ", end="")
                else:
                    print("██", end="")
            print("|")
        print("-" * (self.width * 2 + 2))
        print(f"Lines: {self.total_lines_cleared}, Level: {self.tetris_engine.level}")


# Define the Value Network (Critic part of A2C/PPO)
class TetrisValueNetwork:
    """Neural network for evaluating Tetris board states"""
    
    def __init__(self, input_shape=(20, 10, 1), piece_embedding_dim=8):
        self.input_shape = input_shape
        self.piece_embedding_dim = piece_embedding_dim
        self.model = self._build_model()
        
    def _build_model(self):
        # Input for the board state
        board_input = keras.Input(shape=self.input_shape, name="board")
        
        # Input for piece information
        current_piece_input = keras.Input(shape=(1,), name="current_piece")
        next_piece_input = keras.Input(shape=(1,), name="next_piece")
        held_piece_input = keras.Input(shape=(1,), name="held_piece")
        
        # Process board with convolutional layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(board_input)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.Flatten()(x)
        
        # Embed piece information
        piece_embedding = layers.Embedding(
            input_dim=8,  # 7 pieces + None
            output_dim=self.piece_embedding_dim
        )
        
        current_piece_embedded = piece_embedding(current_piece_input)
        next_piece_embedded = piece_embedding(next_piece_input)
        held_piece_embedded = piece_embedding(held_piece_input)
        
        # Flatten piece embeddings
        current_piece_embedded = layers.Flatten()(current_piece_embedded)
        next_piece_embedded = layers.Flatten()(next_piece_embedded)
        held_piece_embedded = layers.Flatten()(held_piece_embedded)
        
        # Concatenate all features
        concat = layers.Concatenate()([
            x,
            current_piece_embedded,
            next_piece_embedded,
            held_piece_embedded
        ])
        
        # Process combined features
        concat = layers.Dense(256, activation='relu')(concat)
        concat = layers.Dense(128, activation='relu')(concat)
        
        # Output value (state value estimation)
        value_output = layers.Dense(1, name="value")(concat)
        
        # Create model
        model = keras.Model(
            inputs=[board_input, current_piece_input, next_piece_input, held_piece_input],
            outputs=value_output
        )
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        
        return model
    
    def predict(self, state):
        """Predict the value of a state"""
        # Prepare the inputs
        board = np.expand_dims(state['board'], axis=-1)  # Add channel dimension
        current_piece = np.array([state['current_piece']])
        next_piece = np.array([state['next_piece']])
        held_piece = np.array([state['held_piece']])
        
        # Make prediction
        return self.model.predict(
            [np.expand_dims(board, axis=0), current_piece, next_piece, held_piece],
            verbose=0
        )[0, 0]
    
    def train(self, states, target_values, batch_size=32, epochs=1):
        """Train the value network"""
        boards = np.stack([np.expand_dims(s['board'], axis=-1) for s in states])
        current_pieces = np.array([s['current_piece'] for s in states])
        next_pieces = np.array([s['next_piece'] for s in states])
        held_pieces = np.array([s['held_piece'] for s in states])
        
        self.model.fit(
            [boards, current_pieces, next_pieces, held_pieces],
            target_values,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0
        )
        
    def save(self, path):
        """Save the model to disk"""
        self.model.save(path)
        
    def load(self, path):
        """Load the model from disk"""
        self.model = keras.models.load_model(path)


# Experience Replay Buffer
class ReplayBuffer:
    """Store and sample experiences for training"""
    
    def __init__(self, maxlen=10000):
        self.buffer = deque(maxlen=maxlen)
        
    def add(self, state, action, reward, next_state, done):
        """Add an experience to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
            
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


# PPO-style trainer class for learning the value function
class TetrisValueTrainer:
    """Train a value function using PPO-style updates"""
    
    def __init__(self, env, value_network, gamma=0.99, buffer_size=10000,
                 batch_size=64, training_epochs=4, entropy_coef=0.01):
        self.env = env
        self.value_network = value_network
        self.gamma = gamma
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.entropy_coef = entropy_coef
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_values = []
        
    def collect_experience(self, num_steps):
        """Collect experience by interacting with the environment"""
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_values = []
        
        # Initialize episode statistics
        episode_stats = {
            'lines_cleared': 0,
            'garbage_sent': 0,
            'combos': 0,
            'max_combo': 0,
            'back_to_back': 0,
            'tetris_count': 0,
            'tspin_count': 0,
            'perfect_clears': 0,
            'max_height': 0,
            'blocks_placed': {
                'I': 0, 'O': 0, 'T': 0, 'S': 0, 'Z': 0, 'J': 0, 'L': 0
            }
        }
        
        # Store last move stats for tracking T-spins and line clears
        last_move_stats = None
        
        # Add progress bar for experience collection
        for step in tqdm(range(num_steps), desc="Collecting experience", leave=False):
            # Get state value
            value = self.value_network.predict(state)
            episode_values.append(value)
            
            # Sample action based on a simple strategy (to be improved with search later)
            action = self._select_action(state)
            
            # Take action in environment
            next_state, reward, done, _, info = self.env.step(action)
            
            # Store experience
            self.buffer.add(state, action, reward, next_state, done)
            
            # Update tracking metrics
            episode_reward += reward
            episode_length += 1
            
            # Update episode statistics
            # Track lines cleared in this step
            new_lines = info['lines_cleared'] - episode_stats['lines_cleared']
            if new_lines > 0:
                episode_stats['lines_cleared'] = info['lines_cleared']
                if new_lines == 4:
                    episode_stats['tetris_count'] += 1
                    
            # Track garbage sent
            new_garbage = info['garbage_sent'] - episode_stats['garbage_sent']
            if new_garbage > 0:
                episode_stats['garbage_sent'] = info['garbage_sent']
            
            # Track combos
            if info['combo'] > 0:
                episode_stats['combos'] += 1
                episode_stats['max_combo'] = max(episode_stats['max_combo'], info['combo'])
                
            # Track back-to-back
            if info['back_to_back']:
                episode_stats['back_to_back'] += 1
                
            # Track max height
            episode_stats['max_height'] = max(episode_stats['max_height'], info['max_height'])
                
            # Track T-spins and perfect clears through last move stats
            current_last_move = self.env.tetris_engine.get_last_move_stats()
            
            if last_move_stats != current_last_move:
                last_move_stats = current_last_move
                
                # Check for T-spins
                if 'T-Spin' in current_last_move['clear_type']:
                    episode_stats['tspin_count'] += 1
                    
                # Check for perfect clears
                if current_last_move['is_perfect_clear']:
                    episode_stats['perfect_clears'] += 1
                    
                # Track blocks placed
                if current_last_move['piece'] and current_last_move['piece'] in episode_stats['blocks_placed']:
                    episode_stats['blocks_placed'][current_last_move['piece']] += 1
            
            # Move to next state
            state = next_state
            
            if done:
                # Reset environment
                state, _ = self.env.reset()
                
                # Record metrics
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_values.append(np.mean(episode_values) if episode_values else 0)
                
                # Print episode statistics
                print(f"\nEpisode Completed:")
                print(f"  Lines cleared: {episode_stats['lines_cleared']}")
                print(f"  Tetris count: {episode_stats['tetris_count']}")
                print(f"  T-spin count: {episode_stats['tspin_count']}")
                print(f"  Garbage sent: {episode_stats['garbage_sent']}")
                print(f"  Max combo: {episode_stats['max_combo']}")
                print(f"  Back-to-back count: {episode_stats['back_to_back']}")
                print(f"  Perfect clears: {episode_stats['perfect_clears']}")
                print(f"  Max height: {episode_stats['max_height']}")
                print(f"  Blocks placed:")
                total_blocks = sum(episode_stats['blocks_placed'].values())
                for piece, count in episode_stats['blocks_placed'].items():
                    percentage = (count / total_blocks * 100) if total_blocks > 0 else 0
                    print(f"    {piece}: {count} ({percentage:.1f}%)")
                print(f"  Total blocks: {total_blocks}")
                print(f"  Episode length: {episode_length}")
                print(f"  Total reward: {episode_reward:.2f}")
                print(f"  Average value: {np.mean(episode_values):.4f}")
                
                # Reset episode metrics
                episode_reward = 0
                episode_length = 0
                episode_values = []
                
                # Reset episode statistics
                episode_stats = {
                    'lines_cleared': 0,
                    'garbage_sent': 0,
                    'combos': 0,
                    'max_combo': 0,
                    'back_to_back': 0,
                    'tetris_count': 0,
                    'tspin_count': 0,
                    'perfect_clears': 0,
                    'max_height': 0,
                    'blocks_placed': {
                        'I': 0, 'O': 0, 'T': 0, 'S': 0, 'Z': 0, 'J': 0, 'L': 0
                    }
                }
        
    def train_value_function(self):
        """Train the value function using collected experiences"""
        if len(self.buffer) < self.batch_size:
            return
            
        # Sample experiences
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Compute target values (TD learning)
        target_values = []
        for i in tqdm(range(len(states)), desc="Computing target values", leave=False):
            if dones[i]:
                target_values.append(rewards[i])
            else:
                next_value = self.value_network.predict(next_states[i])
                target_values.append(rewards[i] + self.gamma * next_value)
                
        # Train value network
        self.value_network.train(states, np.array(target_values), 
                                 batch_size=self.batch_size,
                                 epochs=self.training_epochs)
    
    def _select_action(self, state):
        """Simple action selection strategy for collecting experience"""
        # For initial training, we'll use a mix of random actions and a simple heuristic
        # This will be replaced by a more sophisticated search using the value function
        
        # With 20% probability, take random action
        if random.random() < 0.2:
            return random.randint(0, self.env.action_space.n - 1)
            
        # Otherwise, prefer hard drop and moves that lead to line clears
        # We'll prioritize hard drop (5) and soft drop (4) with higher probability
        action_probs = [0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.1]  # Simple distribution
        return np.random.choice(self.env.action_space.n, p=action_probs)


# Search-based Tetris Bot using the learned value function
class TetrisSearchBot:
    """Bot that uses the value network to evaluate states and search for best moves"""
    
    def __init__(self, env, value_network, search_depth=3, num_simulations=10):
        self.env = env
        self.value_network = value_network
        self.search_depth = search_depth
        self.num_simulations = num_simulations
        
    def select_action(self, state):
        """Select the best action using a search algorithm"""
        best_action = None
        best_value = float('-inf')
        
        # Make a copy of the environment for search
        env_copy = TetrisEnv(self.env.width, self.env.height, self.env.buffer_zone)
        env_copy.tetris_engine = self.env.tetris_engine  # This needs a deep copy in production
        
        # For each possible action
        for action in range(self.env.action_space.n):
            # Reset search environment to current state
            env_copy.tetris_engine = self.env.tetris_engine  # In production, use deep copy
            
            # Take action and evaluate resulting state
            _, reward, done, _, _ = env_copy.step(action)
            
            if done:
                # If the action leads to game over, avoid it
                continue
                
            # If action leads to hard drop (piece placement), evaluate the resulting board
            if action == 5:  # Hard drop
                # Evaluate the state deeply
                value = self._evaluate_state(env_copy.tetris_engine.get_state())
                
                # Account for immediate reward
                value += reward
                
                # Update best action
                if value > best_value:
                    best_value = value
                    best_action = action
            else:
                # For other actions, look ahead a few steps
                value = self._rollout(env_copy, self.search_depth)
                
                # Update best action
                if value > best_value:
                    best_value = value
                    best_action = action
        
        # If no good action found, default to hard drop
        if best_action is None:
            best_action = 5  # Hard drop
            
        return best_action
    
    def _rollout(self, env, depth):
        """Perform a rollout from the current state"""
        total_reward = 0
        discount = 1.0
        
        # Make a copy of the environment for rollout
        env_copy = TetrisEnv(env.width, env.height, env.buffer_zone)
        env_copy.tetris_engine = env.tetris_engine  # In production, use deep copy
        
        for _ in range(depth):
            # Take a random action
            action = random.randint(0, env_copy.action_space.n - 1)
            state, reward, done, _, _ = env_copy.step(action)
            
            # Accumulate discounted reward
            total_reward += discount * reward
            discount *= self.value_network.gamma
            
            if done or action == 5:  # Game over or hard drop
                break
                
        # Add final state evaluation
        if not done:
            state = env_copy._get_obs()
            value = self.value_network.predict(state)
            total_reward += discount * value
            
        return total_reward
    
    def _evaluate_state(self, state):
        """Evaluate a state using the value network and additional heuristics"""
        # Convert state to format expected by value network
        value_network_state = {
            'board': state['board'],
            'current_piece': self.env.piece_to_idx[self.env.tetris_engine.current_piece_name],
            'next_piece': self.env.piece_to_idx[state['next_piece']],
            'held_piece': self.env.piece_to_idx[state['held_piece']]
        }
        
        # Get base evaluation from value network
        value = self.value_network.predict(value_network_state)
        
        # Add heuristics for additional evaluation
        # Calculate board metrics
        heights = np.zeros(self.env.width, dtype=int)
        for col in range(self.env.width):
            for row in range(self.env.height):
                if state['board'][row, col] > 0:
                    heights[col] = self.env.height - row
                    break
                    
        # Penalize high stacks
        max_height = np.max(heights) if heights.size > 0 else 0
        value -= max_height * 0.05
        
        # Penalize bumpiness
        bumpiness = np.sum(np.abs(np.diff(heights)))
        value -= bumpiness * 0.01
        
        # Reward low average height
        avg_height = np.mean(heights) if heights.size > 0 else 0
        value += (1.0 - avg_height / self.env.height) * 0.05
        
        return value


# Checkpoint Manager for saving and loading training state
class CheckpointManager:
    """Handles saving and loading of training checkpoints"""
    
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, value_network, trainer, episode, metrics=None):
        """Save model weights and training metrics"""
        # Create timestamp for this checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create checkpoint directory without .keras extension
        checkpoint_dir = os.path.join(self.checkpoint_dir, f"checkpoint_ep{episode}_{timestamp}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model weights with proper .keras extension
        model_path = os.path.join(checkpoint_dir, "model.keras")
        value_network.save(model_path)
        
        # Save training metrics
        metrics_data = {
            "episode": episode,
            "timestamp": timestamp,
            "episode_rewards": trainer.episode_rewards,
            "episode_lengths": trainer.episode_lengths,
            "episode_values": trainer.episode_values
        }
        
        # Add any additional metrics
        if metrics:
            metrics_data.update(metrics)
            
        metrics_path = os.path.join(checkpoint_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            for key, value in metrics_data.items():
                if isinstance(value, (np.ndarray, list)):
                    metrics_data[key] = list(map(float, value))
            json.dump(metrics_data, f, indent=2)
            
        print(f"Saved checkpoint at episode {episode} to {checkpoint_dir}")
        return checkpoint_dir
        
    def load_latest_checkpoint(self, value_network=None, trainer=None):
        """Load the most recent checkpoint"""
        checkpoints = self._get_checkpoints()
        
        if not checkpoints:
            print("No checkpoints found.")
            return None, None, 0
            
        # Get the latest checkpoint
        latest_checkpoint = checkpoints[-1]
        episode = latest_checkpoint["episode"]
        
        print(f"Loading checkpoint from episode {episode}...")
        
        # Create a new value network if one wasn't provided
        if value_network is None:
            value_network = TetrisValueNetwork()
        
        # Load model weights
        model_path = os.path.join(latest_checkpoint["path"], "model")
        value_network.load(model_path)
        
        # Load metrics if trainer was provided
        if trainer is not None:
            metrics_path = os.path.join(latest_checkpoint["path"], "metrics.json")
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                trainer.episode_rewards = metrics["episode_rewards"]
                trainer.episode_lengths = metrics["episode_lengths"] 
                trainer.episode_values = metrics["episode_values"]
                
        return value_network, trainer, episode
    
    def load_checkpoint_by_episode(self, episode, value_network=None):
        """Load a specific checkpoint by episode number"""
        checkpoints = self._get_checkpoints()
        
        # Find the checkpoint with the specified episode number
        checkpoint = next((c for c in checkpoints if c["episode"] == episode), None)
        
        if not checkpoint:
            print(f"No checkpoint found for episode {episode}.")
            return None
            
        print(f"Loading checkpoint from episode {episode}...")
        
        # Create a new value network if one wasn't provided
        if value_network is None:
            value_network = TetrisValueNetwork()
            
        # Load model weights
        model_path = os.path.join(checkpoint["path"], "model")
        value_network.load(model_path)
        
        return value_network
    
    def list_checkpoints(self):
        """List all available checkpoints"""
        checkpoints = self._get_checkpoints()
        
        if not checkpoints:
            print("No checkpoints found.")
            return
            
        print(f"{'Episode':<10} {'Timestamp':<20} {'Path':<50}")
        print("-" * 80)
        
        for ckpt in checkpoints:
            print(f"{ckpt['episode']:<10} {ckpt['timestamp']:<20} {ckpt['path']:<50}")
    
    def _get_checkpoints(self):
        """Get a list of all available checkpoints"""
        if not os.path.exists(self.checkpoint_dir):
            return []
            
        checkpoints = []
        
        for item in os.listdir(self.checkpoint_dir):
            item_path = os.path.join(self.checkpoint_dir, item)
            
            if os.path.isdir(item_path) and item.startswith("checkpoint_ep"):
                # Extract episode number from directory name
                try:
                    # Format: checkpoint_ep{episode}_{timestamp}
                    parts = item.split("_")
                    episode = int(parts[1].replace("ep", ""))
                    timestamp = parts[2]
                    
                    checkpoints.append({
                        "episode": episode,
                        "timestamp": timestamp,
                        "path": item_path
                    })
                except (IndexError, ValueError):
                    # Skip directories with unexpected naming format
                    continue
        
        # Sort checkpoints by episode number
        checkpoints.sort(key=lambda x: x["episode"])
        return checkpoints


# Training procedure
def train_tetris_value_network(episodes=100, steps_per_episode=1000, 
                               eval_interval=10, checkpoint_interval=10,
                               model_path="tetris_value_network", 
                               resume_training=False):
    """Train a value network for Tetris state evaluation"""
    # Create environment
    env = TetrisEnv()
    
    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager()
    
    # Create value network and trainer
    value_network = TetrisValueNetwork(input_shape=(env.height, env.width, 1))
    trainer = TetrisValueTrainer(env, value_network)
    
    # Resume training from checkpoint if requested
    start_episode = 1
    if resume_training:
        value_network, trainer, loaded_episode = checkpoint_manager.load_latest_checkpoint(
            value_network, trainer
        )
        if loaded_episode > 0:
            start_episode = loaded_episode + 1
            print(f"Resuming training from episode {start_episode}")
    
    # Training loop with progress bar for overall episodes
    for episode in tqdm(range(start_episode, episodes + 1), desc="Training Episodes", total=episodes):
        start_time = time.time()
        
        # Collect experience
        trainer.collect_experience(steps_per_episode)
        
        # Train value function
        trainer.train_value_function()
        
        # Print metrics
        if episode % eval_interval == 0:
            avg_reward = np.mean(trainer.episode_rewards[-eval_interval:])
            avg_length = np.mean(trainer.episode_lengths[-eval_interval:])
            avg_value = np.mean(trainer.episode_values[-eval_interval:])
            
            print(f"\nEpisode {episode}/{episodes}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Episode Length: {avg_length:.2f}")
            print(f"Average State Value: {avg_value:.4f}")
            print(f"Time: {time.time() - start_time:.2f}s")
            print("-" * 40)
            
            # Additional metrics to save with checkpoint
            metrics = {
                "avg_reward": float(avg_reward),
                "avg_length": float(avg_length),
                "avg_value": float(avg_value)
            }
            
            # Save checkpoint at specified intervals
            if episode % checkpoint_interval == 0:
                checkpoint_manager.save_checkpoint(value_network, trainer, episode, metrics)
    
    # Save final model
    value_network.save(model_path)
    
    # Always save a final checkpoint
    if episodes % checkpoint_interval != 0:
        checkpoint_manager.save_checkpoint(
            value_network, trainer, episodes, 
            {"final": True}
        )
    
    return value_network, trainer


# Evaluation procedure
def evaluate_tetris_bot(value_network, num_games=10, render=False):
    """Evaluate the performance of a Tetris bot using the value network"""
    env = TetrisEnv()
    bot = TetrisSearchBot(env, value_network)
    
    game_rewards = []
    game_lengths = []
    lines_cleared = []
    
    for game in range(num_games):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Bot selects action
            action = bot.select_action(state)
            
            # Take action
            state, reward, done, _, info = env.step(action)
            
            # Update metrics
            total_reward += reward
            steps += 1
            
            if render and game == 0:  # Only render the first game
                env.render()
                time.sleep(0.1)  # Slow down rendering
        
        game_rewards.append(total_reward)
        game_lengths.append(steps)
        lines_cleared.append(info['lines_cleared'])
        
        print(f"Game {game + 1}: Reward={total_reward:.2f}, "
              f"Length={steps}, Lines={info['lines_cleared']}")
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Average Reward: {np.mean(game_rewards):.2f}")
    print(f"Average Game Length: {np.mean(game_lengths):.2f}")
    print(f"Average Lines Cleared: {np.mean(lines_cleared):.2f}")
    
    return {
        'rewards': game_rewards,
        'lengths': game_lengths,
        'lines': lines_cleared
    }


# Main execution
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tetris ML - Value Network for State Evaluation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'resume', 'evaluate', 'list-checkpoints'],
                        help='Mode: train from scratch, resume training, evaluate model, or list checkpoints')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes for training')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Steps per episode for training')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='Save checkpoints every N episodes')
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='Print evaluation metrics every N episodes')
    parser.add_argument('--checkpoint', type=int, default=None,
                        help='Specific checkpoint episode to load (for evaluation)')
    parser.add_argument('--games', type=int, default=5,
                        help='Number of games to play during evaluation')
    parser.add_argument('--render', action='store_true',
                        help='Render games during evaluation')
    
    args = parser.parse_args()
    
    print("Tetris ML - Training a Value Network for Tetris State Evaluation")
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager()
    
    # List available checkpoints
    if args.mode == 'list-checkpoints':
        print("\nAvailable Checkpoints:")
        checkpoint_manager.list_checkpoints()
        exit(0)
    
    # Train from scratch
    if args.mode == 'train':
        print(f"\nTraining from scratch for {args.episodes} episodes with {args.steps} steps per episode...")
        value_network, trainer = train_tetris_value_network(
            episodes=args.episodes,
            steps_per_episode=args.steps,
            eval_interval=args.eval_interval,
            checkpoint_interval=args.checkpoint_interval
        )
        
        # Plot training progress
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(trainer.episode_rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        
        plt.subplot(3, 1, 2)
        plt.plot(trainer.episode_lengths)
        plt.title("Episode Lengths")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        
        plt.subplot(3, 1, 3)
        plt.plot(trainer.episode_values)
        plt.title("Average State Values")
        plt.xlabel("Episode")
        plt.ylabel("Value")
        
        plt.tight_layout()
        plt.savefig("training_progress.png")
        print("Training progress plot saved to training_progress.png")
        
    # Resume training from latest checkpoint
    elif args.mode == 'resume':
        print("\nResuming training from latest checkpoint...")
        value_network, trainer = train_tetris_value_network(
            episodes=args.episodes,
            steps_per_episode=args.steps,
            eval_interval=args.eval_interval,
            checkpoint_interval=args.checkpoint_interval,
            resume_training=True
        )
        
        # Plot training progress
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(trainer.episode_rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        
        plt.subplot(3, 1, 2)
        plt.plot(trainer.episode_lengths)
        plt.title("Episode Lengths")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        
        plt.subplot(3, 1, 3)
        plt.plot(trainer.episode_values)
        plt.title("Average State Values")
        plt.xlabel("Episode")
        plt.ylabel("Value")
        
        plt.tight_layout()
        plt.savefig("training_progress_resumed.png")
        print("Training progress plot saved to training_progress_resumed.png")
        
    # Evaluate a trained model
    elif args.mode == 'evaluate':
        # Load model from checkpoint if specified
        if args.checkpoint is not None:
            print(f"\nLoading model from checkpoint at episode {args.checkpoint}...")
            value_network = checkpoint_manager.load_checkpoint_by_episode(args.checkpoint)
        else:
            print("\nLoading latest checkpoint for evaluation...")
            value_network, _, _ = checkpoint_manager.load_latest_checkpoint()
            
        if value_network is None:
            print("Error: Could not load model. Make sure checkpoints exist.")
            exit(1)
            
        print(f"\nEvaluating model on {args.games} games...")
        results = evaluate_tetris_bot(value_network, num_games=args.games, render=args.render)
        
    print("\nProcess complete!")