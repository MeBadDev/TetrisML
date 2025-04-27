import numpy as np
from tqdm import tqdm
import random

from tetris_ml.utils.replay_buffer import ReplayBuffer

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
        
        # Board state evaluation metrics
        self.evaluation_loss = []
        self.board_states_evaluated = 0
        
    def collect_experience(self, num_steps):
        """Collect experience by interacting with the environment"""
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_values = []
        
        # Store board states for pure evaluation training
        board_states = []
        board_values = []
        
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
            # Get the current piece from the environment
            current_piece = self.env.tetris_engine.current_piece_name
            
            # Add current piece to the state dictionary
            if 'current_piece' not in state:
                state['current_piece'] = self.env.piece_to_idx.get(current_piece, 7)  # Use 7 (None) as default
            
            # Get state evaluation score
            value = self.value_network.predict(state)
            episode_values.append(value)
            
            # Store board state and its predicted evaluation score
            board_states.append(state)
            
            # Sample action based on a simple strategy (to be improved with search later)
            action = self._select_action(state)
            
            # Take action in environment
            next_state, reward, done, _, info = self.env.step(action)
            
            # Store experience
            self.buffer.add(state, action, reward, next_state, done)
            
            # Update tracking metrics
            episode_reward += reward
            episode_length += 1
            
            # For hard drops (piece placements), we specially record the resulting board state
            # since these are the most important for evaluation
            if action == 5:  # Hard drop
                # Get the board state after placement and record it for evaluation training
                next_value = 0 if done else self.gamma * self.value_network.predict(next_state)
                board_values.append(reward + next_value)
                self.board_states_evaluated += 1
                
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
                
                # Calculate total blocks placed
                total_blocks = sum(episode_stats['blocks_placed'].values())
                
                # Print episode statistics
                print(f"\nEpisode Completed:")
                print(f"  Model Performance:")
                print(f"  - Average evaluation score: {np.mean(episode_values):.4f}")
                print(f"  - Board evaluation loss: {self.evaluation_loss[-1] if self.evaluation_loss else 'N/A'}")
                print(f"  - Total reward: {episode_reward:.2f}")
                print(f"  - States evaluated: {self.board_states_evaluated}")
                print(f"  - Model confidence: {np.std(episode_values):.4f}")
                
                print(f"  Learning Progress:")
                print(f"  - Episode length: {episode_length}")
                rewards_trend = ""
                if len(self.episode_rewards) > 1:
                    if episode_reward > self.episode_rewards[-2]:
                        rewards_trend = "↑ Improving"
                    elif episode_reward < self.episode_rewards[-2]:
                        rewards_trend = "↓ Declining"
                    else:
                        rewards_trend = "→ Stable"
                print(f"  - Reward trend: {rewards_trend}")
                
                print(f"  Game Statistics:")
                print(f"  - Lines cleared: {episode_stats['lines_cleared']}")
                print(f"  - Tetris rate: {episode_stats['tetris_count'] / max(1, episode_stats['lines_cleared'] / 4):.2f}")
                print(f"  - Garbage efficiency: {episode_stats['garbage_sent'] / max(1, total_blocks):.2f} lines/block")
                print(f"  - Max combo: {episode_stats['max_combo']}")
                
                # Only show block distribution if significant number of blocks placed
                if total_blocks > 5:
                    print(f"  - Block efficiency:")
                    for piece, count in episode_stats['blocks_placed'].items():
                        percentage = (count / total_blocks * 100) if total_blocks > 0 else 0
                        print(f"    {piece}: {count} ({percentage:.1f}%)")
                
                # Advanced metrics
                print(f"  Advanced Metrics:")
                print(f"  - Decision quality: {episode_reward / max(1, episode_length):.4f}")
                print(f"  - Stack height control: {20 - episode_stats['max_height']:.1f}/20")
                print(f"  - Special move rate: {(episode_stats['tspin_count'] + episode_stats['perfect_clears']) / max(1, total_blocks):.3f}")
                print(f"  - Back-to-back efficiency: {episode_stats['back_to_back'] / max(1, episode_stats['tetris_count'] + episode_stats['tspin_count']):.2f} if applicable")
                
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
                
        # Perform additional focused training on board states
        if len(board_states) > 0 and len(board_values) > 0:
            # Train specifically on the board states resulting from piece placements
            loss = self._train_board_evaluation(board_states, board_values)
            self.evaluation_loss.append(loss)
        
    def train_value_function(self):
        """Train the evaluation function using collected experiences"""
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
    
    def _train_board_evaluation(self, states, values):
        """Special training focused on board state evaluation"""
        if len(states) == 0 or len(values) == 0:
            return 0
            
        # Ensure we have the same number of states and values
        min_len = min(len(states), len(values))
        states = states[:min_len]
        values = values[:min_len]
        
        # Train the value network with more epochs for better board evaluation
        loss_history = self.value_network.train(
            states, 
            np.array(values),
            batch_size=min(32, len(states)),
            epochs=5  # More epochs for focused board evaluation training
        )
        
        print(f"Board state evaluation training on {len(states)} states")
        return np.mean(loss_history.history['loss']) if hasattr(loss_history, 'history') else 0
    
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