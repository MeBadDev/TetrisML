import random
import numpy as np

from tetris_ml.core.environment import TetrisEnv

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