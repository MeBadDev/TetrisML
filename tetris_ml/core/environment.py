import numpy as np
import gymnasium as gym
import sys
import os

# Add parent directory to path so we can import tetris module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import tetris

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
        
        # Define observation space: The board plus next and held pieces (no current piece)
        self.observation_space = gym.spaces.Dict({
            'board': gym.spaces.Box(low=0, high=8, shape=(height, width), dtype=np.int32),
            'next_piece': gym.spaces.Discrete(7),  # 7 different tetrominos
            'held_piece': gym.spaces.Discrete(8),  # 7 pieces + 1 for no held piece
            'current_piece': gym.spaces.Discrete(8),  # 7 pieces + 1 for no current piece
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
        next_piece_idx = self.piece_to_idx[state['next_piece']]
        held_piece_idx = self.piece_to_idx[state['held_piece']]
        current_piece_idx = self.piece_to_idx[self.tetris_engine.current_piece_name]
        
        return {
            'board': state['board'].astype(np.int32),
            'next_piece': next_piece_idx,
            'held_piece': held_piece_idx,
            'current_piece': current_piece_idx
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