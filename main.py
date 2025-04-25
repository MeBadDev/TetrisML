import tensorflow as tf
import numpy as np
import gymnasium as gym
from enum import Enum  # Import Python's built-in Enum
import tetris
from tetris.types import MoveKind, PieceType, PlayingStatus  # Import required types
import os
import time
from datetime import datetime
from collections import deque
import random

# Force TensorFlow to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')

# Create directories for saving models and logs
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Parameters
EPISODES = 10000
UPDATE_TARGET_EVERY = 5  # Update target network every N episodes
MODEL_SAVE_FREQ = 100    # Save model every N episodes
LOG_FREQ = 10            # Log stats every N episodes

class Actions(Enum):  # Use Python's built-in Enum instead of gym.Enum
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    ROTATE = 3
    SOFT_DROP = 4
    HARD_DROP = 5
    SWAP = 6

class TetrisEnv(gym.Env):
    def __init__(self, seed=None):
        super(TetrisEnv, self).__init__()
        
        # Initialize the Tetris game with modern rules
        self.game = tetris.BaseGame(seed=seed)
        self.game_over = False
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        
        # Observation space: 10x20 playfield + piece info
        self.observation_space = gym.spaces.Box(
            low=0, high=8,  # 0=empty, 1-7=piece types, 8=ghost
            shape=(22, 10),  # 20 rows + 2 for buffer zone at top
            dtype=np.int32
        )
        
        self.previous_score = 0
        self.previous_lines = 0
        self.combo_count = 0
        self.perfect_clear = False
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.game = tetris.BaseGame(seed=seed)
        else:
            self.game = tetris.BaseGame()
        
        self.game_over = False
        self.previous_score = 0
        self.previous_lines = 0
        self.combo_count = 0
        self.perfect_clear = False
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, {"game_over": True}
        
        # Execute action
        self._execute_action(action)
        
        # Tick the game
        self.game.tick()
        
        # Check if game is over
        if self.game.status == PlayingStatus.STOPPED:
            self.game_over = True
            return self._get_observation(), -10, True, False, {"game_over": True}
        
        # Calculate reward
        reward = self._calculate_reward()
        
        return self._get_observation(), reward, False, False, {}
    
    def _execute_action(self, action_idx):
        action = Actions(action_idx)
        
        if action == Actions.NOOP:
            pass
        elif action == Actions.LEFT:
            self.game.left()
        elif action == Actions.RIGHT:
            self.game.right()
        elif action == Actions.ROTATE:
            self.game.rotate()
        elif action == Actions.SOFT_DROP:
            self.game.soft_drop()
        elif action == Actions.HARD_DROP:
            self.game.hard_drop()
        elif action == Actions.SWAP:
            self.game.swap()
    
    def _get_observation(self):
        # Get the playfield
        playfield = np.array(self.game.get_playfield())
        
        # Ensure output shape is (22, 10) as defined in observation_space
        # If playfield is (20, 10), pad with zeros at the top for buffer zone
        if playfield.shape[0] == 20:
            padded_playfield = np.zeros((22, 10), dtype=np.int32)
            padded_playfield[2:, :] = playfield
            return padded_playfield
        elif playfield.shape == (22, 10):
            return playfield
        else:
            # Resize to match the expected shape
            resized_playfield = np.zeros((22, 10), dtype=np.int32)
            # Copy as much of the playfield as possible
            h, w = min(playfield.shape[0], 22), min(playfield.shape[1], 10)
            resized_playfield[:h, :w] = playfield[:h, :w]
            return resized_playfield
    
    def _calculate_reward(self):
        reward = 0
        
        # Score-based reward
        score_delta = self.game.score - self.previous_score
        self.previous_score = self.game.score
        
        # Basic reward for score increase
        reward += score_delta * 0.01
        
        # Get stats from the game (we need to infer some of these)
        lines_cleared = 0
        t_spin = False
        
        # Check if lines were cleared (simplified)
        if score_delta > 0:
            # Estimate lines cleared based on score
            if score_delta >= 800:  # Tetris (4 lines)
                lines_cleared = 4
                reward += 5  # Bonus for Tetris
            elif score_delta >= 500:  # Triple
                lines_cleared = 3
                reward += 2
            elif score_delta >= 300:  # Double
                lines_cleared = 2
                reward += 1
            elif score_delta >= 100:  # Single
                lines_cleared = 1
                reward += 0.5
            
            # Check for T-Spin (simplified heuristic)
            # A proper T-spin detection would need more game state info
            last_move = self.game.delta
            if last_move and last_move.kind == MoveKind.ROTATE:
                if self.game.piece and self.game.piece.type == PieceType.T:
                    t_spin = True
                    reward += 3  # Bonus for T-spin
        
        # Perfect clear detection (simplified)
        playfield = np.array(self.game.get_playfield())
        empty_count = np.sum(playfield == 0)
        if empty_count == self.game.width * self.game.height and lines_cleared > 0:
            self.perfect_clear = True
            reward += 10  # Big bonus for perfect clear
        
        # Combo tracking
        if lines_cleared > 0:
            self.combo_count += 1
            reward += 0.5 * self.combo_count  # Progressive reward for combos
        else:
            self.combo_count = 0
        
        # Penalize high stack
        stack_heights = []
        playfield = self.game.get_playfield()
        for col in range(self.game.width):
            height = 0
            for row in range(self.game.height-1, -1, -1):
                if playfield[row][col] != 0:
                    height = self.game.height - row
                    break
            stack_heights.append(height)
        
        max_height = max(stack_heights)
        height_penalty = max_height / self.game.height
        reward -= height_penalty * 0.5
        
        # Penalize height differences (holes and wells)
        height_diffs = np.abs(np.diff(stack_heights))
        reward -= np.sum(height_diffs) * 0.02
        
        return reward

    def render(self):
        # Simple rendering for debugging
        playfield = self.game.get_playfield()
        for row in playfield:
            print(" ".join([str(cell) if cell else "." for cell in row]))
        print("\n")

class DQNAgent:
    def __init__(
        self,
        state_shape,
        action_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=64
    ):
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """Neural network model for DQN"""
        model = tf.keras.Sequential([
            # Convolutional layers to extract spatial features
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                                   input_shape=self.state_shape),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model
    
    def update_target_model(self):
        """Update target model to match primary model weights"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose an action based on epsilon-greedy policy"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self):
        """Train the model using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.zeros((self.batch_size,) + self.state_shape)
        next_states = np.zeros((self.batch_size,) + self.state_shape)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
        
        # Predict Q-values for current states and next states
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load model weights"""
        self.model.load_weights(name)
        self.update_target_model()
    
    def save(self, name):
        """Save model weights"""
        self.model.save_weights(name)

def main():
    # Initialize environment and agent
    env = TetrisEnv(seed=42)
    state_shape = (22, 10, 1)  # Add channel dimension
    action_size = 7  # NOOP, LEFT, RIGHT, ROTATE, SOFT_DROP, HARD_DROP, SWAP
    
    agent = DQNAgent(
        state_shape=state_shape,
        action_size=action_size,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.9995,
        memory_size=100000,
        batch_size=32
    )
    
    # Create TensorBoard logs
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/tetris_{current_time}"
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    # Training variables
    scores = []
    avg_scores = []
    
    for episode in range(1, EPISODES + 1):
        state, _ = env.reset()
        state = np.expand_dims(state, axis=-1)  # Add channel dimension
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Choose and execute action
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=-1)  # Add channel dimension
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            
            # Train agent
            agent.replay()
            
            total_reward += reward
            steps += 1
            
            # Limit max steps per episode
            if steps > 10000:  # Safeguard to prevent infinite loops
                break
        
        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])  # Moving average of last 100 episodes
        avg_scores.append(avg_score)
        
        # Update target network
        if episode % UPDATE_TARGET_EVERY == 0:
            agent.update_target_model()
        
        # Save model
        if episode % MODEL_SAVE_FREQ == 0:
            agent.save(f"models/tetris_dqn_{episode}.h5")
        
        # Log stats
        if episode % LOG_FREQ == 0:
            print(f"Episode: {episode}/{EPISODES}, Score: {total_reward:.2f}, " +
                  f"Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.4f}, " +
                  f"Steps: {steps}")
            
            # TensorBoard logging
            with summary_writer.as_default():
                tf.summary.scalar('episode_reward', total_reward, step=episode)
                tf.summary.scalar('avg_reward_100', avg_score, step=episode)
                tf.summary.scalar('epsilon', agent.epsilon, step=episode)
                tf.summary.scalar('steps_per_episode', steps, step=episode)
    
    # Save final model
    agent.save("models/tetris_dqn_final.h5")

if __name__ == "__main__":
    main()

