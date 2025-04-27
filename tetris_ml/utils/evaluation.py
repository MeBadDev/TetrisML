import time
import numpy as np

from tetris_ml.core.environment import TetrisEnv
from tetris_ml.agents.search_bot import TetrisSearchBot

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