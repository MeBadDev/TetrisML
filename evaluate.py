import numpy as np
import time
import tensorflow as tf
import os
from main import TetrisEnv, DQNAgent

def evaluate_agent(model_path, num_games=10, render=False, delay=0):
    """
    Evaluate a trained agent on multiple games
    
    Args:
        model_path: Path to the saved model
        num_games: Number of games to evaluate
        render: Whether to render the game
        delay: Time delay between steps for visualization
    """
    env = TetrisEnv()
    state_shape = (22, 10, 1)
    action_size = 7
    
    agent = DQNAgent(
        state_shape=state_shape,
        action_size=action_size
    )
    
    # Load trained model
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration during evaluation
    
    scores = []
    lines_cleared = []
    steps_taken = []
    
    for game in range(num_games):
        state, _ = env.reset()
        state = np.expand_dims(state, axis=-1)  # Add channel dimension
        
        done = False
        total_reward = 0
        game_lines = env.game.lines
        steps = 0
        
        while not done:
            if render:
                env.render()
                time.sleep(delay)
            
            action = agent.act(state, training=False)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=-1)  # Add channel dimension
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if steps > 10000:  # Prevent infinite loops
                break
        
        # Calculate lines cleared in this game
        final_lines = env.game.lines
        lines_in_game = final_lines - game_lines
        
        scores.append(total_reward)
        lines_cleared.append(lines_in_game)
        steps_taken.append(steps)
        
        print(f"Game {game+1}/{num_games}: Score = {total_reward:.2f}, Lines = {lines_in_game}, Steps = {steps}")
    
    print("\nEvaluation Results:")
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Average Lines Cleared: {np.mean(lines_cleared):.2f} ± {np.std(lines_cleared):.2f}")
    print(f"Average Steps: {np.mean(steps_taken):.2f} ± {np.std(steps_taken):.2f}")
    
    return scores, lines_cleared, steps_taken

if __name__ == "__main__":
    # Check if we have trained models
    if not os.path.exists("models"):
        print("No trained models found. Please run main.py first to train a model.")
        exit()
    
    # Try to find the final model
    if os.path.exists("models/tetris_dqn_final.h5"):
        model_path = "models/tetris_dqn_final.h5"
    else:
        # Find the latest model
        model_files = [f for f in os.listdir("models") if f.startswith("tetris_dqn_") and f.endswith(".h5")]
        if not model_files:
            print("No trained models found. Please run main.py first to train a model.")
            exit()
        
        # Get model with highest episode number
        highest_ep = max([int(f.split("_")[-1].split(".")[0]) for f in model_files])
        model_path = f"models/tetris_dqn_{highest_ep}.h5"
    
    print(f"Evaluating model: {model_path}")
    evaluate_agent(
        model_path=model_path,
        num_games=5,
        render=True,
        delay=0.1
    )