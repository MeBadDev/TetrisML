import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from tetris_ml.core.environment import TetrisEnv
from tetris_ml.models.value_network import TetrisValueNetwork
from tetris_ml.agents.value_trainer import TetrisValueTrainer
from tetris_ml.utils.checkpoint_manager import CheckpointManager

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

def plot_training_progress(trainer, filename="training_progress.png"):
    """Plot training metrics from a trainer"""
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
    plt.savefig(filename)
    print(f"Training progress plot saved to {filename}")