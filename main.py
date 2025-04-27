#!/usr/bin/env python3
"""
TetrisML - A machine learning project for evaluating Tetris board states

This program trains a neural network to evaluate Tetris board states
and can use this network to play Tetris using lookahead search.
"""

import argparse

# Import our modules
from tetris_ml.core.environment import TetrisEnv
from tetris_ml.models.value_network import TetrisValueNetwork
from tetris_ml.utils.checkpoint_manager import CheckpointManager
from tetris_ml.utils.training import train_tetris_value_network, plot_training_progress
from tetris_ml.utils.evaluation import evaluate_tetris_bot

def main():
    """Main entry point"""
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
        return
    
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
        plot_training_progress(trainer, "training_progress.png")
        
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
        plot_training_progress(trainer, "training_progress_resumed.png")
        
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
            return
            
        print(f"\nEvaluating model on {args.games} games...")
        results = evaluate_tetris_bot(value_network, num_games=args.games, render=args.render)
        
    print("\nProcess complete!")

if __name__ == "__main__":
    main()