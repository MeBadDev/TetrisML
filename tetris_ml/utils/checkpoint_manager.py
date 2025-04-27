import os
import json
import numpy as np
from datetime import datetime

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
            from tetris_ml.models.value_network import TetrisValueNetwork
            value_network = TetrisValueNetwork()
        
        # Load model weights
        model_path = os.path.join(latest_checkpoint["path"], "model.keras")
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
            from tetris_ml.models.value_network import TetrisValueNetwork
            value_network = TetrisValueNetwork()
            
        # Load model weights
        model_path = os.path.join(checkpoint["path"], "model.keras")
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
                    timestamp = parts[2] if len(parts) > 2 else ""
                    
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