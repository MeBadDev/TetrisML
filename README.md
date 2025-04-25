# TetrisRLA
### Tetris Reinforcement Learning Agent

This project implements a reinforcement learning model that plays modern battle Tetris using Deep Q-Networks (DQN).

## Features

- Custom Gymnasium environment for Tetris
- Deep Q-Network (DQN) with experience replay
- Rewards for:
  - Line clears (singles, doubles, triples, tetris)
  - T-spins
  - Perfect clears
  - Combo chains
- Visualization and evaluation tools

## Requirements

- Python 3.9+
- TensorFlow 2.x
- NumPy
- Gymnasium
- python-tetris

## Setup

1. Make sure you have activated the virtual environment:

```bash
source env/bin/activate
```

2. Install any missing dependencies:

```bash
pip install tensorflow gymnasium numpy python-tetris
```

## Training the Model

To train the reinforcement learning agent:

```bash
python main.py
```

The training process will create:
- `models/` directory with saved model checkpoints
- `logs/` directory with TensorBoard logs

You can monitor training progress using TensorBoard:

```bash
tensorboard --logdir=logs
```

## Evaluating the Model

After training or when using a pre-trained model, you can evaluate its performance:

```bash
python evaluate.py
```

This will run the agent with visualization and report statistics on its performance.

## How it Works

### Environment

The Tetris environment is represented as a 22x10 grid (20 visible rows plus 2 buffer rows). The agent observes the current state of the playfield and chooses from 7 possible actions: no-op, move left, move right, rotate, soft drop, hard drop, and swap.

### Rewards

The reward function incentivizes:
- Line clears (with higher rewards for more lines)
- Tetris (4-line clear) with bonus points
- T-spins (detected when a T piece is rotated just before line clear)
- Perfect clears (when the board is completely empty after line clear)
- Combo chains (consecutive line clears)

It also penalizes:
- High stacks (to avoid risky board states)
- Uneven surfaces (to encourage flat stacking)

### Neural Network Architecture

The model uses:
- Convolutional layers to recognize spatial patterns
- Dense layers for decision making
- Target network and experience replay for stable learning

## Customization

You can adjust hyperparameters in `main.py`:
- Learning rate
- Exploration rate (epsilon)
- Discount factor (gamma)
- Batch size
- Memory size
- Network architecture

## Future Improvements

Potential enhancements:
- Implement Double DQN or Dueling DQN
- Add prioritized experience replay
- Train against adversarial garbage lines
- Fine-tune reward function
- Incorporate additional modern Tetris features