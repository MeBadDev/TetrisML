import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class TetrisValueNetwork:
    """Neural network for evaluating Tetris board states with a single score"""
    
    def __init__(self, input_shape=(20, 10, 1), piece_embedding_dim=8):
        self.input_shape = input_shape
        self.piece_embedding_dim = piece_embedding_dim
        self.model = self._build_model()
        self.gamma = 0.99  # Discount factor
        
    def _build_model(self):
        # Input for the board state
        board_input = keras.Input(shape=self.input_shape, name="board")
        
        # Input for piece information
        next_piece_input = keras.Input(shape=(1,), name="next_piece")
        held_piece_input = keras.Input(shape=(1,), name="held_piece")
        current_piece_input = keras.Input(shape=(1,), name="current_piece")
        
        # Process board with convolutional layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(board_input)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.Flatten()(x)
        
        # Embed piece information
        piece_embedding = layers.Embedding(
            input_dim=8,  # 7 pieces + None
            output_dim=self.piece_embedding_dim
        )
        
        next_piece_embedded = piece_embedding(next_piece_input)
        held_piece_embedded = piece_embedding(held_piece_input)
        current_piece_embedded = piece_embedding(current_piece_input)
        
        # Flatten piece embeddings
        next_piece_embedded = layers.Flatten()(next_piece_embedded)
        held_piece_embedded = layers.Flatten()(held_piece_embedded)
        current_piece_embedded = layers.Flatten()(current_piece_embedded)
        
        # Concatenate all features
        concat = layers.Concatenate()([
            x,
            current_piece_embedded,
            next_piece_embedded,
            held_piece_embedded
        ])
        
        # Process combined features
        concat = layers.Dense(256, activation='relu')(concat)
        concat = layers.Dense(128, activation='relu')(concat)
        
        # Output value (board evaluation score)
        evaluation_output = layers.Dense(1, name="evaluation")(concat)
        
        # Create model
        model = keras.Model(
            inputs=[board_input, next_piece_input, held_piece_input, current_piece_input],
            outputs=evaluation_output
        )
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        
        return model
    
    def predict(self, state):
        """Predict the evaluation score of a state"""
        # Prepare the inputs
        board = np.expand_dims(state['board'], axis=-1)  # Add channel dimension
        next_piece = np.array([state['next_piece']])
        held_piece = np.array([state['held_piece']])
        current_piece = np.array([state['current_piece']])
        
        # Make prediction
        return self.model.predict(
            [np.expand_dims(board, axis=0), next_piece, held_piece, current_piece],
            verbose=0
        )[0, 0]
    
    def train(self, states, target_values, batch_size=32, epochs=1):
        """Train the evaluation network"""
        boards = np.stack([np.expand_dims(s['board'], axis=-1) for s in states])
        next_pieces = np.array([s['next_piece'] for s in states])
        held_pieces = np.array([s['held_piece'] for s in states])
        current_pieces = np.array([s['current_piece'] for s in states])
        
        history = self.model.fit(
            [boards, next_pieces, held_pieces, current_pieces],
            target_values,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0
        )
        
        return history
        
    def save(self, path):
        """Save the model to disk"""
        self.model.save(path)
        
    def load(self, path):
        """Load the model from disk"""
        self.model = keras.models.load_model(path)