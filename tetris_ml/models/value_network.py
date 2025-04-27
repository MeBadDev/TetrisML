import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers

class TetrisValueNetwork:
    """Neural network for evaluating Tetris board states with a single score"""
    
    def __init__(self, input_shape=(20, 10, 1), piece_embedding_dim=16):
        self.input_shape = input_shape
        self.piece_embedding_dim = piece_embedding_dim
        self.model = self._build_model()
        self.gamma = 0.99  # Discount factor
        
    def _conv_block(self, x, filters, kernel_size, dropout_rate=0.0):
        """Create a convolutional block with optional residual connection"""
        conv = layers.Conv2D(filters, kernel_size, padding='same')(x)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        
        if dropout_rate > 0:
            conv = layers.Dropout(dropout_rate)(conv)
            
        return conv
        
    def _residual_block(self, x, filters):
        """Create a residual block with two convolutional layers"""
        # Save input for residual connection
        input_tensor = x
        
        # First convolutional layer
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Second convolutional layer
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Add residual connection - ensure dimensions match
        if input_tensor.shape[-1] != filters:
            input_tensor = layers.Conv2D(filters, (1, 1), padding='same')(input_tensor)
            
        x = layers.add([x, input_tensor])
        x = layers.Activation('relu')(x)
        
        return x
        
    def _board_feature_extractor(self, board_input):
        """Extract spatial features from the Tetris board"""
        # Initial convolution with batch normalization
        x = layers.Conv2D(32, (3, 3), padding='same')(board_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Residual blocks
        x = self._residual_block(x, 64)
        x = self._residual_block(x, 64)
        
        # Row-wise features (horizontal pattern detection)
        row_features = layers.Conv2D(32, (1, 5), padding='same', activation='relu')(board_input)
        row_features = layers.Conv2D(32, (1, 3), padding='same', activation='relu')(row_features)
        
        # Column-wise features (vertical pattern detection)
        col_features = layers.Conv2D(32, (5, 1), padding='same', activation='relu')(board_input)
        col_features = layers.Conv2D(32, (3, 1), padding='same', activation='relu')(col_features)
        
        # Combine all features - ensure all have the same spatial dimensions
        combined = layers.Concatenate(axis=-1)([x, row_features, col_features])
        
        # Final convolution
        combined = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(combined)
        
        # Global spatial features
        spatial_features = layers.GlobalAveragePooling2D()(combined)
        
        # Add global max pooling features
        max_features = layers.GlobalMaxPooling2D()(combined)
        
        # Combine global features
        global_features = layers.Concatenate()([spatial_features, max_features])
        
        return global_features
        
    def _build_model(self):
        # Input for the board state
        board_input = keras.Input(shape=self.input_shape, name="board")
        
        # Input for piece information
        next_piece_input = keras.Input(shape=(1,), name="next_piece")
        held_piece_input = keras.Input(shape=(1,), name="held_piece")
        current_piece_input = keras.Input(shape=(1,), name="current_piece")
        
        # Process board with advanced feature extraction
        board_features = self._board_feature_extractor(board_input)
        
        # Embed piece information with a larger embedding dimension
        piece_embedding = layers.Embedding(
            input_dim=8,  # 7 pieces + None
            output_dim=self.piece_embedding_dim,
            embeddings_regularizer=regularizers.l2(1e-4)
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
            board_features,
            current_piece_embedded,
            next_piece_embedded,
            held_piece_embedded
        ])
        
        # Process combined features
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(concat)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output value (board evaluation score)
        evaluation_output = layers.Dense(1, name="evaluation")(x)
        
        # Create model
        model = keras.Model(
            inputs=[board_input, next_piece_input, held_piece_input, current_piece_input],
            outputs=evaluation_output
        )
        
        # Compile model with improved optimizer settings
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.0005,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            ),
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