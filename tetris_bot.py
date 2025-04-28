import numpy as np
from tetris import TetrisEngine

class TetrisBot:
    """
    An improved Tetris bot that uses a more sophisticated evaluation system
    to find the best moves, with emphasis on line clearing.
    """
    
    def __init__(self, game_engine):
        """
        Initialize the bot with a game engine.
        
        Args:
            game_engine: TetrisEngine instance
        """
        self.game = game_engine
        self.last_best_move = None
        self.planned_moves = []
        self.executing_plan = False
    
    def make_move(self):
        """
        Determine and execute the best move for the current piece.
        
        Returns:
            bool: True if a move was made, False if the game is over
        """
        if self.game.game_over:
            return False
        
        # Continue executing a plan if we have one
        if self.executing_plan and self.planned_moves:
            return self._execute_next_move()
        
        # Otherwise plan a new move
        return self._plan_and_execute()
    
    def _plan_and_execute(self):
        """Find the best move and create an execution plan"""
        best_move = self._find_best_move()
        self.last_best_move = best_move
        
        if not best_move:
            return False
        
        # Create a sequence of actions to perform the move
        self.planned_moves = []
        
        # First, check if we should hold the current piece
        if best_move['should_hold']:
            self.planned_moves.append('hold')
            # After hold, we need to recalculate the best move for the new piece
            # So we return after adding the hold move
            self.executing_plan = True
            return self._execute_next_move()
        
        # Otherwise, add the rotation moves
        for _ in range(best_move['rotation']):
            self.planned_moves.append('rotate_right')
        
        # Add horizontal movements
        current_x = self._get_piece_center_x()
        target_x = best_move['column']
        
        if current_x < target_x:
            # Need to move right
            for _ in range(target_x - current_x):
                self.planned_moves.append('move_right')
        else:
            # Need to move left
            for _ in range(current_x - target_x):
                self.planned_moves.append('move_left')
        
        # Finally add the drop
        self.planned_moves.append('hard_drop')
        
        # Execute the plan
        self.executing_plan = True
        return self._execute_next_move()
    
    def _execute_next_move(self):
        """Execute the next move in the planned sequence"""
        if not self.planned_moves:
            self.executing_plan = False
            return True
        
        move = self.planned_moves.pop(0)
        
        if move == 'hold':
            self.game.hold()
            # After hold, we need a new plan for the new piece
            self.executing_plan = False
            self.planned_moves = []
            return self._plan_and_execute()
        elif move == 'rotate_right':
            self.game.rotate_right()
        elif move == 'move_left':
            self.game.move_left()
        elif move == 'move_right':
            self.game.move_right()
        elif move == 'hard_drop':
            self.game.hard_drop()
            # We're done after a hard drop
            self.executing_plan = False
            self.planned_moves = []
        
        # Return True if the game is still going
        return not self.game.game_over
    
    def _get_piece_center_x(self):
        """Get the current horizontal center of the piece"""
        if not self.game.current_piece:
            return 0
        
        # Calculate the center of the piece
        min_col = min(c for _, c in self.game.current_piece)
        max_col = max(c for _, c in self.game.current_piece)
        return (min_col + max_col) // 2
    
    def _find_best_move(self):
        """
        Find the best move by simulating all possible positions.
        
        Returns:
            dict: Information about the best move
        """
        best_score = float('-inf')
        best_move = None
        
        # Get current piece and consider all rotations and positions
        current_piece = self.game.current_piece_name
        
        # Decide whether to try holding
        should_try_hold = self.game.can_hold and self.game.held_piece_name != current_piece
        
        # First, evaluate keeping the current piece
        best_move, best_score = self._evaluate_piece(current_piece, best_move, best_score, False)
        
        # Then try holding if possible
        if should_try_hold and self.game.held_piece_name is not None:
            held_piece = self.game.held_piece_name
            hold_move, hold_score = self._evaluate_piece(held_piece, best_move, best_score, True)
            
            if hold_score > best_score:
                best_move = hold_move
                best_score = hold_score
        
        # If we have no held piece but holding is allowed, also consider holding
        # to get the next piece from the queue
        if should_try_hold and self.game.held_piece_name is None and len(self.game.piece_queue) > 0:
            next_piece = self.game.piece_queue[0]
            hold_move, hold_score = self._evaluate_piece(next_piece, best_move, best_score, True)
            
            if hold_score > best_score:
                best_move = hold_move
                best_score = hold_score
        
        return best_move
    
    def _evaluate_piece(self, piece_name, best_move, best_score, should_hold):
        """Evaluate all possible placements of a specific piece"""
        # The maximum number of unique rotations for each piece
        max_rotations = {'O': 1, 'I': 2, 'S': 2, 'Z': 2, 'J': 4, 'L': 4, 'T': 4}
        
        for rotation in range(max_rotations.get(piece_name, 4)):
            for column in range(-2, self.game.width + 2):  # Try all possible columns
                # Create a test game to simulate this move
                test_game = self._create_game_copy()
                
                # If we're evaluating a held/next piece, update the current piece
                if should_hold:
                    # Save the original piece to restore later
                    original_piece = test_game.current_piece_name
                    test_game.current_piece_name = piece_name
                    # Reset position and get new coordinates
                    spawn_row = test_game.buffer_zone - 2
                    if piece_name in ['I', 'O']:
                        spawn_row = test_game.buffer_zone - 1
                    spawn_col = test_game.width // 2 - 2
                    test_game.current_pos = [spawn_row, spawn_col]
                    test_game.current_rotation = 0
                    test_game.current_piece = test_game._get_piece_coords(
                        piece_name, 0, [spawn_row, spawn_col]
                    )
                
                # Apply rotation
                for _ in range(rotation):
                    if not test_game.rotate_right():
                        break
                
                # Calculate the center of the piece after rotation
                min_col = min(c for _, c in test_game.current_piece)
                max_col = max(c for _, c in test_game.current_piece)
                center_col = (min_col + max_col) // 2
                
                # Move to target column
                delta_col = column - center_col
                move_func = test_game.move_right if delta_col > 0 else test_game.move_left
                for _ in range(abs(delta_col)):
                    if not move_func():
                        break
                
                # If move failed, the piece is in an invalid position
                if test_game.game_over:
                    continue
                
                # Drop the piece
                lines_before = test_game.lines_cleared
                test_game.hard_drop()
                lines_cleared = test_game.lines_cleared - lines_before
                
                # Skip if move led to game over
                if test_game.game_over:
                    continue
                
                # Evaluate the resulting position
                score = self._evaluate_board(test_game, lines_cleared)
                
                # Update best move if this is better
                if score > best_score:
                    best_score = score
                    best_move = {
                        'rotation': rotation,
                        'column': column,
                        'score': score,
                        'should_hold': should_hold
                    }
        
        return best_move, best_score
    
    def _create_game_copy(self):
        """Create a copy of the game for simulation"""
        # Create a new game with the same dimensions
        new_game = TetrisEngine(
            width=self.game.width,
            height=self.game.height,
            buffer_zone=self.game.buffer_zone
        )
        
        # Copy the board state
        new_game.board = self.game.board.copy()
        
        # Copy current piece information
        new_game.current_piece_name = self.game.current_piece_name
        new_game.current_rotation = self.game.current_rotation
        new_game.current_pos = self.game.current_pos.copy()
        new_game.current_piece = [tuple(coord) for coord in self.game.current_piece]
        
        # Copy other game state
        new_game.held_piece_name = self.game.held_piece_name
        new_game.can_hold = self.game.can_hold
        new_game.lines_cleared = self.game.lines_cleared
        new_game.combo_count = self.game.combo_count
        
        # Copy the piece queue (only need the first piece)
        if self.game.piece_queue:
            new_game.piece_queue = [self.game.piece_queue[0]]
        
        return new_game
    
    def _evaluate_board(self, game, lines_cleared):
        """
        Evaluate a board state with sophisticated metrics.
        
        Args:
            game: The game state to evaluate
            lines_cleared: Number of lines cleared in the last move
            
        Returns:
            float: A score representing the quality of the position
        """
        board = game.board[game.buffer_zone:, :]  # Only consider the visible part
        
        # Calculate key metrics
        heights = self._get_heights(board)
        holes = self._count_holes(board, heights)
        aggregate_height = sum(heights)
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
        wells = self._count_wells(board, heights)
        hole_depths = self._get_hole_depths(board, heights)
        
        # Highly reward line clears (exponential for multiple line clears)
        line_clear_score = 0
        if lines_cleared == 1:
            line_clear_score = 100  # Single
        elif lines_cleared == 2:
            line_clear_score = 300  # Double
        elif lines_cleared == 3:
            line_clear_score = 700  # Triple
        elif lines_cleared == 4:
            line_clear_score = 1500  # Tetris
        
        # Calculate a weighted score
        score = (
            line_clear_score * 5.0 +          # Extremely high priority on line clears
            -0.51 * aggregate_height +        # Lower height is better
            -0.36 * bumpiness +               # Smoother surface is better
            -0.76 * holes +                   # Heavily penalize holes
            -0.31 * sum(hole_depths) +        # Penalize deep holes more
            0.15 * wells                      # Small reward for wells (for Tetris setups)
        )
        
        # Additional penalty for very high stacks (risk of game over)
        max_height = max(heights) if heights else 0
        if max_height > game.height * 0.8:  # If stack is over 80% of board height
            score -= (max_height - game.height * 0.8) * 100  # Heavy penalty
        
        return score
    
    def _get_heights(self, board):
        """Calculate the height of each column"""
        heights = []
        for col in range(board.shape[1]):
            for row in range(board.shape[0]):
                if board[row, col] != 0:
                    heights.append(board.shape[0] - row)
                    break
            else:
                heights.append(0)  # Empty column
        return heights
    
    def _count_holes(self, board, heights):
        """Count empty cells that have blocks above them"""
        holes = 0
        for col in range(board.shape[1]):
            col_height = heights[col]
            # Start from the top of the column and count holes
            for row in range(board.shape[0] - col_height, board.shape[0]):
                if board[row, col] == 0:
                    holes += 1
        return holes
    
    def _count_wells(self, board, heights):
        """
        Count cells in wells (deep empty columns surrounded by higher columns)
        Wells are good for inserting I pieces for Tetris clears
        """
        wells = 0
        width = board.shape[1]
        
        # Check leftmost column
        if width > 1 and heights[0] < heights[1] - 1:
            wells += heights[1] - heights[0] - 1
        
        # Check middle columns
        for col in range(1, width - 1):
            left_higher = heights[col-1] > heights[col] + 1
            right_higher = heights[col+1] > heights[col] + 1
            
            if left_higher and right_higher:
                wells += min(heights[col-1], heights[col+1]) - heights[col] - 1
        
        # Check rightmost column
        if width > 1 and heights[width-1] < heights[width-2] - 1:
            wells += heights[width-2] - heights[width-1] - 1
        
        return wells
    
    def _get_hole_depths(self, board, heights):
        """
        Calculate how deep each hole is (distance from surface)
        Deeper holes are harder to clear and should be penalized more
        """
        hole_depths = []
        for col in range(board.shape[1]):
            col_height = heights[col]
            if col_height == 0:  # Skip empty columns
                continue
            
            # Find the highest block in this column
            top_block_row = board.shape[0] - col_height
            
            # Check for holes below it
            for row in range(top_block_row + 1, board.shape[0]):
                if board[row, col] == 0:
                    # Add the depth of this hole (distance from surface)
                    hole_depths.append(row - top_block_row)
        
        return hole_depths

def make_best_move(bot):
    """Wrapper function to make the best move"""
    return bot.make_move()