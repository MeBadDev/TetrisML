import unittest
import numpy as np
from tetris import TetrisEngine, TETROMINOS
import sys
import time
import random

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Custom test result class for colorful output
class ColorfulTestResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.stream = stream
        self.descriptions = descriptions
        self.verbosity = verbosity

    def startTest(self, test):
        super().startTest(test)
        if self.verbosity > 1:
            test_name = self.getDescription(test)
            self.stream.write(f"{Colors.CYAN}⏳ Running: {test_name}{Colors.ENDC} ... ")
            self.stream.flush()

    def addSuccess(self, test):
        super().addSuccess(test)
        if self.verbosity > 1:
            self.stream.write(f"{Colors.GREEN}✅ PASS{Colors.ENDC}\n")
        else:
            self.stream.write(f"{Colors.GREEN}✅{Colors.ENDC}")

    def addError(self, test, err):
        super().addError(test, err)
        if self.verbosity > 1:
            self.stream.write(f"{Colors.RED}❌ ERROR{Colors.ENDC}\n")
        else:
            self.stream.write(f"{Colors.RED}E{Colors.ENDC}")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.verbosity > 1:
            self.stream.write(f"{Colors.RED}❌ FAIL{Colors.ENDC}\n")
        else:
            self.stream.write(f"{Colors.RED}F{Colors.ENDC}")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.verbosity > 1:
            self.stream.write(f"{Colors.YELLOW}⏭️ SKIP{Colors.ENDC}\n")
        else:
            self.stream.write(f"{Colors.YELLOW}s{Colors.ENDC}")

    def printErrors(self):
        if self.errors or self.failures:
            self.stream.write(f"\n{Colors.RED}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")
            self.stream.write(f"{Colors.RED}{Colors.BOLD}FAILED TESTS{Colors.ENDC}\n")
            self.stream.write(f"{Colors.RED}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")
        super().printErrors()

    def printErrorList(self, flavour, errors):
        for test, err in errors:
            self.stream.write(f"\n{Colors.RED}{Colors.BOLD}{flavour}: {self.getDescription(test)}{Colors.ENDC}\n")
            self.stream.write(f"{Colors.RED}{err}{Colors.ENDC}\n")

# Custom test runner using our colorful result class
class ColorfulTestRunner(unittest.TextTestRunner):
    def __init__(self, stream=sys.stderr, descriptions=True, verbosity=2,
                 failfast=False, buffer=False, resultclass=ColorfulTestResult):
        super().__init__(stream, descriptions, verbosity,
                         failfast, buffer, resultclass)

    def run(self, test):
        result = super().run(test)
        
        # Print a summary
        self.stream.write("\n")
        self.stream.write(f"{Colors.BOLD}{'='*70}{Colors.ENDC}\n")
        self.stream.write(f"{Colors.BOLD}TEST SUMMARY{Colors.ENDC}\n")
        self.stream.write(f"{Colors.BOLD}{'='*70}{Colors.ENDC}\n")
        
        # Count test cases
        run = result.testsRun
        
        # Print counts with colors based on results
        if result.wasSuccessful():
            self.stream.write(f"{Colors.GREEN}✅ PASSED: {run}/{run} tests passed{Colors.ENDC}\n")
        else:
            fails = len(result.failures)
            errors = len(result.errors)
            skips = len(getattr(result, 'skipped', []))
            success = run - fails - errors - skips
            
            if success > 0:
                self.stream.write(f"{Colors.GREEN}✅ PASSED: {success}/{run} tests passed{Colors.ENDC}\n")
            if fails > 0:
                self.stream.write(f"{Colors.RED}❌ FAILED: {fails}/{run} tests failed{Colors.ENDC}\n")
            if errors > 0:
                self.stream.write(f"{Colors.RED}❌ ERRORS: {errors}/{run} tests had errors{Colors.ENDC}\n")
            if skips > 0:
                self.stream.write(f"{Colors.YELLOW}⏭️ SKIPPED: {skips}/{run} tests skipped{Colors.ENDC}\n")
        
        elapsed = time.time() - getattr(result, 'start_time', time.time())
        self.stream.write(f"{Colors.BLUE}⏱️ TIME: {elapsed:.3f} seconds{Colors.ENDC}\n")
        
        return result

class TestTetrisEngine(unittest.TestCase):
    def setUp(self):
        """Set up a fresh TetrisEngine instance for each test."""
        self.engine = TetrisEngine(width=10, height=20, buffer_zone=2)
        
    def test_initialization(self):
        """Test that the game engine initializes correctly."""
        self.assertEqual(self.engine.width, 10)
        self.assertEqual(self.engine.height, 20)
        self.assertEqual(self.engine.buffer_zone, 2)
        self.assertEqual(self.engine.board_height, 22)
        self.assertEqual(self.engine.garbage_sent, 0)
        self.assertEqual(self.engine.garbage_queue, 0)
        self.assertEqual(self.engine.lines_cleared, 0)
        self.assertEqual(self.engine.level, 1)
        self.assertEqual(self.engine.combo_count, 0)
        self.assertFalse(self.engine.back_to_back)
        self.assertFalse(self.engine.game_over)
        self.assertTrue(self.engine.can_hold)
        self.assertIsNone(self.engine.held_piece_name)
        self.assertIsNotNone(self.engine.current_piece)
        self.assertIsNotNone(self.engine.current_piece_name)
        self.assertIn(self.engine.current_piece_name, TETROMINOS.keys())
        
    def test_piece_movement(self):
        """Test piece movement (left, right, soft drop)."""
        # Store initial position
        initial_pos = self.engine.current_pos.copy()
        
        # Test move left
        self.engine.move_left()
        self.assertEqual(self.engine.current_pos[1], initial_pos[1] - 1)
        
        # Test move right (twice to return to original and move one more)
        self.engine.move_right()
        self.engine.move_right()
        self.assertEqual(self.engine.current_pos[1], initial_pos[1] + 1)
        
        # Test soft drop
        self.engine.soft_drop()
        self.assertEqual(self.engine.current_pos[0], initial_pos[0] + 1)
        
    def test_rotation(self):
        """Test piece rotation."""
        # Skip if the current piece is 'O' (doesn't rotate)
        if self.engine.current_piece_name == 'O':
            self.engine._spawn_piece()  # Get new piece
        
        initial_rotation = self.engine.current_rotation
        
        # Test rotation right
        self.engine.rotate_right()
        self.assertEqual(self.engine.current_rotation, (initial_rotation + 1) % 4)
        
        # Test rotation left (returns to initial state)
        self.engine.rotate_left()
        self.assertEqual(self.engine.current_rotation, initial_rotation)
        
    def test_hard_drop(self):
        """Test hard drop and piece locking."""
        initial_board = np.copy(self.engine.board)
        piece_name = self.engine.current_piece_name
        
        # Hard drop the piece
        self.engine.hard_drop()
        
        # Check that a new piece has spawned
        self.assertNotEqual(self.engine.current_piece_name, piece_name)
        
        # Check that the board has changed (piece was placed)
        self.assertFalse(np.array_equal(self.engine.board, initial_board))
        
    def test_hold_piece(self):
        """Test the hold piece functionality."""
        original_piece = self.engine.current_piece_name
        
        # Hold current piece
        self.engine.hold()
        
        # Check that held_piece is set and a new current_piece is spawned
        self.assertEqual(self.engine.held_piece_name, original_piece)
        self.assertNotEqual(self.engine.current_piece_name, original_piece)
        
        # Check that can_hold is now False
        self.assertFalse(self.engine.can_hold)
        
        # Try to hold again (should fail)
        new_piece = self.engine.current_piece_name
        result = self.engine.hold()
        self.assertFalse(result)
        self.assertEqual(self.engine.current_piece_name, new_piece)
        
    def test_line_clearing(self):
        """Test line clearing and garbage calculation."""
        # Create a board with a complete line at the bottom
        self.engine.board = np.zeros((self.engine.board_height, self.engine.width), dtype=int)
        bottom_row = self.engine.board_height - 1
        self.engine.board[bottom_row, :] = 1  # Fill the bottom row
        
        # Place the current piece to trigger line clearing
        self.engine._lock_piece()
        
        # Check that one line was cleared
        self.assertEqual(self.engine.lines_cleared, 1)
        
        # Single line clear sends no garbage in modern Tetris
        self.assertEqual(self.engine.garbage_sent, 0)
        
    def test_game_over(self):
        """Test game over condition."""
        # Fill the spawn area and a bit below to cause game over
        # We need to fill enough cells to ensure the piece has nowhere to spawn
        for r in range(self.engine.buffer_zone + 2):
            self.engine.board[r, 3:7] = 1  # Fill the center columns
            
        # Store current piece name
        current_piece = self.engine.current_piece_name
            
        # Try to hard drop and spawn a new piece (should trigger game over)
        self.engine.hard_drop()
        
        # Check game over flag
        self.assertTrue(self.engine.game_over)
        
    def test_get_state(self):
        """Test that get_state returns the correct game state."""
        state = self.engine.get_state()
        
        # Check that state contains the expected keys
        self.assertIn('board', state)
        self.assertIn('garbage_sent', state)
        self.assertIn('garbage_queue', state)
        self.assertIn('combo', state)
        self.assertIn('back_to_back', state)
        self.assertIn('lines', state)
        self.assertIn('level', state)
        self.assertIn('game_over', state)
        self.assertIn('next_piece', state)
        self.assertIn('held_piece', state)
        
        # Check that board has correct dimensions (without buffer zone)
        self.assertEqual(state['board'].shape, (self.engine.height, self.engine.width))
        
    def test_wall_collision(self):
        """Test wall collision detection."""
        # Move piece to left wall
        while self.engine.move_left():
            pass
            
        # Try to move left again (should fail)
        self.assertFalse(self.engine.move_left())
        
        # Move piece to right wall
        while self.engine.move_right():
            pass
            
        # Try to move right again (should fail)
        self.assertFalse(self.engine.move_right())
        
    def test_floor_collision(self):
        """Test floor collision detection."""
        # Move piece to the floor with soft drops
        while self.engine.soft_drop():
            pass
            
        # Check that piece has been locked and a new piece spawned
        self.assertTrue(np.any(self.engine.board > 0))
        
    def test_piece_collision(self):
        """Test piece-to-piece collision."""
        # Create a "well" by filling all but the middle columns
        board_height = self.engine.board_height
        for r in range(board_height - 5, board_height):
            for c in range(self.engine.width):
                if c < 4 or c > 5:
                    self.engine.board[r, c] = 1
                    
        # Drop a piece into the well
        self.engine.hard_drop()
        
        # Drop another piece (should rest on top of the first)
        current_piece = self.engine.current_piece_name
        self.engine.hard_drop()
        
        # Verify that the pieces stacked
        self.assertNotEqual(self.engine.current_piece_name, current_piece)
        
    def test_calculate_garbage(self):
        """Test garbage calculation for different scenarios."""
        # Reset garbage counters
        self.engine.garbage_sent = 0
        self.engine.garbage_queue = 0
        
        # Test single line clear (should give 0 garbage)
        self.engine._calculate_garbage(1)
        self.assertEqual(self.engine.garbage_queue, 0)
        
        # Reset counters
        self.engine.garbage_queue = 0
        self.engine.combo_count = 0
        
        # Test double line clear (should give 1 garbage)
        self.engine._calculate_garbage(2)
        self.assertEqual(self.engine.garbage_queue, 1)
        
        # Reset counters
        self.engine.garbage_queue = 0
        self.engine.combo_count = 0
        
        # Test triple line clear (should give 2 garbage)
        self.engine._calculate_garbage(3)
        self.assertEqual(self.engine.garbage_queue, 2)
        
        # Reset counters
        self.engine.garbage_queue = 0
        self.engine.combo_count = 0
        
        # Test Tetris (should give 4 garbage)
        self.engine._calculate_garbage(4)
        self.assertEqual(self.engine.garbage_queue, 4)
        
    def test_t_spin_garbage(self):
        """Test garbage calculation for T-spins."""
        # Reset garbage counters and ensure combo count is 0
        self.engine.garbage_sent = 0
        self.engine.garbage_queue = 0
        self.engine.combo_count = 0  # Reset combo to avoid interference
        self.engine.back_to_back = False  # Reset back-to-back
        
        # Test T-spin mini (no lines cleared, should give 1 garbage)
        self.engine._calculate_garbage(0, True)
        self.assertEqual(self.engine.garbage_queue, 1)
        
        # Reset counters
        self.engine.garbage_queue = 0
        self.engine.combo_count = 0
        self.engine.back_to_back = False
        
        # Test T-spin single (should give 2 garbage)
        self.engine._calculate_garbage(1, True)
        self.assertEqual(self.engine.garbage_queue, 2)
        
        # Reset counters
        self.engine.garbage_queue = 0
        self.engine.combo_count = 0
        self.engine.back_to_back = False
        
        # Test T-spin double (should give 4 garbage)
        self.engine._calculate_garbage(2, True)
        self.assertEqual(self.engine.garbage_queue, 4)
        
        # Reset counters
        self.engine.garbage_queue = 0
        self.engine.combo_count = 0
        self.engine.back_to_back = False
        
        # Test T-spin triple (should give 6 garbage)
        self.engine._calculate_garbage(3, True)
        self.assertEqual(self.engine.garbage_queue, 6)

    def test_perfect_clear(self):
        """Test perfect clear detection and garbage calculation."""
        # Start with an empty board
        self.engine.board = np.zeros((self.engine.board_height, self.engine.width), dtype=int)
        self.engine.garbage_sent = 0
        self.engine.garbage_queue = 0
        
        # Create a special scenario for perfect clear
        # Add blocks to form a single line at the bottom, leaving one gap for an I piece
        bottom_row = self.engine.board_height - 1
        for c in range(self.engine.width):
            if c != 4:  # Leave a gap at column 4
                self.engine.board[bottom_row, c] = 1
        
        # Position an I piece to fill the gap
        self.engine.current_piece_name = 'I'
        self.engine.current_rotation = 1  # Vertical I piece
        self.engine.current_pos = [bottom_row - 3, 4]  # Position to fill gap
        self.engine.current_piece = self.engine._get_piece_coords('I', 1, [bottom_row - 3, 4])
        
        # Hard drop the piece manually (without calling hard_drop which would also spawn a new piece)
        while self.engine._move(1, 0):
            pass
        
        # Check that board is about to be perfectly cleared
        self.engine.garbage_queue = 0  # Reset garbage for clarity
        
        # Now lock the piece and check perfect clear directly
        self.engine.board[bottom_row, 4] = 1  # Manually fill the gap
        
        # Manually clear the line
        self.engine.board[1:bottom_row+1, :] = self.engine.board[0:bottom_row, :]
        self.engine.board[0, :] = 0
        
        # Board should now be empty in the visible portion
        self.assertTrue(np.all(self.engine.board[self.engine.buffer_zone:, :] == 0))
        
        # Test perfect clear detection directly with single line cleared
        result = self.engine._check_perfect_clear(1)
        self.assertTrue(result)
        
        # Check that perfect clear bonus was applied (10 garbage for single line)
        self.assertEqual(self.engine.garbage_queue, 10)

    def test_perfect_clear_direct(self):
        """Test perfect clear directly using the _check_perfect_clear function."""
        # Start with an empty board
        self.engine.board = np.zeros((self.engine.board_height, self.engine.width), dtype=int)
        self.engine.garbage_queue = 0
        
        # Board should be empty, so calling _check_perfect_clear should return True
        # and add garbage to queue (10 for default perfect clear)
        self.assertTrue(self.engine._check_perfect_clear())
        self.assertEqual(self.engine.garbage_queue, 10)
        
        # Reset and test with specific line clears
        self.engine.garbage_queue = 0
        self.assertTrue(self.engine._check_perfect_clear(2))  # Double perfect clear
        self.assertEqual(self.engine.garbage_queue, 12)
        
        self.engine.garbage_queue = 0
        self.assertTrue(self.engine._check_perfect_clear(4))  # Tetris perfect clear  
        self.assertEqual(self.engine.garbage_queue, 16)

    def test_back_to_back_bonus(self):
        """Test back-to-back bonus for difficult clears (Tetris or T-spin)."""
        # Reset garbage counters and combo count
        self.engine.garbage_sent = 0
        self.engine.garbage_queue = 0
        self.engine.back_to_back = False
        self.engine.combo_count = 0  # Reset combo to avoid interference
        
        # First Tetris doesn't get B2B bonus
        self.engine._calculate_garbage(4)  # Tetris (4 lines)
        first_tetris_garbage = self.engine.garbage_queue
        self.assertEqual(first_tetris_garbage, 4)  # 4 lines for a Tetris
        self.assertTrue(self.engine.back_to_back)  # B2B flag should be set
        
        # Reset combo before second tetris
        self.engine.combo_count = 0
        
        # Second Tetris gets B2B bonus (+1)
        self.engine.garbage_queue = 0
        self.engine._calculate_garbage(4)  # Second Tetris
        second_tetris_garbage = self.engine.garbage_queue
        self.assertEqual(second_tetris_garbage, 5)  # 4 lines + 1 B2B bonus
        
        # Reset combo before double
        self.engine.combo_count = 0
        
        # Regular line clear (not Tetris) breaks B2B
        self.engine.garbage_queue = 0
        self.engine._calculate_garbage(2)  # Double line clear
        self.assertEqual(self.engine.garbage_queue, 1)  # 1 line for double
        self.assertFalse(self.engine.back_to_back)  # B2B flag should be reset
        
        # Reset combo before T-spin
        self.engine.combo_count = 0
        
        # T-spin also counts as difficult clear for B2B
        self.engine.garbage_queue = 0
        self.engine._calculate_garbage(2, True)  # T-spin double
        first_tspin_garbage = self.engine.garbage_queue
        self.assertEqual(first_tspin_garbage, 4)  # 4 lines for T-spin double
        self.assertTrue(self.engine.back_to_back)  # B2B flag should be set
        
        # Reset combo before second T-spin
        self.engine.combo_count = 0
        
        # T-spin after T-spin gets B2B bonus
        self.engine.garbage_queue = 0
        self.engine._calculate_garbage(2, True)  # Second T-spin double
        second_tspin_garbage = self.engine.garbage_queue
        self.assertEqual(second_tspin_garbage, 5)  # 4 lines + 1 B2B bonus
        
        # Reset combo before Tetris
        self.engine.combo_count = 0
        
        # T-spin after Tetris gets B2B bonus too (and vice versa)
        self.engine.garbage_queue = 0
        self.engine._calculate_garbage(4)  # Tetris after T-spin
        self.assertEqual(self.engine.garbage_queue, 5)  # 4 lines + 1 B2B bonus

    def test_receive_garbage_basic(self):
        """Test basic garbage receiving functionality with fixed amount of lines."""
        # Start with a clean board
        self.engine.board = np.zeros((self.engine.board_height, self.engine.width), dtype=int)
        
        # Store initial state for comparison
        initial_board = np.copy(self.engine.board)
        
        # Add 3 lines of garbage
        lines_added = self.engine.receive_garbage(3)
        
        # Check that correct number of lines were added
        self.assertEqual(lines_added, 3)
        
        # Check that garbage rows were added at the bottom
        # Rows above buffer_zone + height - 3 should be empty
        self.assertTrue(np.all(self.engine.board[:self.engine.board_height-3, :] == 0))
        
        # Check bottom rows contain garbage (value 8)
        for r in range(self.engine.board_height-3, self.engine.board_height):
            # Count non-zero cells in this row
            filled_cells = np.count_nonzero(self.engine.board[r, :])
            # Should be width - 1 (one cell is the garbage hole)
            self.assertEqual(filled_cells, self.engine.width - 1)
            
            # Verify cells are filled with garbage blocks (value 8)
            for c in range(self.engine.width):
                if self.engine.board[r, c] != 0:
                    self.assertEqual(self.engine.board[r, c], 8)
    
    def test_receive_garbage_random(self):
        """Test garbage receiving with random number of lines."""
        # Set random seed for reproducibility
        random.seed(42)
        
        # Start with a clean board
        self.engine.board = np.zeros((self.engine.board_height, self.engine.width), dtype=int)
        
        # Add random number of garbage lines (between 2-6)
        lines_added = self.engine.receive_garbage()
        
        # Check that a valid number of lines were added (2-6)
        self.assertTrue(2 <= lines_added <= 6)
        
        # Check garbage properties similar to basic test
        # Count the number of rows with garbage
        garbage_rows = 0
        for r in range(self.engine.board_height):
            # If row has garbage (value 8), count it
            if np.any(self.engine.board[r, :] == 8):
                garbage_rows += 1
                
        self.assertEqual(garbage_rows, lines_added)
    
    def test_receive_garbage_with_piece(self):
        """Test garbage receiving handles active piece correctly."""
        # Position the current piece at the bottom of the board
        self.engine.current_pos[0] = self.engine.board_height - 4
        original_pos = self.engine.current_pos.copy()
        piece_name = self.engine.current_piece_name
        rotation = self.engine.current_rotation
        
        # Add 3 lines of garbage
        self.engine.receive_garbage(3)
        
        # Check that the piece was moved up by 3 rows
        self.assertEqual(self.engine.current_pos[0], original_pos[0] - 3)
        self.assertEqual(self.engine.current_piece_name, piece_name)
        self.assertEqual(self.engine.current_rotation, rotation)
        
    def test_receive_garbage_game_over(self):
        """Test that game over is triggered when piece can't be repositioned."""
        # Fill most of the board except a small area at the bottom
        for r in range(4, self.engine.board_height - 2):
            for c in range(self.engine.width):
                self.engine.board[r, c] = 1
        
        # Position piece just above filled area
        self.engine.current_pos = [3, 4]
        self.engine.current_piece = self.engine._get_piece_coords(
            self.engine.current_piece_name, self.engine.current_rotation, [3, 4])
        
        # Add 4 lines of garbage (should trigger game over as piece can't move up enough)
        self.engine.receive_garbage(4)
        
        # Check game over state
        self.assertTrue(self.engine.game_over)
    
    def test_receive_garbage_with_holes(self):
        """Test that garbage lines have exactly one empty cell each (garbage hole)."""
        # Start with a clean board
        self.engine.board = np.zeros((self.engine.board_height, self.engine.width), dtype=int)
        
        # Add 5 lines of garbage
        lines_added = self.engine.receive_garbage(5)
        
        # For each garbage line, check that it has exactly one hole
        for r in range(self.engine.board_height - lines_added, self.engine.board_height):
            # Count zero cells (holes)
            holes = np.count_nonzero(self.engine.board[r, :] == 0)
            self.assertEqual(holes, 1, f"Row {r} has {holes} holes, expected exactly 1")
            
    def test_receive_garbage_zero_lines(self):
        """Test receiving zero garbage lines (shouldn't change board)."""
        # Start with a clean board
        self.engine.board = np.zeros((self.engine.board_height, self.engine.width), dtype=int)
        original_board = np.copy(self.engine.board)
        
        # Try to add 0 garbage lines
        lines_added = self.engine.receive_garbage(0)
        
        # Check no lines were added
        self.assertEqual(lines_added, 0)
        
        # Check board is unchanged
        self.assertTrue(np.array_equal(self.engine.board, original_board))

if __name__ == '__main__':
    # Add start time attribute to the result
    ColorfulTestResult.start_time = time.time()
    
    # Use our custom runner for more colorful output
    runner = ColorfulTestRunner()
    unittest.main(testRunner=runner)