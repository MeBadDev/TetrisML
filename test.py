import pygame
import numpy as np
import time
import sys
from tetris import TetrisEngine, TETROMINOS
from tetris_bot import TetrisBot, make_best_move

# Define colors for the Tetris blocks
COLORS = {
    0: (0, 0, 0),        # Empty space (black)
    1: (0, 240, 240),    # I (cyan)
    2: (240, 240, 0),    # O (yellow)
    3: (160, 0, 240),    # T (purple)
    4: (0, 240, 0),      # S (green)
    5: (240, 0, 0),      # Z (red)
    6: (0, 0, 240),      # J (blue)
    7: (240, 160, 0),    # L (orange)
    8: (80, 80, 80)      # Garbage (gray)
}

# Display settings
BLOCK_SIZE = 30
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BUFFER_ZONE = 2
BOARD_OFFSET_X = 100
BOARD_OFFSET_Y = 50
NEXT_PIECE_OFFSET_X = BOARD_OFFSET_X + (BOARD_WIDTH * BLOCK_SIZE) + 50
INFO_OFFSET_X = NEXT_PIECE_OFFSET_X
INFO_OFFSET_Y = BOARD_OFFSET_Y + 200
NEXT_PIECE_OFFSET_Y = BOARD_OFFSET_Y

class TetrisVisualizer:
    def __init__(self, game, bot=None, speed=5, auto_play=True):
        self.game = game
        self.bot = bot
        self.speed = speed  # Moves per second
        self.auto_play = auto_play
        
        # Initialize pygame
        pygame.init()
        
        # Calculate window size
        self.window_width = BOARD_OFFSET_X + (BOARD_WIDTH * BLOCK_SIZE) + 200
        self.window_height = BOARD_OFFSET_Y + (BOARD_HEIGHT * BLOCK_SIZE) + 50
        
        # Create window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Tetris Bot Visualization")
        
        # Font for text display
        self.font = pygame.font.SysFont('Arial', 18)
        self.large_font = pygame.font.SysFont('Arial', 24)
        
        # Game state
        self.paused = False
        self.last_move_time = time.time()
        self.move_delay = 1.0 / self.speed
        
    def draw_board(self):
        """Draw the Tetris board with blocks"""
        # Draw border
        pygame.draw.rect(
            self.screen, 
            (50, 50, 50), 
            (BOARD_OFFSET_X - 2, BOARD_OFFSET_Y - 2, 
             BLOCK_SIZE * BOARD_WIDTH + 4, BLOCK_SIZE * BOARD_HEIGHT + 4),
            2
        )
            
        # Draw blocks
        board = self.game.get_render_board()
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                block_value = board[row, col]
                color = COLORS.get(block_value, (100, 100, 100))
                
                # Draw filled block if it's not empty
                if block_value != 0:
                    pygame.draw.rect(
                        self.screen, 
                        color, 
                        (BOARD_OFFSET_X + col * BLOCK_SIZE, 
                         BOARD_OFFSET_Y + row * BLOCK_SIZE, 
                         BLOCK_SIZE, BLOCK_SIZE)
                    )
                
                # Draw block outline
                pygame.draw.rect(
                    self.screen, 
                    (20, 20, 20), 
                    (BOARD_OFFSET_X + col * BLOCK_SIZE, 
                     BOARD_OFFSET_Y + row * BLOCK_SIZE, 
                     BLOCK_SIZE, BLOCK_SIZE),
                    1
                )
                
    def draw_next_piece(self):
        """Draw the next piece preview"""
        # Get the next piece
        state = self.game.get_state()
        next_piece = state['next_piece']
        
        if next_piece:
            # Draw title
            text = self.font.render("Next Piece:", True, (255, 255, 255))
            self.screen.blit(text, (NEXT_PIECE_OFFSET_X, NEXT_PIECE_OFFSET_Y))
            
            # Get piece data
            piece_data = TETROMINOS[next_piece]
            color = COLORS.get(piece_data['color'], (100, 100, 100))
            
            # Draw piece blocks
            for r, c in piece_data['shape']:
                pygame.draw.rect(
                    self.screen, 
                    color, 
                    (NEXT_PIECE_OFFSET_X + c * BLOCK_SIZE, 
                     NEXT_PIECE_OFFSET_Y + 30 + r * BLOCK_SIZE, 
                     BLOCK_SIZE, BLOCK_SIZE)
                )
                pygame.draw.rect(
                    self.screen, 
                    (20, 20, 20), 
                    (NEXT_PIECE_OFFSET_X + c * BLOCK_SIZE, 
                     NEXT_PIECE_OFFSET_Y + 30 + r * BLOCK_SIZE, 
                     BLOCK_SIZE, BLOCK_SIZE),
                    1
                )
    
    def draw_held_piece(self):
        """Draw the held piece"""
        state = self.game.get_state()
        held_piece = state['held_piece']
        
        if held_piece:
            # Draw title
            text = self.font.render("Held Piece:", True, (255, 255, 255))
            self.screen.blit(text, (NEXT_PIECE_OFFSET_X, NEXT_PIECE_OFFSET_Y + 120))
            
            # Get piece data
            piece_data = TETROMINOS[held_piece]
            color = COLORS.get(piece_data['color'], (100, 100, 100))
            
            # Draw piece blocks
            for r, c in piece_data['shape']:
                pygame.draw.rect(
                    self.screen, 
                    color, 
                    (NEXT_PIECE_OFFSET_X + c * BLOCK_SIZE, 
                     NEXT_PIECE_OFFSET_Y + 150 + r * BLOCK_SIZE, 
                     BLOCK_SIZE, BLOCK_SIZE)
                )
                pygame.draw.rect(
                    self.screen, 
                    (20, 20, 20), 
                    (NEXT_PIECE_OFFSET_X + c * BLOCK_SIZE, 
                     NEXT_PIECE_OFFSET_Y + 150 + r * BLOCK_SIZE, 
                     BLOCK_SIZE, BLOCK_SIZE),
                    1
                )
    
    def draw_stats(self):
        """Draw game statistics"""
        state = self.game.get_state()
        last_move = self.game.get_last_move_stats()
        
        # Game stats
        stats_text = [
            f"Lines: {state['lines']}",
            f"Level: {state['level']}",
            f"Garbage Sent: {state['garbage_sent']}",
            f"Combo: {state['combo']}",
            "", 
            "Last Move:",
            f"Clear: {last_move['clear_type']}",
            f"Lines: {last_move['lines_cleared']}",
            f"B2B: {'Yes' if last_move['is_back_to_back'] else 'No'}",
            f"Perfect: {'Yes' if last_move['is_perfect_clear'] else 'No'}"
        ]
        
        # Add Bot Evaluation Score if bot exists and has last_best_move info
        if self.bot and hasattr(self.bot, 'last_best_move') and self.bot.last_best_move:
            score = self.bot.last_best_move.get('score', 'N/A')
            stats_text.extend([
                "",
                "Bot Evaluation:",
                f"Best Score: {score:.2f}" if isinstance(score, (int, float)) else f"Best Score: {score}"
            ])
        
        # Display each stat
        for i, text in enumerate(stats_text):
            rendered_text = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(rendered_text, (INFO_OFFSET_X, INFO_OFFSET_Y + i * 25))
            
        # Game over message
        if state['game_over']:
            game_over_text = self.large_font.render("GAME OVER", True, (255, 0, 0))
            text_rect = game_over_text.get_rect(center=(
                BOARD_OFFSET_X + (BOARD_WIDTH * BLOCK_SIZE) // 2,
                BOARD_OFFSET_Y + (BOARD_HEIGHT * BLOCK_SIZE) // 2
            ))
            self.screen.blit(game_over_text, text_rect)
            
        # Paused message
        if self.paused:
            paused_text = self.large_font.render("PAUSED", True, (255, 255, 0))
            text_rect = paused_text.get_rect(center=(
                BOARD_OFFSET_X + (BOARD_WIDTH * BLOCK_SIZE) // 2,
                BOARD_OFFSET_Y + (BOARD_HEIGHT * BLOCK_SIZE) // 2 - 40
            ))
            self.screen.blit(paused_text, text_rect)
            
    def draw_controls(self):
        """Draw control instructions"""
        controls = [
            "Controls:",
            "P: Pause/Resume",
            "A: Toggle Auto-play",
            "Up/Down: Speed",
            "R: Restart",
            "Esc/Q: Quit"
        ]
        
        for i, text in enumerate(controls):
            rendered_text = self.font.render(text, True, (200, 200, 200))
            self.screen.blit(
                rendered_text, 
                (20, BOARD_OFFSET_Y + BOARD_HEIGHT * BLOCK_SIZE - 150 + i * 25)
            )
    
    def handle_events(self):
        """Process pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            if event.type == pygame.KEYDOWN:
                # Exit game
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                    
                # Pause game
                if event.key == pygame.K_p:
                    self.paused = not self.paused
                    
                # Toggle auto-play
                if event.key == pygame.K_a:
                    self.auto_play = not self.auto_play
                    
                # Adjust speed
                if event.key == pygame.K_UP:
                    self.speed = min(60, self.speed + 1)
                    self.move_delay = 1.0 / self.speed
                if event.key == pygame.K_DOWN:
                    self.speed = max(1, self.speed - 1)
                    self.move_delay = 1.0 / self.speed
                    
                # Restart game
                if event.key == pygame.K_r:
                    self.game = TetrisEngine(width=BOARD_WIDTH, height=BOARD_HEIGHT, buffer_zone=BUFFER_ZONE)
                    if self.bot:
                        self.bot.game = self.game
                    
                # Manual controls (only active if not in auto-play)
                if not self.auto_play and not self.game.game_over:
                    if event.key == pygame.K_LEFT:
                        self.game.move_left()
                    if event.key == pygame.K_RIGHT:
                        self.game.move_right()
                    if event.key == pygame.K_DOWN:
                        self.game.soft_drop()
                    if event.key == pygame.K_SPACE:
                        self.game.hard_drop()
                    if event.key == pygame.K_z:
                        self.game.rotate_left()
                    if event.key == pygame.K_x:
                        self.game.rotate_right()
                    if event.key == pygame.K_c:
                        self.game.hold()
                        
        return True
        
    def update(self):
        """Update game state"""
        if self.paused or self.game.game_over:
            return
            
        current_time = time.time()
        if current_time - self.last_move_time >= self.move_delay:
            self.last_move_time = current_time
            
            # Let the bot make a move in auto-play mode
            if self.auto_play and self.bot:
                self.bot.make_move()
            else:
                # Auto gravity
                if not self.game.soft_drop():
                    self.game.hard_drop()
                    
    def run(self):
        """Main game loop"""
        running = True
        clock = pygame.time.Clock()
        
        while running:
            # Process events
            running = self.handle_events()
            
            # Update game state
            self.update()
            
            # Draw everything
            self.screen.fill((0, 0, 0))
            self.draw_board()
            self.draw_next_piece()
            self.draw_held_piece()
            self.draw_stats()
            self.draw_controls()
            
            # Update display
            pygame.display.flip()
            
            # Cap at 60 FPS
            clock.tick(60)
            
        pygame.quit()


def main():
    # Create the Tetris game instance
    game = TetrisEngine(width=BOARD_WIDTH, height=BOARD_HEIGHT, buffer_zone=BUFFER_ZONE)
    
    # Create the bot with the new implementation (doesn't need an evaluator anymore)
    bot = TetrisBot(game)
    
    # Create and run the visualizer with faster speed
    visualizer = TetrisVisualizer(game, bot, speed=12, auto_play=True)
    visualizer.run()


if __name__ == "__main__":
    main()