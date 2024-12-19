import chess
from transformers import LlamaTokenizerFast, LlamaForCausalLM
from llmimic import logger

class ChessInstance:
  """I wrote this chess class that prompts an LLM for moves and manages the various aspects of a game:
  basically encapsulating everything you would need to play chess to varying degrees in a human-like way.
  I wrote this entire thing in an afternoon. It hasn't been updated since then, while the main code has 
  undergone three rewrites at this point. I don't remember what half of this code does off-rip, or why the design 
  decisions were made as they were. Until I can refactor and rewrite some of this, I apologize for not being 
  100% on some things.
  """
  def __init__(self, difficulty_level="normal"):
    self.board = chess.Board()
    self.moves_uci = []
    self.difficulty_level = difficulty_level

    self.tokenizer = LlamaTokenizerFast.from_pretrained('lazy-guy12/chess-llama')
    self.model = LlamaForCausalLM.from_pretrained('lazy-guy12/chess-llama')

    self.difficulty_settings = {
        "very_easy": {"temperature": 1.0, "top_k": 100, "retries": 100},
        "easy": {"temperature": 0.8, "top_k": 75, "retries": 75},
        "normal": {"temperature": 0.6, "top_k": 50, "retries": 50},
        "intermediate": {"temperature": 0.4, "top_k": 30, "retries": 30},
        "hard": {"temperature": 0.2, "top_k": 20, "retries": 20},
    }
    
  def get_difficulty_params(self):
    """Pretty sure this is a roundabout way of getting difficulty, which is setting difficultly (the floor is made of floor).

    Returns:
        dict: The difficulty parameters.
    """
    return self.difficulty_settings.get(self.difficulty_level, self.difficulty_settings["normal"])

  def _play_llama(self) -> str:
    """Internal function for prompting the chess LLM for the next move.

    Returns:
      str: Next move in UCI format.
    """
    input_text = "1-0 " + " ".join(self.moves_uci)
    logger.info(f"AI Input Moves (UCI): {input_text}")
    inputs = self.tokenizer(input_text, return_tensors="pt")
    
    params = self.get_difficulty_params()
    temperature = params["temperature"]
    top_k = params["top_k"]
    max_retries = params["retries"]

    for attempt in range(max_retries):
      generated = self.model.generate(
          input_ids=inputs['input_ids'],
          max_new_tokens=5,
          do_sample=True,
          top_k=top_k,
          temperature=temperature,
      )

      generated_move_full = self.tokenizer.decode(generated[0], skip_special_tokens=True)
      logger.debug(f"Attempt {attempt + 1} - Full Generated Move: {generated_move_full}")

      move_index = len(self.moves_uci) + 1
      generated_move = generated_move_full.split()[move_index] if len(generated_move_full.split()) > move_index else ""
      logger.debug(f"Attempt {attempt + 1} - Trimmed Move (UCI): {generated_move}")
      
      try:
        generated_move_obj = chess.Move.from_uci(generated_move)
        if self.board.is_legal(generated_move_obj):
          return generated_move
        else:
          logger.debug(f"Move {generated_move} is illegal on the current board.")
      except ValueError:
        logger.debug(f"Invalid UCI move format: {generated_move}")

    return "0000"

  def get_valid_moves(self) -> dict:
    """Returns valid moves as a dictionary.
    I don't even remember why we did this?

    Returns:
        dict: _description_
    """
    return {move.uci(): move for move in self.board.legal_moves}
  
  def get_board(self):
    """Getter function for board.

    Returns:
        _type_: Returns the board, but I don't remember why?
    """
    return self.board
  
  def check_game_over(self) -> bool:
    """Checks if game is over.

    Returns:
        bool: True if the game is over (I think).
    """
    if self.board.is_game_over():
      logger.info("Game Over!")
      return True
    else:
      return False
      
  def _validate_and_apply_move(self, move_uci: str) -> bool:
    """Validates and applies moves, probably.

    Args:
        move_uci (str): The move as a string in UCI format.

    Returns:
        bool: True is move applied, I think.
    """
    try:
      move_obj = chess.Move.from_uci(move_uci)
      if move_obj in self.board.legal_moves:
        self.board.push(move_obj)
        self.moves_uci.append(move_uci)
        logger.info(f"Move applied: {move_uci} (UCI)")
        return True
      else:
        logger.warning(f"Illegal move attempted: {move_uci}")
        return False
    except ValueError:
      logger.error(f"Invalid UCI format move: {move_uci}")
      return False

  def submit_player_move(self, user_move: str) -> bool:
    """Attempts to play player move.

    Args:
        user_move (str): The user move as a string in UCI format.

    Returns:
        bool: Whether the player succeeded at moving.
    """
    print("Player is making move.")
    success = self._validate_and_apply_move(user_move)
    if not success:
      logger.warning(f"Player attempted an illegal move: {user_move}")
    return success

  def prompt_ai_move(self) -> tuple[bool, str]:
    """Encapsulation of move prompting and validation.

    Returns:
        tuple[bool, str]: I think bool is whether it succeeded and string is the move in UCI format either way.
    """
    print("AI is making move.")
    ai_move = self._play_llama()

    success = self._validate_and_apply_move(ai_move)
    if not success:
      logger.error(f"AI attempted an illegal move: {ai_move}")
    return success, ai_move

  def get_move_history(self) -> list:
    """Get the current move history.

    Returns:
        list: The current move history, probably?
    """
    return [move.uci() for move in self.board.move_stack]
  
  def uci_to_san(self, uci_moves):
    """
    Converts a list of UCI moves into Standard Algebraic Notation (SAN) using the current board state.
    
    Args:
      uci_moves (list of str): A list of moves in UCI format (e.g., ['e2e4', 'e7e5']).
    
    Returns:
      list of str: A list of moves in SAN format.
    """
    san_moves = []
    for uci in uci_moves:
      try:
        move = self.board.parse_uci(uci)
        san_moves.append(self.board.san(move))
        self.board.push(move)
      except ValueError as e:
        print(f"Invalid UCI move '{uci}': {e}")
        break
    return san_moves