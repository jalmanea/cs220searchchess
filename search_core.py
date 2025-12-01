import dataclasses
from dataclasses import dataclass
from typing import List, Callable, Iterable, Optional, Tuple
import chess

MoveOrderingFn = Callable[[chess.Board, Iterable[chess.Move]], List[chess.Move]]


@dataclass
class SearchResult:
    found: bool
    nodes_expanded: int
    solution_moves: List[chess.Move]
    solution_depth: int
    reached_max_nodes: bool


def is_goal(board: chess.Board) -> bool:
    """
    Goal: the side to move is checkmated.
    For mate puzzles we assume the attacking side is at the root, and the goal
    is to reach a position where the defender is checkmated.
    """
    return board.is_checkmate()


def generate_successors(
    board: chess.Board,
    move_order_fn: Optional[MoveOrderingFn] = None,
) -> List[Tuple[chess.Move, chess.Board]]:
    """
    Generate (move, new_board) successors from a board.
    """
    legal_moves = list(board.legal_moves)
    if move_order_fn is not None:
        legal_moves = move_order_fn(board, legal_moves)

    successors: List[Tuple[chess.Move, chess.Board]] = []
    for move in legal_moves:
        new_board = board.copy(stack=False)
        new_board.push(move)
        successors.append((move, new_board))
    return successors


def format_move_sequence(board: chess.Board, moves: List[chess.Move]) -> List[str]:
    """
    Convert a list of moves into SAN strings starting from a given board.
    """
    temp = board.copy(stack=False)
    out: List[str] = []
    for mv in moves:
        out.append(temp.san(mv))
        temp.push(mv)
    return out