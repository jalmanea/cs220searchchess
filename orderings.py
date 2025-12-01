from typing import Iterable, List, Dict
import chess


def basic_move_ordering(board: chess.Board, moves: Iterable[chess.Move]) -> List[chess.Move]:
    """
    Checks first, then captures, then the rest.
    """
    def score(move: chess.Move) -> int:
        s = 0
        if board.gives_check(move):
            s -= 20
        if board.is_capture(move):
            s -= 10
        return s

    return sorted(moves, key=score)


class TranspositionTable:
    """
    Simple transposition table keyed by FEN.
    Stores best seen depth for a position.
    If we revisit a position at a depth that is not better, we can prune.
    """

    def __init__(self) -> None:
        self._store: Dict[str, int] = {}

    def seen(self, fen: str, depth: int) -> bool:
        """
        Returns True if this FEN has been seen at a depth that is
        less than or equal to the current depth.
        """
        prev = self._store.get(fen)
        if prev is not None and prev <= depth:
            return True
        self._store[fen] = depth
        return False

    def clear(self) -> None:
        self._store.clear()