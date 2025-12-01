from typing import Callable, Dict, Iterable, List, Tuple, Optional
import heapq
import chess

from search_core import SearchResult, is_goal, generate_successors
from orderings import TranspositionTable, basic_move_ordering

HeuristicFn = Callable[[chess.Board, chess.Color], float]


# ---------------- heuristics ----------------

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}


def null_heuristic(board: chess.Board, root_color: chess.Color) -> float:
    return 0.0


def material_heuristic(board: chess.Board, root_color: chess.Color) -> float:
    """
    Negated material balance from side to move perspective.

    score = sum(values of own pieces) - sum(values of opponent pieces)
    h = -score
    Lower h is considered better for mate puzzles.
    """
    side = board.turn
    score = 0.0
    for sq, piece in board.piece_map().items():
        value = PIECE_VALUES.get(piece.piece_type, 0.0)
        if piece.color == side:
            score += value
        else:
            score -= value
    return -score


def mate_in_one_heuristic(board: chess.Board, root_color: chess.Color) -> float:
    """
    Returns 0.0 if there is a move that delivers checkmate, otherwise 1.0.
    """
    for move in board.legal_moves:
        child = board.copy(stack=False)
        child.push(move)
        if child.is_checkmate():
            return 0.0
    return 1.0


def attacker_proximity_heuristic(board: chess.Board, root_color: chess.Color) -> float:
    """
    Minimum Chebyshev distance from any attacker piece (root_color)
    to the opponent king.
    """
    attacker_color = root_color
    defender_color = not attacker_color

    king_sq = board.king(defender_color)
    if king_sq is None:
        return 0.0  # already no king

    k_file = chess.square_file(king_sq)
    k_rank = chess.square_rank(king_sq)

    best = None

    for sq, piece in board.piece_map().items():
        if piece.color != attacker_color:
            continue
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        cheb = max(abs(f - k_file), abs(r - k_rank))
        if best is None or cheb < best:
            best = cheb

    if best is None:
        return 8.0
    return float(best)


def opponent_king_escape_heuristic(board: chess.Board, root_color: chess.Color) -> float:
    """
    Number of legal destination squares for the opponent king.
    Evaluated by switching turn to opponent and counting king moves.
    """
    attacker_color = root_color
    defender_color = not attacker_color

    king_sq = board.king(defender_color)
    if king_sq is None:
        return 0.0

    temp = board.copy(stack=False)
    temp.turn = defender_color

    escapes = 0
    for move in temp.legal_moves:
        piece = temp.piece_at(move.from_square)
        if piece is not None and piece.piece_type == chess.KING and piece.color == defender_color:
            escapes += 1
    return float(escapes)


def mobility_heuristic(board: chess.Board, root_color: chess.Color) -> float:
    """
    Number of legal moves for side to move.
    """
    return float(sum(1 for _ in board.legal_moves))


def checks_available_heuristic(board: chess.Board, root_color: chess.Color) -> float:
    """
    Number of moves that give check from the current position.
    """
    checks = 0
    for move in board.legal_moves:
        if board.gives_check(move):
            checks += 1
    return float(checks)


def composite_heuristic(board: chess.Board, root_color: chess.Color) -> float:
    """
    Weighted combination of proximity, king escape, mobility, and checks.
    Lower is considered better.
    """
    prox = attacker_proximity_heuristic(board, root_color)
    escape = opponent_king_escape_heuristic(board, root_color)
    mob = mobility_heuristic(board, root_color)
    checks = checks_available_heuristic(board, root_color)

    # checks are good so subtract them
    h = 1.0 * prox + 1.5 * escape + 0.5 * mob - 0.7 * checks
    # simple normalization
    return h / 10.0


ALL_HEURISTICS: Dict[str, HeuristicFn] = {
    "null_heuristic": null_heuristic,
    "material_heuristic": material_heuristic,
    "mate_in_one_heuristic": mate_in_one_heuristic,
    "attacker_proximity_heuristic": attacker_proximity_heuristic,
    "opponent_king_escape_heuristic": opponent_king_escape_heuristic,
    "mobility_heuristic": mobility_heuristic,
    "checks_available_heuristic": checks_available_heuristic,
    "composite_heuristic": composite_heuristic,
}


# ---------------- A* search ----------------

def astar_search(
    initial_board: chess.Board,
    heuristic: HeuristicFn,
    max_depth: int = 10,
    max_nodes: int = 100000,
    move_order_fn: Optional[Callable[[chess.Board, Iterable[chess.Move]], List[chess.Move]]] = basic_move_ordering,
    use_tt: bool = True,
) -> SearchResult:
    root_color = initial_board.turn
    root_fen = initial_board.fen()

    pq: List[Tuple[float, int, int, chess.Board, List[chess.Move]]] = []
    counter = 0
    g0 = 0
    h0 = heuristic(initial_board, root_color)
    heapq.heappush(pq, (g0 + h0, g0, counter, initial_board, []))

    best_g = {root_fen: 0}
    nodes_expanded = 0
    reached_limit = False

    tt = TranspositionTable() if use_tt else None
    if tt is not None:
        tt.seen(root_fen, 0)

    while pq:
        f, g, _, board, path = heapq.heappop(pq)
        depth = g

        if depth > max_depth:
            continue

        nodes_expanded += 1

        if is_goal(board):
            return SearchResult(True, nodes_expanded, path, depth, reached_limit)

        if nodes_expanded >= max_nodes:
            reached_limit = True
            break

        for move, child in generate_successors(board, move_order_fn):
            new_g = g + 1
            if new_g > max_depth:
                continue

            fen = child.fen()
            if tt is not None and tt.seen(fen, new_g):
                continue

            old_g = best_g.get(fen)
            if old_g is not None and old_g <= new_g:
                continue

            best_g[fen] = new_g
            counter += 1
            h_val = heuristic(child, root_color)
            new_f = new_g + h_val
            heapq.heappush(pq, (new_f, new_g, counter, child, path + [move]))

    return SearchResult(False, nodes_expanded, [], -1, reached_limit)