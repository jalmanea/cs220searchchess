from collections import deque
from typing import Optional, List, Callable, Iterable, Tuple
import heapq
import chess

from search_core import SearchResult, is_goal, generate_successors
from orderings import TranspositionTable, basic_move_ordering

MoveOrderingFn = Callable[[chess.Board, Iterable[chess.Move]], List[chess.Move]]


def bfs(
    initial_board: chess.Board,
    max_depth: int = 10,
    max_nodes: int = 100000,
    move_order_fn: Optional[MoveOrderingFn] = basic_move_ordering,
    use_tt: bool = True,
) -> SearchResult:
    queue = deque()
    queue.append((initial_board, [], 0))  # board, path_moves, depth
    nodes_expanded = 0
    reached_limit = False

    tt = TranspositionTable() if use_tt else None
    if tt is not None:
        tt.seen(initial_board.fen(), 0)

    while queue:
        board, path, depth = queue.popleft()
        nodes_expanded += 1

        if is_goal(board):
            return SearchResult(True, nodes_expanded, path, depth, reached_limit)

        if nodes_expanded >= max_nodes:
            reached_limit = True
            break

        if depth >= max_depth:
            continue

        for move, child in generate_successors(board, move_order_fn):
            new_depth = depth + 1
            fen = child.fen()
            if tt is not None:
                if tt.seen(fen, new_depth):
                    continue
            queue.append((child, path + [move], new_depth))

    return SearchResult(False, nodes_expanded, [], -1, reached_limit)


def dfs_depth_limited(
    initial_board: chess.Board,
    max_depth: int = 5,
    max_nodes: int = 100000,
    move_order_fn: Optional[MoveOrderingFn] = basic_move_ordering,
    use_tt: bool = True,
) -> SearchResult:
    nodes_expanded = 0
    reached_limit = False

    tt = TranspositionTable() if use_tt else None

    def dfs(board: chess.Board, path: List[chess.Move], depth: int) -> Optional[SearchResult]:
        nonlocal nodes_expanded, reached_limit

        if nodes_expanded >= max_nodes:
            reached_limit = True
            return None

        if tt is not None:
            fen = board.fen()
            if tt.seen(fen, depth):
                return None

        nodes_expanded += 1

        if is_goal(board):
            return SearchResult(True, nodes_expanded, path, depth, reached_limit)

        if depth >= max_depth:
            return None

        for move, child in generate_successors(board, move_order_fn):
            res = dfs(child, path + [move], depth + 1)
            if res is not None and res.found:
                return res
        return None

    res = dfs(initial_board, [], 0)
    if res is not None:
        return res
    return SearchResult(False, nodes_expanded, [], -1, reached_limit)


def uniform_cost_search(
    initial_board: chess.Board,
    max_depth: int = 10,
    max_nodes: int = 100000,
    move_order_fn: Optional[MoveOrderingFn] = basic_move_ordering,
    use_tt: bool = True,
) -> SearchResult:
    """
    Cost is 1 per ply.
    This is Dijkstra on the state space.
    """
    root_fen = initial_board.fen()
    root_color = initial_board.turn

    pq: List[Tuple[int, int, chess.Board, List[chess.Move]]] = []
    counter = 0
    heapq.heappush(pq, (0, counter, initial_board, []))

    best_g = {root_fen: 0}
    nodes_expanded = 0
    reached_limit = False

    tt = TranspositionTable() if use_tt else None
    if tt is not None:
        tt.seen(root_fen, 0)

    while pq:
        g, _, board, path = heapq.heappop(pq)
        depth = g  # since cost 1 per move

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
            fen = child.fen()

            if tt is not None and tt.seen(fen, new_g):
                continue

            old_g = best_g.get(fen)
            if old_g is not None and old_g <= new_g:
                continue

            best_g[fen] = new_g
            counter += 1
            heapq.heappush(pq, (new_g, counter, child, path + [move]))

    return SearchResult(False, nodes_expanded, [], -1, reached_limit)