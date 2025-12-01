import argparse
import csv
import time
from typing import List, Dict, Any, Tuple

import chess

from uninformed import bfs, dfs_depth_limited, uniform_cost_search
from informed import astar_search, ALL_HEURISTICS

MAX_DEPTH = 5


def infer_solution_depth_from_themes(themes: str, default_max: int = 5) -> int:
    """
    Themes field looks like:
    "['kingsideAttack' 'mate' 'mateIn1' 'oneMove' 'opening']"

    Infer mate depth from 'mateInX' tag.
    """
    for d in range(1, default_max + 1):
        if f"mateIn{d}" in themes:
            return d
    return default_max


def parse_dataset_row(row: Dict[str, str]) -> Tuple[str, int, Dict[str, Any]]:
    """
    Normalize a row from either:
      - custom CSV: fen,solution_depth,description
      - Lichess-style CSV: PuzzleId,FEN,Moves,Rating,Themes

    Returns:
      fen, solution_depth, meta_dict
    """
    # Lichess-style
    if "FEN" in row and "PuzzleId" in row:
        fen = row["FEN"].strip()
        themes = row.get("Themes", "")
        solution_depth = infer_solution_depth_from_themes(themes)

        meta = {
            "puzzle_id": row.get("PuzzleId", ""),
            "rating": row.get("Rating", ""),
            "moves": row.get("Moves", ""),
            "themes": themes,
            # description: reuse Themes string for human-readable context
            "description": themes,
        }
        return fen, solution_depth, meta

    # Old custom format
    if "fen" in row:
        fen = row["fen"].strip()
        solution_depth = int(row["solution_depth"])
        desc = row.get("description", "")
        meta = {
            "puzzle_id": "",
            "rating": "",
            "moves": "",
            "themes": "",
            "description": desc,
        }
        return fen, solution_depth, meta

    raise ValueError("Unrecognized dataset format; expected either fen/solution_depth or PuzzleId/FEN/… columns.")


def run_for_position(
    fen: str,
    solution_depth: int,
    meta: Dict[str, Any],
    max_nodes: int,
    verbose: bool,
) -> List[Dict[str, Any]]:
    board = chess.Board(fen)

    # generous ply bound: mate depth in moves -> at most 2 * depth plies, plus some slack
    max_depth_plies = min(2 * solution_depth + 4, MAX_DEPTH)

    results: List[Dict[str, Any]] = []

    def record_result(name: str, heuristic_name: str, res, runtime: float) -> None:
        row = {
            "fen": fen,
            "solution_depth": solution_depth,
            "algo": name,
            "heuristic": heuristic_name,
            "found": int(res.found),
            "solution_length": res.solution_depth,
            "nodes_expanded": res.nodes_expanded,
            "runtime": runtime,
            "reached_max_nodes": int(res.reached_max_nodes),
        }
        # attach puzzle metadata
        row.update(meta)
        results.append(row)

    # Uninformed algorithms
    algo_specs = [
        ("bfs", bfs),
        ("dfs", dfs_depth_limited),
        ("ucs", uniform_cost_search),
    ]

    for name, func in algo_specs:
        if verbose:
            print(f"[{meta.get('puzzle_id', '') or fen}] running {name}")
        start = time.perf_counter()
        res = func(board, max_depth=max_depth_plies, max_nodes=max_nodes)
        end = time.perf_counter()
        record_result(name, "na", res, end - start)

    # A* with all heuristics
    for h_name, h_fn in ALL_HEURISTICS.items():
        if verbose:
            print(f"[{meta.get('puzzle_id', '') or fen}] running astar with {h_name}")
        start = time.perf_counter()
        res = astar_search(
            board,
            heuristic=h_fn,
            max_depth=max_depth_plies,
            max_nodes=max_nodes,
        )
        end = time.perf_counter()
        record_result("astar", h_name, res, end - start)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="CSV with either: fen,solution_depth,description or PuzzleId,FEN,Moves,Rating,Themes",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=100000,
        help="Maximum nodes expanded per run",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after solving at most N puzzles (default: all puzzles)."
    )

    args = parser.parse_args()

    input_rows: List[Dict[str, Any]] = []
    with open(args.dataset, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            input_rows.append(row)

    all_results: List[Dict[str, Any]] = []

    puzzle_counter = 0
    for i, row in enumerate(input_rows):
        fen, solution_depth, meta = parse_dataset_row(row)

        if args.verbose:
            pid = meta.get("puzzle_id", "")
            desc = meta.get("description", "")
            print(f"=== puzzle {i} ===")
            if pid:
                print(f"PuzzleId: {pid}")
            if desc:
                print(desc)

        per_puzzle_results = run_for_position(
            fen=fen,
            solution_depth=solution_depth,
            meta=meta,
            max_nodes=args.max_nodes,
            verbose=args.verbose,
        )
        all_results.extend(per_puzzle_results)
        puzzle_counter += 1
        if args.limit is not None and puzzle_counter >= args.limit:
            if args.verbose:
                print(f"Reached limit of {args.limit} puzzles; stopping.")
            break


    # unified schema: supports both custom and Lichess-style datasets
    fieldnames = [
        "puzzle_id",
        "rating",
        "moves",
        "themes",
        "fen",
        "solution_depth",
        "description",
        "algo",
        "heuristic",
        "found",
        "solution_length",
        "nodes_expanded",
        "runtime",
        "reached_max_nodes",
    ]

    with open(args.out, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    if args.verbose:
        print(f"Wrote {len(all_results)} rows to {args.out}")


if __name__ == "__main__":
    main()