"""Conway's Game of Life simulation using NumPy.

Usage:
    python game_of_life.py [--width W] [--height H] [--steps N]
                           [--chunk-size C] [--output PATH] [--seed S]

The simulation is computed in chunks of ``--chunk-size`` steps. After every
chunk the accumulated results are saved (appended) to a single NumPy ``.npy``
file so that memory usage stays bounded even for long simulations.
"""

import argparse
import numpy as np


def next_generation(grid: np.ndarray) -> np.ndarray:
    """Return the next generation of *grid* according to Conway's rules.

    Uses ``numpy.roll`` for wrap-around (toroidal) boundary conditions so that
    no cell is ever on an "edge".

    Parameters
    ----------
    grid:
        2-D boolean/uint8 array where 1 means alive and 0 means dead.

    Returns
    -------
    np.ndarray
        New grid of the same shape and dtype.
    """
    # Count the live neighbours of every cell using 8-directional wrap-around.
    # Pre-compute row-shifted arrays once to halve the number of roll calls.
    row_up   = np.roll(grid, -1, axis=0)
    row_same = grid
    row_down = np.roll(grid,  1, axis=0)
    neighbours = (
        np.roll(row_up,   -1, axis=1) + np.roll(row_up,   0, axis=1) + np.roll(row_up,   1, axis=1)
        + np.roll(row_same, -1, axis=1)                                + np.roll(row_same, 1, axis=1)
        + np.roll(row_down, -1, axis=1) + np.roll(row_down, 0, axis=1) + np.roll(row_down, 1, axis=1)
    )

    # Apply Conway's rules:
    #   - A live cell survives if it has 2 or 3 neighbours.
    #   - A dead cell becomes alive if it has exactly 3 neighbours.
    return np.where(
        (grid == 1) & ((neighbours == 2) | (neighbours == 3)),
        1,
        np.where((grid == 0) & (neighbours == 3), 1, 0),
    ).astype(np.uint8)


def simulate(
    width: int,
    height: int,
    steps: int,
    chunk_size: int,
    output: str,
    seed: int | None = None,
) -> None:
    """Run the simulation and save every generation to *output*.

    The initial state is a random grid. Generations are collected in chunks of
    ``chunk_size`` and written to *output* (overwriting each time) so that the
    full history is always on disk.

    Parameters
    ----------
    width:
        Number of columns in the grid.
    height:
        Number of rows in the grid.
    steps:
        Total number of generations to simulate (the initial state counts as
        generation 0 and is always included, giving ``steps + 1`` frames).
    chunk_size:
        Number of generations to compute before flushing results to disk.
    output:
        Path to the output ``.npy`` file.
    seed:
        Optional random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    grid = rng.integers(0, 2, size=(height, width), dtype=np.uint8)

    # history shape: (steps + 1, height, width)
    total_frames = steps + 1
    history = np.empty((total_frames, height, width), dtype=np.uint8)
    history[0] = grid

    if steps == 0:
        np.save(output, history)
        print(
            f"\nDone. Simulation saved to '{output}' "
            f"(1 frame, grid {height}×{width})."
        )
        return

    frame = 1
    while frame <= steps:
        chunk_end = min(frame + chunk_size, steps + 1)
        print(
            f"Computing generations {frame} – {chunk_end - 1} "
            f"(chunk size {chunk_size})…"
        )
        for i in range(frame, chunk_end):
            grid = next_generation(grid)
            history[i] = grid
        frame = chunk_end

        # Persist progress after every chunk.
        np.save(output, history[:frame])
        print(f"  Saved {frame} frame(s) to '{output}'.")

    print(
        f"\nDone. Simulation saved to '{output}' "
        f"({total_frames} frames, grid {height}×{width})."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conway's Game of Life simulator (NumPy-backed)."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=100,
        help="Number of columns in the grid (default: 100).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=100,
        help="Number of rows in the grid (default: 100).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of generations to simulate after the initial state "
             "(default: 100).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        dest="chunk_size",
        help="Number of generations to compute per chunk before saving "
             "(default: 10).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="simulation.npy",
        help="Output file path (default: simulation.npy).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive integers.")
    if args.steps < 0:
        raise ValueError("--steps must be a non-negative integer.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be a positive integer.")

    simulate(
        width=args.width,
        height=args.height,
        steps=args.steps,
        chunk_size=args.chunk_size,
        output=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
