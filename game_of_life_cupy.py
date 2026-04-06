"""Conway's Game of Life simulation using CuPy (GPU-accelerated).

Usage:
    python game_of_life_cupy.py [--width W] [--height H] [--steps N]
                                [--chunk-size C] [--output PATH] [--seed S]

The simulation runs entirely in a dedicated CuPy (CUDA) stream.  A pair of
alternating output streams handle non-blocking device-to-host (D2H) transfers
to pinned host memory.  After every chunk the D2H stream is synchronised and
the CPU writes the accumulated frames to disk *while* the GPU simulation
stream is already processing the next chunk.
"""

import argparse
import contextlib
from typing import Optional

import numpy as np

try:
    import cupy as cp
    from cupyx.profiler import time_range as _nvtx_range
    _CUPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    cp = None  # type: ignore[assignment]
    _nvtx_range = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False


def _profiling_range(name: str):
    """Return a context manager that annotates *name* as an NVTX range.

    When the process is launched via ``nsys profile`` (NVIDIA Nsight Systems),
    each entered range appears as a labelled band in the timeline, making it
    easy to distinguish simulation-kernel time from D2H-transfer time.

    ``cupyx.profiler.time_range`` is used when available; otherwise a no-op
    ``contextlib.nullcontext()`` is returned so the code path is unchanged.
    """
    if _nvtx_range is not None:
        return _nvtx_range(name)
    return contextlib.nullcontext()


def next_generation_cupy(grid: "cp.ndarray") -> "cp.ndarray":
    """Return the next generation of *grid* according to Conway's rules.

    Mirrors :func:`game_of_life.next_generation` but operates on a CuPy
    array so that the computation runs on the GPU.

    Parameters
    ----------
    grid:
        2-D CuPy uint8 array where 1 means alive and 0 means dead.

    Returns
    -------
    cp.ndarray
        New grid of the same shape and dtype.
    """
    row_up   = cp.roll(grid, -1, axis=0)
    row_same = grid
    row_down = cp.roll(grid,  1, axis=0)
    neighbours = (
        cp.roll(row_up,   -1, axis=1) + cp.roll(row_up,   0, axis=1) + cp.roll(row_up,   1, axis=1)
        + cp.roll(row_same, -1, axis=1)                                + cp.roll(row_same, 1, axis=1)
        + cp.roll(row_down, -1, axis=1) + cp.roll(row_down, 0, axis=1) + cp.roll(row_down, 1, axis=1)
    )

    return cp.where(
        (grid == 1) & ((neighbours == 2) | (neighbours == 3)),
        1,
        cp.where((grid == 0) & (neighbours == 3), 1, 0),
    ).astype(cp.uint8)


def simulate_cupy(
    width: int,
    height: int,
    steps: int,
    chunk_size: int,
    output: str,
    seed: Optional[int] = None,
    profile: bool = False,
) -> None:
    """Run the GPU simulation and save every generation to *output*.

    Architecture
    ------------
    * **sim_stream** – CuPy CUDA stream that owns all simulation kernels.
    * **out_streams[0/1]** – a pair of CUDA streams that alternate across
      chunks.  Each stream performs a non-blocking D2H transfer of its chunk
      into pinned (page-locked) host memory after the simulation kernels for
      that chunk have finished.
    * **CPU / GPU overlap** – after enqueuing the kernels for chunk *N* and
      scheduling the D2H for chunk *N* on the current output stream, the
      previous output stream (chunk *N-1*) is synchronised and its data is
      written to disk.  Because the GPU's ``sim_stream`` is asynchronous,
      chunk *N* is already executing on the GPU while the CPU handles the
      disk write for chunk *N-1*.

    Parameters
    ----------
    width, height:
        Grid dimensions in cells.
    steps:
        Number of generations after the initial state.
    chunk_size:
        Generations per GPU chunk.
    output:
        Destination ``.npy`` file.
    seed:
        Optional RNG seed for reproducibility.
    profile:
        When ``True``, record ``cp.cuda.Event`` timestamps around each chunk's
        simulation kernels and D2H transfer, print per-chunk timings, and
        print a total summary at the end.  Has no effect when ``False``
        (default).

        ``cupyx.profiler.time_range`` NVTX annotations are always inserted
        around each kernel batch and each D2H transfer, regardless of this
        flag, so the timeline is visible in ``nsys profile`` without any extra
        argument::

            nsys profile python game_of_life_cupy.py --steps 100
    """
    if not _CUPY_AVAILABLE:
        raise RuntimeError(
            "CuPy is not installed.  "
            "Install a CUDA-compatible build, e.g.: pip install cupy-cuda12x"
        )

    rng = np.random.default_rng(seed)
    initial = rng.integers(0, 2, size=(height, width), dtype=np.uint8)

    total_frames = steps + 1
    history = np.empty((total_frames, height, width), dtype=np.uint8)
    history[0] = initial

    if steps == 0:
        np.save(output, history)
        print(
            f"\nDone. Simulation saved to '{output}' "
            f"(1 frame, grid {height}×{width})."
        )
        return

    # One simulation stream; two output streams for double-buffered D2H.
    sim_stream = cp.cuda.Stream(non_blocking=True)
    out_streams = [
        cp.cuda.Stream(non_blocking=True),
        cp.cuda.Stream(non_blocking=True),
    ]

    # Pinned (page-locked) host memory enables DMA-based D2H transfers that
    # run independently of the CPU, allowing true GPU / CPU overlap.
    item_bytes = np.dtype(np.uint8).itemsize
    n = chunk_size * height * width
    pinned_mems = [
        cp.cuda.alloc_pinned_memory(n * item_bytes),
        cp.cuda.alloc_pinned_memory(n * item_bytes),
    ]
    # np.frombuffer requires an explicit count when wrapping a
    # CuPy PinnedMemoryPointer, because the pointer does not always expose
    # its size through the Python buffer protocol.
    pinned_arrs = [
        np.frombuffer(pinned_mems[i], dtype=np.uint8, count=n).reshape(
            chunk_size, height, width
        )
        for i in range(2)
    ]

    # Upload the initial grid and allocate one GPU chunk buffer *per slot*.
    # Using separate buffers per slot is essential for true stream overlap:
    # if both slots shared a single buffer, the CUDA driver would see a
    # read-after-write hazard between the D2H transfer on out_streams[prev]
    # and the simulation kernels on sim_stream, and would serialise the two
    # streams to preserve correctness — defeating the double-buffering.
    with sim_stream:
        grid_gpu = cp.asarray(initial)
    chunk_gpus = [
        cp.empty((chunk_size, height, width), dtype=cp.uint8),
        cp.empty((chunk_size, height, width), dtype=cp.uint8),
    ]

    # Per-buffer CUDA event pairs for optional profiling.
    # sim_evs[i]  = (start, end) events bracketing the simulation kernels.
    # d2h_evs[i]  = (start, end) events bracketing the D2H transfer.
    # Both slots are reused on each double-buffer cycle; timing is read back
    # after the corresponding out_stream is synchronised.
    if profile:
        sim_evs = [
            (cp.cuda.Event(), cp.cuda.Event()),
            (cp.cuda.Event(), cp.cuda.Event()),
        ]
        d2h_evs = [
            (cp.cuda.Event(), cp.cuda.Event()),
            (cp.cuda.Event(), cp.cuda.Event()),
        ]
        total_sim_ms = 0.0
        total_d2h_ms = 0.0

    frame = 1
    buf_idx = 0
    # Pending tracks the D2H-in-flight chunk so it can be written to disk
    # during the *next* iteration while the GPU works on that next chunk.
    pending: Optional[tuple] = None  # (buf_idx, frame_start, chunk_end, chunk_frames)

    while frame <= steps:
        chunk_end = min(frame + chunk_size, steps + 1)
        chunk_frames = chunk_end - frame

        print(
            f"Computing generations {frame} – {chunk_end - 1} "
            f"(chunk size {chunk_size})…"
        )

        # ── 1. Enqueue simulation kernels on sim_stream (returns immediately).
        with sim_stream:
            if profile:
                sim_evs[buf_idx][0].record()  # kernel start (current stream)
            with _profiling_range(f"kernel {frame}-{chunk_end - 1}"):
                for i in range(chunk_frames):
                    grid_gpu = next_generation_cupy(grid_gpu)
                    chunk_gpus[buf_idx][i] = grid_gpu
            if profile:
                sim_evs[buf_idx][1].record()  # kernel end (current stream)

        # ── 2. Enqueue non-blocking D2H transfer on the current output stream.
        #       The output stream waits for sim_stream to record an event,
        #       ensuring the transfer starts only after the kernels complete.
        out_stream = out_streams[buf_idx]
        out_stream.wait_event(sim_stream.record())
        if profile:
            d2h_evs[buf_idx][0].record(out_stream)  # transfer start
        with _profiling_range(f"D2H {frame}-{chunk_end - 1}"):
            chunk_gpus[buf_idx][:chunk_frames].get(
                out=pinned_arrs[buf_idx][:chunk_frames],
                blocking=False,
                stream=out_stream,
            )
        if profile:
            d2h_evs[buf_idx][1].record(out_stream)  # transfer end

        # ── 3. While the GPU runs the kernels above, handle the *previous*
        #       chunk: sync its D2H stream (GPU-side work done) then write to
        #       disk on the CPU.  This is the CPU / GPU overlap: the GPU's
        #       sim_stream is computing chunk N while the CPU writes chunk N-1.
        if pending is not None:
            prev_buf, prev_start, prev_end, prev_frames = pending
            # Synchronise ensures the D2H transfer for this chunk is complete.
            out_streams[prev_buf].synchronize()
            if profile:
                sim_ms = cp.cuda.get_elapsed_time(
                    sim_evs[prev_buf][0], sim_evs[prev_buf][1]
                )
                d2h_ms = cp.cuda.get_elapsed_time(
                    d2h_evs[prev_buf][0], d2h_evs[prev_buf][1]
                )
                total_sim_ms += sim_ms
                total_d2h_ms += d2h_ms
                print(
                    f"  [profile] chunk {prev_start}–{prev_end - 1}: "
                    f"kernel={sim_ms:.3f} ms, D2H={d2h_ms:.3f} ms"
                )
            history[prev_start:prev_end] = pinned_arrs[prev_buf][:prev_frames]
            np.save(output, history[:prev_end])
            print(f"  Saved {prev_end} frame(s) to '{output}'.")

        pending = (buf_idx, frame, chunk_end, chunk_frames)
        frame = chunk_end
        buf_idx = 1 - buf_idx  # Alternate between stream 0 and stream 1.

    # Flush the last pending chunk once the GPU has finished.
    if pending is not None:
        prev_buf, prev_start, prev_end, prev_frames = pending
        out_streams[prev_buf].synchronize()
        if profile:
            sim_ms = cp.cuda.get_elapsed_time(
                sim_evs[prev_buf][0], sim_evs[prev_buf][1]
            )
            d2h_ms = cp.cuda.get_elapsed_time(
                d2h_evs[prev_buf][0], d2h_evs[prev_buf][1]
            )
            total_sim_ms += sim_ms
            total_d2h_ms += d2h_ms
            print(
                f"  [profile] chunk {prev_start}–{prev_end - 1}: "
                f"kernel={sim_ms:.3f} ms, D2H={d2h_ms:.3f} ms"
            )
        history[prev_start:prev_end] = pinned_arrs[prev_buf][:prev_frames]
        np.save(output, history[:prev_end])
        print(f"  Saved {prev_end} frame(s) to '{output}'.")

    if profile:
        print(f"\n[profile] Total kernel time : {total_sim_ms:.3f} ms")
        print(f"[profile] Total D2H time    : {total_d2h_ms:.3f} ms")
        print(f"[profile] Total GPU+D2H     : {total_sim_ms + total_d2h_ms:.3f} ms")

    print(
        f"\nDone. Simulation saved to '{output}' "
        f"({total_frames} frames, grid {height}×{width})."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conway's Game of Life simulator (CuPy / GPU-backed)."
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
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help=(
            "Print per-chunk CUDA kernel and D2H transfer timings using "
            "cp.cuda.Event (default: off)."
        ),
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

    simulate_cupy(
        width=args.width,
        height=args.height,
        steps=args.steps,
        chunk_size=args.chunk_size,
        output=args.output,
        seed=args.seed,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
