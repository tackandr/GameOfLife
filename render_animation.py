"""Render a Conway's Game of Life simulation as an MPEG animation.

The input is a ``.npy`` file produced by ``game_of_life.py`` that contains a
3-D uint8 array of shape ``(frames, height, width)``.  The output is an MPEG-4
(``.mp4``) video file.

Usage:
    python render_animation.py [--input PATH] [--output PATH] [--fps N]
                               [--dpi N] [--cmap NAME]

Requirements:
    pip install matplotlib
    # also requires ffmpeg to be installed on the system PATH
"""

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe in headless environments
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


def render(
    input_path: str,
    output_path: str,
    fps: int,
    dpi: int,
    cmap: str,
) -> None:
    """Load *input_path* and write an MPEG animation to *output_path*.

    Parameters
    ----------
    input_path:
        Path to the ``.npy`` file containing the simulation frames
        (shape ``(frames, height, width)``, dtype uint8).
    output_path:
        Destination ``.mp4`` file path.
    fps:
        Frames per second of the output video.
    dpi:
        Resolution (dots per inch) of each frame.
    cmap:
        Matplotlib colour-map name used to render the grid.
    """
    print(f"Loading simulation from '{input_path}'…")
    frames = np.load(input_path)

    if frames.ndim != 3:
        print(
            f"Error: expected a 3-D array (frames, height, width), "
            f"got shape {frames.shape}.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_frames, height, width = frames.shape
    print(
        f"  {n_frames} frame(s), grid {height}×{width}. "
        f"Encoding at {fps} fps…"
    )

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis("off")

    img = ax.imshow(
        frames[0],
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )

    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
    )

    with writer.saving(fig, output_path, dpi=dpi):
        for i, frame in enumerate(frames):
            img.set_data(frame)
            writer.grab_frame()
            if (i + 1) % max(1, n_frames // 10) == 0 or i + 1 == n_frames:
                print(f"  Encoded frame {i + 1}/{n_frames}…")

    plt.close(fig)
    print(f"\nAnimation saved to '{output_path}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Game of Life .npy simulation as an MPEG animation."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="simulation.npy",
        help="Path to the input .npy simulation file (default: simulation.npy).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="animation.mp4",
        help="Path for the output .mp4 file (default: animation.mp4).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second of the output video (default: 15).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Resolution (DPI) of each video frame (default: 100).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="binary",
        help="Matplotlib colormap name for rendering (default: binary).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.fps <= 0:
        print("Error: --fps must be a positive integer.", file=sys.stderr)
        sys.exit(1)
    if args.dpi <= 0:
        print("Error: --dpi must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    render(
        input_path=args.input,
        output_path=args.output,
        fps=args.fps,
        dpi=args.dpi,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
