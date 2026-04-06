# GameOfLife

A NumPy-based (CPU) and CuPy-based (GPU) implementation of Conway's Game of Life.

## Overview

This project simulates [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) on a toroidal (wrap-around) grid.  Two backends are provided:

| Backend | File | Requires |
|---|---|---|
| NumPy (CPU) | `game_of_life.py` | `numpy` |
| CuPy (GPU) | `game_of_life_cupy.py` | `cupy` + CUDA |

Simulation results are saved as a NumPy `.npy` file and can optionally be rendered as an MPEG animation.

## Requirements

```
pip install -r requirements.txt
```

For GPU acceleration install a CuPy build matching your CUDA version:

```bash
pip install cupy-cuda12x   # CUDA 12.x
pip install cupy-cuda11x   # CUDA 11.x
```

`ffmpeg` must also be installed on the system `PATH` if you want to render animations.

## Usage

### Run the simulation (CPU / NumPy)

```bash
python game_of_life.py [--width W] [--height H] [--steps N] \
                       [--chunk-size C] [--output PATH] [--seed S]
```

| Argument | Default | Description |
|---|---|---|
| `--width` | 100 | Number of columns in the grid |
| `--height` | 100 | Number of rows in the grid |
| `--steps` | 100 | Number of generations to simulate after the initial state |
| `--chunk-size` | 10 | Generations per chunk before flushing results to disk |
| `--output` | `simulation.npy` | Output `.npy` file path |
| `--seed` | None | Random seed for reproducibility |

**Example:**
```bash
python game_of_life.py --width 200 --height 200 --steps 500 --seed 42
```

### Run the simulation (GPU / CuPy)

```bash
python game_of_life_cupy.py [--width W] [--height H] [--steps N] \
                             [--chunk-size C] [--output PATH] [--seed S]
```

The arguments are identical to the NumPy version.  The CuPy simulator uses a
**double-buffered streaming pipeline**:

* One CUDA stream (`sim_stream`) drives all simulation kernels.
* Two alternating CUDA streams (`out_streams[0/1]`) perform non-blocking
  device-to-host (D2H) transfers into pinned host memory after each chunk.
* After enqueuing the simulation and D2H work for chunk *N*, the main thread
  synchronises the output stream for chunk *N-1* and writes that chunk to disk
  **while the GPU is already computing chunk *N***, overlapping GPU computation
  with CPU I/O.

**Example:**
```bash
python game_of_life_cupy.py --width 200 --height 200 --steps 500 --seed 42
```

### Render an animation

```bash
python render_animation.py [--input PATH] [--output PATH] [--fps N] \
                            [--dpi N] [--cmap NAME]
```

| Argument | Default | Description |
|---|---|---|
| `--input` | `simulation.npy` | Path to the `.npy` simulation file |
| `--output` | `animation.mp4` | Path for the output `.mp4` file |
| `--fps` | 15 | Frames per second |
| `--dpi` | 100 | Resolution (DPI) of each frame |
| `--cmap` | `binary` | Matplotlib colormap name |

**Example:**
```bash
python render_animation.py --input simulation.npy --output animation.mp4 --fps 30
```

## Running Tests

```bash
pip install pytest
pytest
```
