# GameOfLife

A NumPy-based implementation of Conway's Game of Life.

## Overview

This project simulates [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) on a toroidal (wrap-around) grid using NumPy for fast computation. Simulation results are saved as a NumPy `.npy` file and can optionally be rendered as an MPEG animation.

## Requirements

```
pip install -r requirements.txt
```

`ffmpeg` must also be installed on the system `PATH` if you want to render animations.

## Usage

### Run the simulation

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
