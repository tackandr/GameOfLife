"""Tests for the Conway's Game of Life simulation."""

import os
import tempfile

import numpy as np
import pytest

from game_of_life import next_generation, simulate


# ---------------------------------------------------------------------------
# next_generation tests
# ---------------------------------------------------------------------------

class TestNextGeneration:
    """Unit tests for the next_generation() function."""

    def test_all_dead_stays_dead(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        result = next_generation(grid)
        np.testing.assert_array_equal(result, grid)

    def test_block_stable(self):
        """2×2 block is a still life and should not change."""
        grid = np.zeros((6, 6), dtype=np.uint8)
        grid[2:4, 2:4] = 1
        result = next_generation(grid)
        np.testing.assert_array_equal(result, grid)

    def test_blinker_oscillates(self):
        """3-cell horizontal blinker should flip to vertical and back."""
        grid = np.zeros((5, 5), dtype=np.uint8)
        # Horizontal blinker at row 2, cols 1-3
        grid[2, 1] = 1
        grid[2, 2] = 1
        grid[2, 3] = 1

        gen1 = next_generation(grid)

        # Should now be a vertical blinker at col 2, rows 1-3
        expected_gen1 = np.zeros((5, 5), dtype=np.uint8)
        expected_gen1[1, 2] = 1
        expected_gen1[2, 2] = 1
        expected_gen1[3, 2] = 1
        np.testing.assert_array_equal(gen1, expected_gen1)

        # One more step should restore the original
        gen2 = next_generation(gen1)
        np.testing.assert_array_equal(gen2, grid)

    def test_underpopulation(self):
        """A live cell with fewer than 2 neighbours dies."""
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 2] = 1  # isolated cell – 0 neighbours
        result = next_generation(grid)
        assert result[2, 2] == 0

    def test_overpopulation(self):
        """A live cell with more than 3 neighbours dies."""
        grid = np.zeros((5, 5), dtype=np.uint8)
        # Centre cell surrounded by 4 neighbours (all 4 cardinal directions)
        grid[2, 2] = 1
        grid[1, 2] = 1
        grid[3, 2] = 1
        grid[2, 1] = 1
        grid[2, 3] = 1
        result = next_generation(grid)
        assert result[2, 2] == 0  # centre dies (4 neighbours)

    def test_reproduction(self):
        """A dead cell with exactly 3 neighbours becomes alive."""
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[1, 2] = 1
        grid[2, 1] = 1
        grid[2, 3] = 1  # three neighbours around (2,2)
        result = next_generation(grid)
        assert result[2, 2] == 1

    def test_output_dtype(self):
        grid = np.zeros((4, 4), dtype=np.uint8)
        result = next_generation(grid)
        assert result.dtype == np.uint8

    def test_output_shape_preserved(self):
        for shape in [(3, 3), (10, 20), (1, 1)]:
            grid = np.zeros(shape, dtype=np.uint8)
            result = next_generation(grid)
            assert result.shape == shape


# ---------------------------------------------------------------------------
# simulate tests
# ---------------------------------------------------------------------------

class TestSimulate:
    """Integration tests for the simulate() function."""

    def _run(self, width=10, height=10, steps=5, chunk_size=2, seed=42):
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            output = f.name
        try:
            simulate(
                width=width,
                height=height,
                steps=steps,
                chunk_size=chunk_size,
                output=output,
                seed=seed,
            )
            data = np.load(output)
        finally:
            os.unlink(output)
        return data

    def test_output_shape(self):
        width, height, steps = 8, 6, 4
        data = self._run(width=width, height=height, steps=steps)
        assert data.shape == (steps + 1, height, width)

    def test_output_values_binary(self):
        data = self._run()
        assert set(np.unique(data)).issubset({0, 1})

    def test_output_dtype(self):
        data = self._run()
        assert data.dtype == np.uint8

    def test_zero_steps(self):
        """With steps=0 only the initial frame is saved."""
        data = self._run(steps=0, chunk_size=1)
        assert data.shape[0] == 1

    def test_chunk_size_larger_than_steps(self):
        """chunk_size > steps should still produce the correct output."""
        data = self._run(steps=3, chunk_size=100)
        assert data.shape[0] == 4  # 3 steps + initial frame

    def test_reproducibility(self):
        """Same seed should produce identical results."""
        data1 = self._run(seed=7)
        data2 = self._run(seed=7)
        np.testing.assert_array_equal(data1, data2)

    def test_different_seeds_differ(self):
        """Different seeds should (almost certainly) produce different results."""
        data1 = self._run(seed=1)
        data2 = self._run(seed=2)
        assert not np.array_equal(data1[0], data2[0])

    def test_file_written_to_disk(self):
        """The .npy file must exist after the simulation."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            output = f.name
        try:
            simulate(width=5, height=5, steps=2, chunk_size=1,
                     output=output, seed=0)
            assert os.path.isfile(output)
        finally:
            os.unlink(output)
