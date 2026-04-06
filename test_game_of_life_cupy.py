"""Tests for the CuPy-backed Game of Life simulation.

Because the sandboxed CI environment has no GPU / CuPy installation, these
tests inject a lightweight numpy-backed mock for the ``cp`` module so that
the full logic of ``simulate_cupy`` and ``next_generation_cupy`` can be
exercised without real GPU hardware.
"""

import os
import tempfile
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Lightweight numpy-backed mock of the cupy namespace
# ---------------------------------------------------------------------------

class _CpArray(np.ndarray):
    """np.ndarray subclass that adds the ``get()`` method expected by CuPy arrays.

    ``__array_finalize__`` ensures that all ufunc and slicing results are also
    ``_CpArray`` instances, so operations like ``.astype()`` and indexing
    transparently preserve the type throughout ``next_generation_cupy``.
    """

    def __array_finalize__(self, obj):
        pass  # No extra attributes to propagate.

    def get(self, order="C", out=None, blocking=True, stream=None):
        """Copy array data to a host numpy array (mock of cupy's D2H transfer)."""
        arr = self.view(np.ndarray)
        if out is not None:
            out[:] = arr
            return out
        return arr.copy()


def _wrap(arr):
    """View a numpy array as ``_CpArray``."""
    if isinstance(arr, np.ndarray):
        return arr.view(_CpArray)
    return arr


class _MockStream:
    """Synchronous no-op replacement for ``cp.cuda.Stream``."""

    def __init__(self, non_blocking=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def wait_event(self, event):
        pass

    def synchronize(self):
        pass

    def record(self):
        return MagicMock()


class _MockCuda:
    Stream = _MockStream

    @staticmethod
    def alloc_pinned_memory(size):
        # Return a plain numpy byte-array; supports the buffer protocol, so
        # ``np.frombuffer`` works on it exactly as on real pinned memory.
        return np.empty(size, dtype=np.uint8)


def _make_mock_cp():
    """Return a ``types.SimpleNamespace`` that mimics the cupy public API."""
    mock = types.SimpleNamespace()
    mock.array   = lambda a, **kw: _wrap(np.array(a, **kw))
    mock.asarray = lambda a, **kw: _wrap(np.asarray(a, **kw))
    mock.empty   = lambda shape, **kw: _wrap(np.empty(shape, **kw))
    mock.roll    = lambda a, shift, axis=None: _wrap(
        np.roll(np.asarray(a), shift, axis=axis)
    )
    mock.where   = lambda c, x, y: _wrap(
        np.where(np.asarray(c), np.asarray(x), np.asarray(y))
    )
    mock.uint8   = np.uint8
    mock.cuda    = _MockCuda()
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_cupy(monkeypatch):
    """Replace the ``cp`` global in ``game_of_life_cupy`` with the numpy mock."""
    import game_of_life_cupy as _mod
    monkeypatch.setattr(_mod, "cp", _make_mock_cp())
    monkeypatch.setattr(_mod, "_CUPY_AVAILABLE", True)


# ---------------------------------------------------------------------------
# next_generation_cupy tests
# ---------------------------------------------------------------------------

# Import after the mock fixture has been defined.
import game_of_life_cupy as _gol_cupy  # noqa: E402


class TestNextGenerationCupy:
    """Unit tests for ``next_generation_cupy``."""

    def _gen(self, grid_np):
        """Run one generation and return a plain numpy array."""
        import game_of_life_cupy as mod
        result = mod.next_generation_cupy(_wrap(grid_np))
        return np.asarray(result)

    def test_all_dead_stays_dead(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        result = self._gen(grid)
        np.testing.assert_array_equal(result, grid)

    def test_block_stable(self):
        """2×2 block is a still life."""
        grid = np.zeros((6, 6), dtype=np.uint8)
        grid[2:4, 2:4] = 1
        result = self._gen(grid)
        np.testing.assert_array_equal(result, grid)

    def test_blinker_oscillates(self):
        """Horizontal blinker flips to vertical and back."""
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 1] = grid[2, 2] = grid[2, 3] = 1

        gen1 = self._gen(grid)

        expected = np.zeros((5, 5), dtype=np.uint8)
        expected[1, 2] = expected[2, 2] = expected[3, 2] = 1
        np.testing.assert_array_equal(gen1, expected)

        gen2 = self._gen(gen1)
        np.testing.assert_array_equal(gen2, grid)

    def test_underpopulation(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 2] = 1  # isolated – dies
        assert self._gen(grid)[2, 2] == 0

    def test_overpopulation(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 2] = grid[1, 2] = grid[3, 2] = grid[2, 1] = grid[2, 3] = 1
        assert self._gen(grid)[2, 2] == 0  # 4 neighbours → dies

    def test_reproduction(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[1, 2] = grid[2, 1] = grid[2, 3] = 1
        assert self._gen(grid)[2, 2] == 1

    def test_output_dtype(self):
        grid = np.zeros((4, 4), dtype=np.uint8)
        result = self._gen(grid)
        assert result.dtype == np.uint8

    def test_output_shape_preserved(self):
        for shape in [(3, 3), (10, 20), (1, 1)]:
            grid = np.zeros(shape, dtype=np.uint8)
            result = self._gen(grid)
            assert result.shape == shape

    def test_matches_numpy_implementation(self):
        """CuPy and NumPy implementations must agree on the same grid."""
        from game_of_life import next_generation
        rng = np.random.default_rng(99)
        grid_np = rng.integers(0, 2, size=(20, 20), dtype=np.uint8)

        expected = next_generation(grid_np)
        actual = self._gen(grid_np)
        np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# simulate_cupy tests
# ---------------------------------------------------------------------------

class TestSimulateCupy:
    """Integration tests for ``simulate_cupy``."""

    def _run(self, width=10, height=10, steps=5, chunk_size=2, seed=42):
        import glob as _glob
        import game_of_life_cupy as mod
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "sim.npy")
            mod.simulate_cupy(
                width=width,
                height=height,
                steps=steps,
                chunk_size=chunk_size,
                output=output,
                seed=seed,
            )
            chunk_files = sorted(_glob.glob(os.path.join(tmpdir, "sim_*.npy")))
            return np.concatenate([np.load(f) for f in chunk_files], axis=0)

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
        """chunk_size > steps should still produce correct output."""
        data = self._run(steps=3, chunk_size=100)
        assert data.shape[0] == 4  # 3 steps + initial frame

    def test_reproducibility(self):
        """Same seed must produce identical results."""
        data1 = self._run(seed=7)
        data2 = self._run(seed=7)
        np.testing.assert_array_equal(data1, data2)

    def test_different_seeds_differ(self):
        """Different seeds should (almost certainly) produce different frames."""
        data1 = self._run(seed=1)
        data2 = self._run(seed=2)
        assert not np.array_equal(data1[0], data2[0])

    def test_file_written_to_disk(self):
        """At least one chunk .npy file must exist after the simulation."""
        import glob as _glob
        import game_of_life_cupy as mod
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "sim.npy")
            mod.simulate_cupy(width=5, height=5, steps=2, chunk_size=1,
                               output=output, seed=0)
            chunk_files = _glob.glob(os.path.join(tmpdir, "sim_*.npy"))
            assert len(chunk_files) > 0

    def test_matches_numpy_simulation(self):
        """CuPy and NumPy simulators must agree given the same seed."""
        import glob as _glob
        from game_of_life import simulate
        import game_of_life_cupy as mod

        common_kwargs = dict(width=10, height=10, steps=6, chunk_size=3, seed=123)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np_out = f.name
        try:
            simulate(**common_kwargs, output=np_out)
            np_data = np.load(np_out)
        finally:
            os.unlink(np_out)

        with tempfile.TemporaryDirectory() as tmpdir:
            cp_out = os.path.join(tmpdir, "sim.npy")
            mod.simulate_cupy(**common_kwargs, output=cp_out)
            chunk_files = sorted(_glob.glob(os.path.join(tmpdir, "sim_*.npy")))
            cp_data = np.concatenate([np.load(f) for f in chunk_files], axis=0)

        np.testing.assert_array_equal(cp_data, np_data)

    def test_cupy_unavailable_raises(self, monkeypatch):
        """simulate_cupy must raise RuntimeError when CuPy is absent."""
        import game_of_life_cupy as mod
        monkeypatch.setattr(mod, "_CUPY_AVAILABLE", False)
        with pytest.raises(RuntimeError, match="CuPy is not installed"):
            mod.simulate_cupy(
                width=5, height=5, steps=1, chunk_size=1, output="/tmp/x.npy"
            )

    def test_nvtx_ranges_entered(self, monkeypatch):
        """cupyx time_range must be entered for each chunk's kernels and D2H."""
        import game_of_life_cupy as mod

        entered_ranges = []

        class _TrackingRange:
            def __init__(self, name):
                self._name = name

            def __enter__(self):
                entered_ranges.append(self._name)
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(mod, "_nvtx_range", _TrackingRange)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sim.npy")
            # 4 steps / chunk_size=2 → 2 chunks → 2 kernel + 2 D2H ranges
            mod.simulate_cupy(width=5, height=5, steps=4, chunk_size=2,
                               output=out, seed=0)

        kernel_ranges = [r for r in entered_ranges if r.startswith("kernel")]
        d2h_ranges = [r for r in entered_ranges if r.startswith("D2H")]
        assert len(kernel_ranges) == 2, f"kernel ranges: {kernel_ranges}"
        assert len(d2h_ranges) == 2, f"D2H ranges: {d2h_ranges}"

    def test_streams_alternated(self, monkeypatch):
        """The two output streams must be used in alternating order."""
        import game_of_life_cupy as mod

        call_log = []

        class _TrackingStream(_MockStream):
            def __init__(self, idx, non_blocking=False):
                self._idx = idx

            def synchronize(self):
                call_log.append(("sync", self._idx))

            def wait_event(self, event):
                call_log.append(("wait", self._idx))

        stream_instances = [_TrackingStream(0), _TrackingStream(1)]
        stream_call_count = [0]

        class _MockCudaTracking:
            @staticmethod
            def alloc_pinned_memory(size):
                return np.empty(size, dtype=np.uint8)

            class Stream:
                def __new__(cls, non_blocking=False):
                    idx = stream_call_count[0] % 2
                    stream_call_count[0] += 1
                    return stream_instances[idx]

        mock_cp = _make_mock_cp()
        mock_cp.cuda = _MockCudaTracking()
        monkeypatch.setattr(mod, "cp", mock_cp)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sim.npy")
            # 4 steps with chunk_size=2 → 2 chunks → both streams used
            mod.simulate_cupy(width=5, height=5, steps=4, chunk_size=2,
                               output=out, seed=0)

        # Each stream must have been synchronised at least once.
        synced = {idx for (op, idx) in call_log if op == "sync"}
        assert 0 in synced and 1 in synced, (
            f"Expected both streams to be synchronised; got: {call_log}"
        )
