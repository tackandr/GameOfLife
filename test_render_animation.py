"""Tests for the render_animation CLI tool."""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest


def _make_npy(frames=10, height=20, width=20, seed=0) -> str:
    """Create a temporary .npy simulation file and return its path."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 2, size=(frames, height, width), dtype=np.uint8)
    fd, path = tempfile.mkstemp(suffix=".npy")
    try:
        np.save(path, data)
    finally:
        os.close(fd)
    return path


class TestRenderAnimation:
    """Integration tests for render_animation.py."""

    def _run_render(self, *extra_args):
        """Run render_animation.py as a subprocess and return (returncode, stdout+stderr)."""
        result = subprocess.run(
            [sys.executable, "render_animation.py"] + list(extra_args),
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout + result.stderr

    def test_produces_mp4_file(self):
        npy = _make_npy()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            mp4 = f.name
        try:
            rc, out = self._run_render("--input", npy, "--output", mp4, "--fps", "5")
            assert rc == 0, f"render_animation.py exited {rc}:\n{out}"
            assert os.path.isfile(mp4), "Output .mp4 file was not created."
            assert os.path.getsize(mp4) > 0, "Output .mp4 file is empty."
        finally:
            os.unlink(npy)
            if os.path.isfile(mp4):
                os.unlink(mp4)

    def test_invalid_input_exits_nonzero(self, tmp_path):
        mp4 = str(tmp_path / "out.mp4")
        rc, out = self._run_render("--input", "/nonexistent/path.npy", "--output", mp4)
        assert rc != 0

    def test_invalid_fps_exits_nonzero(self):
        npy = _make_npy()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            mp4 = f.name
        try:
            rc, out = self._run_render("--input", npy, "--output", mp4, "--fps", "0")
            assert rc != 0
        finally:
            os.unlink(npy)
            if os.path.isfile(mp4):
                os.unlink(mp4)

    def test_custom_fps_and_cmap(self):
        """Different fps and cmap options should still produce a valid file."""
        npy = _make_npy(frames=5, height=10, width=10)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            mp4 = f.name
        try:
            rc, out = self._run_render(
                "--input", npy,
                "--output", mp4,
                "--fps", "24",
                "--cmap", "viridis",
                "--dpi", "50",
            )
            assert rc == 0, f"render_animation.py exited {rc}:\n{out}"
            assert os.path.isfile(mp4)
            assert os.path.getsize(mp4) > 0
        finally:
            os.unlink(npy)
            if os.path.isfile(mp4):
                os.unlink(mp4)
