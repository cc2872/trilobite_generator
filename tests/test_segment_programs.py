"""The two standalone piece programs (segment/): a default build of each comes out watertight, one body, and their
checks run. ~20 s together; marked slow."""
import os, subprocess, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEG = os.path.join(ROOT, "segment")


def _run(prog, *args, out):
    r = subprocess.run([sys.executable, os.path.join(SEG, prog), "--no-sheet", "--out", out, *args], cwd=SEG, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-800:]
    return r.stdout


@pytest.mark.slow
def test_segment_program(tmp_path):
    out = _run("segment.py", "--copies", "2", out=str(tmp_path / "seg"))
    assert "watertight True" in out and "one body True" in out and "copies intersect: 0.00 mm3" in out
    assert (tmp_path / "seg.stl").exists() and (tmp_path / "seg.json").exists()


@pytest.mark.slow
def test_dome_program(tmp_path):
    out = _run("dome.py", "--copies", "2", out=str(tmp_path / "dome"))
    assert "watertight True" in out and "one body True" in out and "neighbours intersect at rest: 0.00 mm3" in out
    assert "ball to socket: least gap" in out and (tmp_path / "dome.stl").exists()
