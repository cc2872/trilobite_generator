# segment/ — two standalone piece programs (10–11 Oct 2026)

Side programs, not used by the site. Each is one file with a `--list` of its parameters, builds an STL + a sheet,
and prints its own measurements (watertight, one body, bed contact, overhang, and for copies the boolean
intersection between neighbours).

- `segment.py` — ONE thoracic tergite drawn from Whittington's terms (ring, half-ring, furrows, pleurae, facet,
  spine), partially filled (solid to `fill`, hollow ring above), `--copies N` posed one pitch apart. No joint.
- `dome.py` — the isopod-model piece as the site prints it: a solid dome shingle with the isopod ball-and-socket
  built in; `--copies N` prints the chain in place. Reports the swing each way before neighbours touch and the
  extension budget `overlap + gap + step + 2 (crest - ball_z) sin(extend) <= pitch`.

    python segment/segment.py --copies 5 --taper 0.95 --out segment/out/five
    python segment/dome.py --copies 5 --out segment/out/dome5

Outputs go to `segment/out/` (git-ignored). Tests: `tests/test_segment_programs.py`.
