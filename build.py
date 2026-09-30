"""
build.py: the generator's default build is the isopod model — the crescent head made on the giant isopod's head
piece, on the isopod-style body with the isopod's own ball joints. This is a thin entry point; the model lives in
isopod_model/ (isopod_model/build.py, body.py, crescent_head.py, README.md).

    python build.py                     # the default animal -> isopod_model/stl/trilobite_isopod.stl, checks, photos
    python build.py --preset lichida    # one preset
    python build.py --no-photos --no-checks
    python build.py --rebuild-head      # remake the crescent band + cache from the isopod asset first

Every flag is isopod_model/build.py's; see its --help. The classic parametric generator (anatomy/ + assemble.py +
joints/pin) is still here and still runs the frozen enrollment instrument (instrument.py, sweep.py) unchanged.
"""
import os, sys, runpy

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "isopod_model")

if __name__ == "__main__":
    # run isopod_model/build.py as __main__ so its own sys.path setup and argparse (all flags) apply verbatim
    sys.path.insert(0, MODEL)
    runpy.run_path(os.path.join(MODEL, "build.py"), run_name="__main__")
