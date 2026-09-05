# Trilobite — skin on skeleton

**Skeleton** = your parametric generator (`schema.py`, `fields.py`). **Skin** = the furrow-cut
Olenoides sculpt, laid on it by your `wrap.py` law and resampled to `segCount`. One animal; body
type is a knob. Verdict from your `instrument.py`, unmodified.

```
07_skin_on_skeleton/   THE DELIVERABLE
  skin_on_skeleton.py     mine: tile sculpt segments onto the skeleton + trim neighbours (open-shell
                          version of your resample_segments). Everything else it calls is yours.
  trilobite.py            3-line shim: hinge_z / joint_offsets / pitch / hinge_width from fields.py,
                          so instrument.py runs without the CAD kernel
  schema.py fields.py wrap.py instrument.py   yours, unmodified
  skin_viewer.html + skin_data.js             your index.html chain, skeleton knobs re-tile the skin live
  instrument_results.json  the four runs (legs on/off × clearance 0.3/0.8)
  skinned_*.stl            P0, segCount 5, segCount 14 (L180), P0 at e_max
  README.md                weakness-first account
01_slicer/             cuts the sculpt along its furrows (spines whole) → the skin
02_phacops_eyes/       the fossil route (eyes extracted, not sliced)
03_legs/               aligns the sculpt's real legs into the shell frame
original_app/          your app, untouched (python trilobite_web.py)
```
