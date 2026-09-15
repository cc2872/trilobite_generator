"""scripts/refreeze_reading.py <preset> [...] — rewrite a reference's measure_v2.json
with the CURRENT (mesh) engine's reading.

Some references (agnostida, asaphida) were frozen as invalid/rest_interference by the
old BREP builder; the mesh engine measures them fine. This re-measures at the SAME
frozen bevel_built_deg (so the frozen part .stl meshes still match, exactly as
tests/test_regression.py builds them) and overwrites only the measurement fields,
preserving provenance/metadata (bevel_probe, build_seconds, preset/order/blurb, hinge
geometry). Prints old-vs-new for every changed field.

    python scripts/refreeze_reading.py agnostida asaphida
    python scripts/refreeze_reading.py --dry-run agnostida     # print, don't write
"""
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import schema, instrument as I

WATCH = ("bevel_built_deg", "limited_by", "reason", "enroll_class", "theta_joint_deg",
         "total_deg", "closure_gap_mm", "gap_over_L", "s_tail", "rest_max_mm3")

def refreeze(name, write=True):
    d = os.path.join(ROOT, "tests", "references", name)
    path = os.path.join(d, "measure_v2.json")
    old = json.load(open(path))
    P = schema.coerce(json.load(open(os.path.join(d, "params.json"))))
    bevel = float(old["bevel_built_deg"])           # re-measure at the frozen bevel; meshes stay valid
    B = I.build_animal(P, bevel)
    r = I.measure(P, B["meshes"], bound_deg=bevel, head_body=B["head_body"],
                  bevel_built_deg=B["bevel_built_deg"], unsane_parts=B["unsane_parts"])
    merged = dict(old); merged.update(r)            # r overwrites measurement fields; metadata r never sets is preserved
    print(f"\n=== {name} (bevel {bevel:g}) ===")
    for k in WATCH:
        o, n = old.get(k), merged.get(k)
        print(f"  {k:18} {o!r:>22}  ->  {n!r}" + ("   [CHANGED]" if o != n else ""))
    if write:
        json.dump(merged, open(path, "w"), indent=1)
        print(f"  written: {path}")
    return merged

if __name__ == "__main__":
    args = sys.argv[1:]; write = "--dry-run" not in args
    names = [a for a in args if not a.startswith("--")]
    for n in names:
        refreeze(n, write=write)
