"""tests/make_references.py <preset> [...] — freeze instrument-2.1 (BREP builder) readings + meshes as regression targets.

Run once per preset on the OpenCascade builder before the mesh builder replaces it. Each target folder holds:
  measure_v2.json   the full 2.1 reading (theta_joint_deg, total_deg, enroll_class, s_tail, limited_by, bevel_built_deg, ...)
  <part>.stl        every part at the measurement bevel (45/35/25, whichever the probe chose — see measure_v2.json bevel_built_deg)
  flat.stl, posed.stl
  params.json       the coerced parameters the target was built from (schema at the time of freezing)
tests/test_regression.py compares a fresh build against these."""
import sys, os, json, shutil, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import batch_v2, schema

def freeze(name):
    t0 = time.time()
    msg = batch_v2.run_inproc(name)                       # writes out10/<name>/
    dst = os.path.join(os.path.dirname(__file__), "references", name)
    if os.path.isdir(dst): shutil.rmtree(dst)
    shutil.copytree(f"out10/{name}", dst)
    d = json.load(open(f"presets/{name}.json"))
    json.dump(schema.coerce(d["params"], base=schema.table_defaults()), open(f"{dst}/params.json", "w"), indent=1)
    open(f"{dst}/frozen.txt", "w").write(f"{time.strftime('%Y-%m-%d %H:%M')} instrument 2.1 BREP builder; {round(time.time()-t0)} s\n{msg}\n")
    return msg

if __name__ == "__main__":
    for n in sys.argv[1:]:
        try: print(freeze(n), flush=True)
        except Exception as ex: print(f"{n}: FAILED {ex}", flush=True)
