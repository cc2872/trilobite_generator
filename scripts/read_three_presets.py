import json, time, sys
import schema, instrument as I
for name in ["proetida", "asaphida", "agnostida"]:
    P = json.load(open(f"presets/{name}.json"))["params"]
    t = time.time()
    r, B = I.read(P)
    keep = {k: r.get(k) for k in ["theta_joint_deg","total_deg","limited_by","enroll_class","s_tail","closure_gap_mm","bevel_built_deg","unsane_parts","limiting_pair","print_valid"] if k in r}
    print(name, f"{time.time()-t:.0f}s", json.dumps(keep, default=str))
