"""
isopod_model/presets.py: the isopod model built from the generator's parameters. Any schema parameter set builds
(the schema is not changed: the model reads the same P the generator does); the named ones are the generator's
default ("textured") and the ten order presets in presets/*.json, loaded exactly as sweep.py loads them.

    python isopod_model/presets.py [name ...] [--stl]   -> readings/<name>.json, photos/presets/<name>.png
                                                          (--stl: also stl/presets/<name>.stl)

Per preset: build (body.build with the isopod head), sanity (every piece watertight, one body), rest contacts, and
the enrollment test (enroll.py). A preset that cannot build is recorded with the reason, not skipped.
"""
import os, sys, json, glob, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np, trimesh
import schema


def names():
    return ["textured"] + [os.path.basename(f)[:-5] for f in sorted(glob.glob(os.path.join(ROOT, "presets", "*.json")))
                           if not os.path.basename(f).startswith("_")]


def params(name):
    """As sweep.py: presets/<name>.json coerced over the table defaults; 'textured' (or any schema.PRESETS key) from the
    schema."""
    f = os.path.join(ROOT, "presets", f"{name}.json")
    if os.path.exists(f): return schema.coerce(json.load(open(f))["params"], base=schema.table_defaults())
    return schema.preset(name)


def run(name, stl=False, overrides=None):
    """overrides: {parameter: value} on top of the preset (the run is saved as <name>+key=value...)."""
    import body as BODY, check as CHECK, enroll as EN, render as R
    t0 = time.time(); base = name
    if overrides: name = name + "".join(f"+{k}={v:g}" for k, v in overrides.items())
    out = dict(preset=name)
    os.makedirs(os.path.join(HERE, "readings"), exist_ok=True)
    try:
        P = params(base)
        if overrides: P = schema.coerce(dict(P, **overrides), base=P); out["overrides"] = overrides
        out["params_hash"] = schema.param_hash(P)
        out.update(segCount=int(P["segCount"]), length=float(P["length"]))
        parts, plans, PV, K = BODY.build(P, "isopod")
        out["dropped_flakes_mm3"] = {str(k): round(v, 3) for k, v in BODY.LAST_DROPPED.items()}
        out["skipped_ornaments"] = list(BODY.SKIPPED)
        s = K["s"]; out["print_scale"] = round(1.0 / s, 4); out["build_seconds"] = round(time.time() - t0, 1)
        out["pieces"] = [dict(watertight=bool(p.is_watertight), bodies=len(p.split(only_watertight=False))) for p in parts]
        out["sane"] = all(p["watertight"] and p["bodies"] == 1 for p in out["pieces"])
        pv = [(y, z) for y, z in PV]
        out["rest_contacts"] = CHECK.contacts(parts, pv, 0.0)
        out["reading"] = EN.reading(P, parts, pv, name)
        pieces = [p.copy().apply_scale(1.0 / s) for p in parts]
        whole = trimesh.util.concatenate(pieces); out["print_size_mm"] = np.round(whole.extents, 1).tolist()
        th = out["reading"].get("theta_joint_deg") or 0.0
        T = CHECK.pose(pv, [th] * len(pv))
        curled = trimesh.util.concatenate([p.copy().apply_transform(T[k]) for k, p in enumerate(parts)]); curled.apply_scale(1.0 / s)
        R.sheet([[(whole, 30, -55, f"{name}"), (whole, 90, -90, "top"), (curled, 4, 0, f"curled {th:g} deg/joint: {out['reading']['limited_by']}, {out['reading']['enroll_class']}")]],
                os.path.join(HERE, "photos", "presets", f"{name}.png"), W=760, H=520)
        if stl:
            os.makedirs(os.path.join(HERE, "stl", "presets"), exist_ok=True); whole.export(os.path.join(HERE, "stl", "presets", f"{name}.stl"))
    except Exception as ex:
        out["error"] = f"{type(ex).__name__}: {ex}"; out["trace"] = traceback.format_exc()[-1500:]
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(HERE, "readings", f"{name}.json"), "w"), indent=1, default=str)
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--") and "=" not in a]
    ov = {a.split("=")[0]: float(a.split("=")[1]) for a in sys.argv[1:] if "=" in a}   # e.g. axialSpine=1.5
    for nm in (args or names()):
        r = run(nm, stl="--stl" in sys.argv, overrides=ov or None)
        rd = r.get("reading", {})
        print(f"{nm}: " + (r["error"] if "error" in r else
              f"sane {r['sane']}, rest {r['rest_contacts'] or 'clear'}, {rd.get('limited_by')} {rd.get('enroll_class')} "
              f"theta {rd.get('theta_joint_deg')} total {rd.get('total_deg')} s_tail {rd.get('s_tail')} ({r['seconds']} s)"), flush=True)
