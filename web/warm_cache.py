"""
web/warm_cache.py: pre-build the isopod model for every preset into web/cache/ so the live site never runs a build
on a visitor's request. The isopod build is a full CSG build (~minutes, up to ~3 GB each); baking them once here
means the default page load and every preset are served instantly from cache, with the exact same geometry.

    python web/warm_cache.py                 # every preset (the default first), skipping ones already cached
    python web/warm_cache.py agnostida ...   # only the named presets

It drives the app's own /api/build endpoint (via the Flask test client) so the cache keys match real requests
byte for byte. Already-cached presets return immediately, so re-running only fills the gaps. Run it on the deploy
box after a code change to isopod_model/ (the cache key folds in ISOPOD_SIG, so a code change rebuilds on its own).
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "web"))
import app


def warm(names=None):
    presets = app._presets()
    order = names or ([app.schema.DEFAULT_PRESET] + [n for n in sorted(presets) if n != app.schema.DEFAULT_PRESET])
    client = app.app.test_client()
    t0 = time.time(); ok = 0
    for i, name in enumerate(order, 1):
        if name not in presets:
            print(f"[{i}/{len(order)}] {name}: no such preset — skipped", flush=True); continue
        t = time.time()
        r = client.post("/api/build", json={"P": presets[name], "joint": "isopod"})
        man = r.get_json() if r.status_code == 200 else {}
        errs = [p for p in man.get("parts", []) if "error" in p] + list(man.get("build_notes", []))
        dt = time.time() - t
        if r.status_code == 200 and man.get("parts") and not errs:
            ok += 1; print(f"[{i}/{len(order)}] {name}: OK  {len(man['parts'])} parts  {dt:.0f}s  key={man['key']}", flush=True)
        else:
            print(f"[{i}/{len(order)}] {name}: FAILED status={r.status_code} errs={errs}", flush=True)
    print(f"warmed {ok}/{len(order)} presets in {time.time() - t0:.0f}s -> {app.CACHE}", flush=True)
    return ok


if __name__ == "__main__":
    warm(sys.argv[1:] or None)
