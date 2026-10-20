"""
landmarks.py — homologous dorsal landmarks for geometric morphometrics, and a TPS writer (geomorph / tpsDig readable).

Each generated specimen gets the SAME ordered set of landmarks, so specimens are comparable across the morphospace
(geomorph aligns them by Procrustes, so only consistency matters — not the absolute frame). Points are the dorsal
plan, in millimetres, in a body frame: x is lateral (right +, midline 0), y runs anterior (cephalon front = 0) to
posterior (pygidium tip = length). The formulas match the generator's own proportions (schema.py / anatomy).

    landmarks(P)              -> [(name, x, y), ...]   (12 points, fixed order)
    tps_string(specimens)     -> TPS text   (specimens: list of (id, P))
    write_tps(specimens, path)

The eye landmark is the eye POSITION (a homologous location on the cheek); it is present even for effaced or blind
forms, so the count stays constant. thorax_25/50/75 are axial semilandmarks at fixed fractions of thorax length, so
the count does not change with segment number.
"""

LANDMARK_NAMES = ["ceph_anterior", "genal_R", "genal_L", "glabella_anterior", "occipital",
                  "eye_R", "eye_L", "thorax_25", "thorax_50", "thorax_75", "thorax_pyg_boundary", "pygidium_tip"]


def landmarks(P):
    """12 homologous dorsal landmarks (name, x, y) in mm. See module docstring for the frame."""
    g = lambda k, d: float(P.get(k, d))
    L = g("length", 150.0); W = g("width", 70.0); hw = W / 2.0
    cL = g("cephFrac", 0.33) * L; pL = g("pygFrac", 0.2) * L; tL = max(L - cL - pL, 1.0)
    eLat = g("eyeLat", 0.0)
    eX = eLat * hw if eLat > 0.01 else (g("axisFrac", 0.28) * g("glabInflate", 1.35) + g("eyeSize", 0.15)) * hw
    eX = min(eX, 0.95 * hw)                                             # keep it on the cheek
    eY = cL * (1.0 - g("eyePos", 0.6))                                  # eyePos: fraction of cephalon length from the rear
    return [("ceph_anterior", 0.0, 0.0),
            ("genal_R", hw, cL), ("genal_L", -hw, cL),
            ("glabella_anterior", 0.0, 0.12 * cL), ("occipital", 0.0, 0.92 * cL),
            ("eye_R", eX, eY), ("eye_L", -eX, eY),
            ("thorax_25", 0.0, cL + 0.25 * tL), ("thorax_50", 0.0, cL + 0.50 * tL), ("thorax_75", 0.0, cL + 0.75 * tL),
            ("thorax_pyg_boundary", 0.0, cL + tL), ("pygidium_tip", 0.0, L)]


def tps_string(specimens):
    """specimens: list of (id, P). Returns one TPS block per specimen (geomorph readland.tps compatible).
    Coordinates are already in mm, so SCALE=1.0. The id travels in IMAGE/ID/COMMENT for traceability."""
    out = []
    for i, (sid, P) in enumerate(specimens):
        pts = landmarks(P)
        out.append("LM=%d" % len(pts))
        out += ["%.4f %.4f" % (x, y) for _, x, y in pts]
        out.append("IMAGE=%s" % sid)
        out.append("ID=%d" % i)
        out.append("SCALE=1.0")
        out.append("COMMENT=hash=%s; landmarks=%s" % (sid, ",".join(LANDMARK_NAMES)))
        out.append("")
    return "\n".join(out)


def write_tps(specimens, path):
    with open(path, "w") as f: f.write(tps_string(specimens))
    return path


if __name__ == "__main__":                                             # demo: all named presets -> one TPS (the batch use case)
    import schema
    specs = [(name, schema.coerce_report(dict(pset), base=schema.table_defaults())[0]) for name, pset in schema.PRESETS.items()]
    print(tps_string(specs))
