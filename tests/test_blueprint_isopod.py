"""tests/test_blueprint_isopod.py — the isopod blueprint sheet (12 Oct 2026): its helpers (bed packing, 2D posing of
the midline sections, overhang, sparkline) and a full render on a small synthetic chain (boxes on ball-less pivots),
so the layout code runs end to end in seconds. The real sheet is drawn by /api/sheet after a build (test_web, slow)."""
import os, sys, math
import numpy as np, pytest, trimesh
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import blueprint as BP


def _chain(n=4, pitch=18.0, w=40.0, h=12.0):
    """n boxes nose to tail along +y (head first) and the pivots between them, like the isopod kit's frame."""
    parts, pivots = [], []
    for k in range(n):
        b = trimesh.creation.box((w - 4 * k, pitch - 1.0, h)); b.apply_translation((0, k * pitch + pitch / 2, h / 2)); parts.append(b)
        if k: pivots.append((k * pitch, 0.45 * h))
    return parts, pivots


def test_pack_bed_counts_plates_and_flags_no_fit():
    plates, nofit = BP._pack_bed([("a", 100, 20), ("b", 100, 20), ("c", 100, 20)], bed=256)
    assert len(plates) == 1 and not nofit and all(bx[3] >= bx[4] for bx in plates[0])          # long side along x
    plates, nofit = BP._pack_bed([(f"p{i}", 250, 60) for i in range(9)], bed=256)
    assert len(plates) == 3 and not nofit                                                       # 4 per plate at 60 + 4 mm margin
    plates, nofit = BP._pack_bed([("giant", 300, 40), ("ok", 50, 50)], bed=256)
    assert nofit == ["giant"] and len(plates) == 1
    for pl in plates:                                                                           # nothing overlaps, nothing leaves the bed
        for i, (n1, x1, y1, w1, h1) in enumerate(pl):
            assert x1 + w1 <= 256 and y1 + h1 <= 256
            for n2, x2, y2, w2, h2 in pl[i + 1:]:
                assert x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1


def test_posed_sections_match_the_3d_pose():
    parts, pivots = _chain()
    secs = BP._sections_x0(parts); assert all(secs)
    for curl in (0.0, 30.0, 71.5):
        T = BP._curl_mats(pivots, curl); ps = BP._posed_sections(secs, pivots, curl)
        for k in range(len(parts)):
            m = parts[k].copy().apply_transform(T[k]); s = m.section(plane_origin=[0, 0, 0], plane_normal=[1, 0, 0])
            ref = np.vstack([s.vertices[e.points][:, 1:3] for e in s.entities]); got = np.vstack(ps[k])
            from scipy.spatial import cKDTree
            d, _ = cKDTree(ref).query(got); assert d.max() < 1e-6


def test_overhang_fraction_sees_a_ledge_not_the_bed():
    slab = trimesh.creation.box((20, 20, 4)); slab.apply_translation((0, 0, 2))
    assert BP._overhang_fraction(slab, 0.0) == 0.0                                              # the bottom face sits on the bed
    ledge = trimesh.creation.box((20, 20, 4)); ledge.apply_translation((0, 0, 12))              # floating: its whole underside overhangs
    f = BP._overhang_fraction(trimesh.util.concatenate([slab, ledge]), 0.0)
    assert abs(f - 400 / (2 * (400 + 80 + 80) * 2)) < 1e-6


def test_spark_is_monotone_in_the_values():
    sp = BP._spark([1, 2, 3, 4]); assert sp[0] == "▁" and sp[-1] == "█" and list(sp) == sorted(sp)
    assert BP._spark([]) == "" and len(BP._spark([5] * 6)) == 6


def test_isopod_sheet_renders_a_synthetic_chain(tmp_path):
    parts, pivots = _chain(5)
    P = dict(clearance=0.3, cephFrac=0.3, pygFrac=0.2, axisFrac=0.3, glabInflate=1.2, eyeSize=0.1, eyePos=0.6, genalSpine=0.2, pygSpine=0.1, segCount=3)
    info = dict(names=["head", "seg0", "seg1", "seg2", "tail"], pivots=pivots, curl_deg=30.0, params="test-key", schema="6.2", build_seconds=1.0,
                volumes=[float(p.volume) for p in parts], meta=[dict(bodies=1, watertight=True)] * 5,
                checks=dict(print_valid=True, touching=[dict(a="seg0", b="seg1", mm3=0.12)], debris_mm3={"seg2": 0.4}, notes=[]),
                head=dict(style="custom", id="abc123", outline={"brim": 1.2}), body=dict(model="thorax", rows=[dict(width=w) for w in (30, 28, 25, 20)]), date="2026-10-12")
    png = BP.isopod_sheet(parts, P, info, str(tmp_path / "sheet.png"))
    assert os.path.getsize(png) > 50_000
    png = BP.isopod_sheet(parts[:1], P, dict(info, names=["head"], pivots=[], volumes=info["volumes"][:1], meta=info["meta"][:1]), str(tmp_path / "one.png"))
    assert os.path.getsize(png) > 20_000                                                        # a single piece: no joints, still a sheet
