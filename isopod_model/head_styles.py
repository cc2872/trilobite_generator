"""
isopod_model/head_styles.py (2 Oct 2026): named heads for the isopod model.

The crescent head's surface (anatomy/head_crescent.py) has sculpt keys beyond the schema: glabella length, side
bulge and frontal boss, furrow type, bullar lobes, median node, eye-lobe length, genal caeca, a pitted brim,
tubercles, effacement. A STYLE sets those keys ("face") plus any schema parameters the look needs ("params": eyes,
glabella numbers, spines, prongs, suture). The sources are the order fact sheets of Gon (2009); the page is in `p`.

    style(name)                 -> (face overrides, schema overrides)
    apply(P, name), outline(name)   -> body.build(P2, "shaped", face=face, outline=outline(name))
    python isopod_model/head_styles.py [name ...]   -> photos/head_styles.png  (the heads alone, 3/4 and top)

A style is a build option, like the head and the joint: it adds no schema key and changes no parameter hash by
itself (its "params" overrides do, as any parameter edit does). Nothing here is read by the instrument.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)

HOLO = dict(eyeSolid=1, eyeSphere=1, lensD=0.05, lensGap=0.0)    # holochroal: a smooth globe set in the cheek
LOBE = dict(eyeSolid=0)                                          # no eye solid: the eye is the palpebral lobe sculpted in the cheek (eye_len stretches it)
SCHIZO = dict(eyeSolid=1, lensD=0.16, lensGap=0.3, eyeProfile=6.0)
BLIND = dict(eyeSize=0.0, eyeSolid=0)

STYLES = {
    # ---- after the guide's orders
    "olenellus":   dict(p=54, doc="Redlichiida: round frontal boss, long crescent eyes joined to it by eye ridges, furrows crossing the glabella",
                        face=dict(glab_boss=0.20, boss_rise=0.22, glab_round=0.7, furrow_cross=0.7, eye_len=2.3, glab_len=1.05, eye_ridge=1.0, border_w=0.05),
                        params=dict(LOBE, glabInflate=0.9, glabRise=0.14, glabLobes=3, eyeSize=0.12, eyePos=0.55, eyeHeight=2.4)),
    "paradoxides": dict(p=55, doc="Redlichiida: glabella swelling forward to the border, transglabellar furrows, long eye lobes",
                        face=dict(glab_len=1.06, furrow_cross=1.0, eye_len=1.9, glab_bulge=0.0, glab_round=0.6),
                        params=dict(LOBE, glabInflate=1.9, glabRise=0.2, glabLobes=2, eyeSize=0.12, eyePos=0.45, eyeHeight=2.2)),
    "agnostus":    dict(p=58, doc="Agnostina: blind, tapering glabella in two lobes with a median node, wide convex border",
                        face=dict(glab_len=0.80, glab_round=0.9, furrow_cross=1.0, node=0.7, border_w=0.15, glab_bulge=0.04),
                        params=dict(BLIND, glabInflate=0.65, glabRise=0.22, glabLobes=1)),
    "olenoides":   dict(p=61, doc="Corynexochina: long pestle-shaped glabella reaching the border, splayed furrows, narrow eyes",
                        face=dict(glab_len=1.06, glab_bulge=-0.14, furrow_splay=1.0, eye_len=1.6, glab_round=0.5),
                        params=dict(LOBE, glabInflate=1.55, glabRise=0.2, glabLobes=4, eyeSize=0.10, eyePos=0.5, eyeHeight=2.2)),
    "bumastus":    dict(p=61, doc="Illaenina: effaced, a smooth dome with small eyes set far back and wide",
                        face=dict(efface=1.0, border_w=0.03),
                        params=dict(HOLO, glabLobes=0, eyeSize=0.10, eyePos=0.3, eyeLat=0.72, eyeHeight=0.9, borderWidth=0.0)),
    "lichas":      dict(p=64, doc="Lichoidea: broad glabella to the border with bullar lobes beside it, tubercles everywhere",
                        face=dict(glab_len=1.06, bullae=0.19, tubercles=0.45, tub_all=1, glab_bulge=0.0, glab_round=0.6),
                        params=dict(HOLO, glabInflate=1.5, glabRise=0.2, glabLobes=2, eyeSize=0.09, eyePos=0.38, eyeLat=0.62)),
    "odontopleura": dict(p=64, doc="Odontopleuroidea: tapering glabella, eye ridges, rows of tubercles, occipital spine",
                        face=dict(glab_len=0.98, tubercles=0.35, tub_all=1, eye_ridge=1.0, glab_round=0.6),
                        params=dict(HOLO, glabInflate=0.85, glabRise=0.2, glabLobes=3, eyeSize=0.09, eyePos=0.4, eyeLat=0.6, eyeHeight=1.5, occipitalSpine=0.7)),
    "modocia":     dict(p=68, doc="Ptychoparioidea: the generalized form: tapering glabella, a clear preglabellar field, eye ridges",
                        face=dict(glab_len=0.86, glab_round=0.6, eye_ridge=0.7, eye_len=1.3),
                        params=dict(LOBE, glabInflate=0.8, glabRise=0.16, glabLobes=3, eyeSize=0.11, eyePos=0.5, eyeHeight=2.0)),
    "triarthrus":  dict(p=69, doc="Olenina: threadlike border, genal caeca fanning over the cheeks, small eyes on ridges",
                        face=dict(caeca=0.45, eye_ridge=0.8, border_w=0.045, glab_len=1.04, furrow_splay=0.6, glab_round=0.6),
                        params=dict(LOBE, glabInflate=0.9, glabRise=0.16, glabLobes=3, eyeSize=0.08, eyePos=0.62, eyeHeight=2.0)),
    "harpes":      dict(p=72, doc="Harpetida: a broad pitted brim, short narrowing glabella, eyes reduced to tubercles on strong ridges",
                        face=dict(glab_len=0.66, glab_round=1.0, eye_ridge=1.1),
                        params=dict(glabInflate=0.7, glabRise=0.3, glabLobes=2, eyeSize=0.05, eyePos=0.42, eyeLat=0.36, eyeHeight=2.5, eyeSolid=0)),
    "phacops":     dict(p=76, doc="Phacopoidea: one great inflated glabella swallowing the front, schizochroal eyes at the front corners",
                        face=dict(glab_len=1.07, glab_bulge=0.16, tubercles=0.3, glab_round=0.8),
                        params=dict(SCHIZO, glabInflate=2.0, glabRise=0.3, glabLobes=1, eyeSize=0.14, eyePos=0.72, eyeLat=0.62, eyeHeight=1.6, eyeArc=120,
                                    sutureDepth=0.3, sutureEnd=-1.0)),
    "dalmanites":  dict(p=76, doc="Dalmanitoidea: preglabellar field kept, lobed glabella, large eyes set back, a front process",
                        face=dict(glab_len=0.93, furrow_splay=0.5, glab_bulge=0.05, glab_round=0.6),
                        params=dict(SCHIZO, glabInflate=1.6, glabRise=0.2, glabLobes=3, eyeSize=0.15, eyePos=0.38, eyeLat=0.58, eyeHeight=1.5, eyeArc=140,
                                    headProngs=1, headProngLen=0.3, sutureDepth=0.3, sutureEnd=-1.0)),
    "encrinurus":  dict(p=78, doc="Cheirurina: a strawberry: the whole glabella in coarse tubercles, small stalked eyes",
                        face=dict(glab_len=1.05, tubercles=0.8, glab_bulge=0.14, glab_round=1.0),
                        params=dict(HOLO, glabInflate=1.6, glabRise=0.28, glabLobes=0, eyeSize=0.07, eyePos=0.45, eyeLat=0.6, eyeStalk=1.2, eyeStalkBend=20)),
    "deiphon":     dict(p=78, doc="Cheirurina: the glabella as a sphere",
                        face=dict(glab_boss=0.42, boss_rise=0.85, glab_len=1.04, tubercles=0.3, glab_round=1.0),
                        params=dict(HOLO, glabInflate=1.3, glabRise=0.1, glabLobes=0, eyeSize=0.06, eyePos=0.6, eyeLat=0.75)),
    "isotelus":    dict(p=81, doc="Asaphoidea: effaced, the preoccipital tubercle, tall holochroal eyes",
                        face=dict(efface=0.85, node=0.6, glab_round=1.0),
                        params=dict(HOLO, glabLobes=0, glabInflate=1.0, eyeSize=0.11, eyePos=0.45, eyeLat=0.5, eyeTall=1.35, eyeSink=0.45)),
    "neoasaphus":  dict(p=84, doc="Asaphoidea: effaced, eyes on long stalks",
                        face=dict(efface=0.9, node=0.5, glab_round=1.0),
                        params=dict(HOLO, glabLobes=0, glabInflate=1.0, eyeSize=0.08, eyePos=0.45, eyeLat=0.5, eyeStalk=3.0, eyeStalkBend=35)),
    "cryptolithus": dict(p=81, doc="Trinucleioidea: blind, a pear-shaped glabella standing out of a brim of pits",
                        face=dict(glab_boss=0.24, boss_rise=0.5, glab_len=0.94, glab_round=1.0),
                        params=dict(BLIND, glabInflate=1.5, glabRise=0.25, glabLobes=0)),
    "cyclopyge":   dict(p=82, doc="Cyclopygoidea: glabella to the front margin, effaced, huge eyes as the sides of the head",
                        face=dict(glab_len=1.06, efface=0.7, glab_round=1.0),
                        params=dict(HOLO, eyeSphere=1, glabLobes=0, glabInflate=1.3, eyeSize=0.36, eyePos=0.6, eyeLat=0.72, eyeArc=280, eyeAxial=1.3, eyeSink=1.0)),
    "walliserops": dict(p=74, doc="Acastoidea: the trident",
                        face=dict(glab_len=1.05, tubercles=0.3, tub_all=1, glab_round=0.7),
                        params=dict(SCHIZO, glabInflate=1.6, glabRise=0.22, glabLobes=3, eyeSize=0.13, eyePos=0.5, eyeLat=0.6, eyeHeight=1.8, eyeArc=130,
                                    headProngs=3, headProngLen=0.9, headProngStem=0.5, headProngSplay=40, headProngCurl=18, occipitalSpine=0.5)),
    # ---- not in the guide: what the keys do at their limits
    "all_on":      dict(p=0, doc="every sculpt key at once",
                        face=dict(glab_len=1.1, glab_boss=0.26, boss_rise=0.5, furrow_cross=0.5, furrow_splay=1.0, bullae=0.15, node=0.6, eye_len=1.8,
                                  caeca=0.4, brim=0.22, brim_pits=2, tubercles=0.4, eye_ridge=1.0),
                        params=dict(SCHIZO, glabInflate=1.3, glabRise=0.2, glabLobes=5 - 1, eyeSize=0.1, eyePos=0.5, eyeLat=0.66, headProngs=2, headProngLen=0.5)),
}

STYLES["ceraurus"] = dict(p=78, doc="Cheirurina: barrel glabella with four furrow pairs, tuberculate, genal spines sweeping outward",
                          face=dict(glab_len=1.08, glab_round=0.8, glab_bulge=0.12, tubercles=0.35, tub_all=1),
                          params=dict(HOLO, glabInflate=1.25, glabRise=0.24, glabLobes=4, eyeSize=0.07, eyePos=0.5, eyeLat=0.55))
STYLES["ampyx"] = dict(p=80, doc="Trinucleioidea (Raphiophoridae): blind, a long median frontal spine and genal spines longer than the body",
                       face=dict(glab_boss=0.22, boss_rise=0.4, glab_len=0.96, glab_round=1.0), params=dict(BLIND, glabInflate=1.4, glabRise=0.25, glabLobes=0))
STYLES["dikelocephalus"] = dict(p=82, doc="Dikelokephaloidea: squat glabella truncate in front, a square-cut head",
                       face=dict(glab_len=0.9, glab_bulge=0.0, furrow_cross=0.6, eye_len=1.5), params=dict(LOBE, glabInflate=1.0, glabRise=0.18, glabLobes=2, eyeSize=0.1, eyePos=0.45, eyeHeight=2.0))
STYLES["bristolia"] = dict(p=52, doc="Olenelloidea: the olenellid face, with the genal spines advanced far up the sides",
                       face=dict(STYLES["olenellus"]["face"]), params=dict(STYLES["olenellus"]["params"]))

# ---- the silhouette of each style (shaped_head.OUTLINE keys; 2 Oct 2026). The major head types of the guide:
SP = dict(genal_w=0.10, genal_tip=0.12, notch=0.08, widest=0.03)              # a slim genal spine on a straight rear margin
OUTLINES = {
    "olenellus":    dict(SP, length=0.95, genal=0.75, width=1.22),                                   # wide semicircle, genal spines
    "paradoxides":  dict(SP, length=0.90, genal=1.6, genal_w=0.12, genal_spread=4, width=1.22),      # ... very long ones
    "agnostus":     dict(length=1.25, front_exp=2.2, widest=0.45, side=0.35, rear_taper=0.14, genal=0, corner=0.2, width=1.02),   # parabolic shield, widest forward
    "olenoides":    dict(SP, length=0.90, genal=0.7, width=1.2),
    "bumastus":     dict(length=1.0, front_exp=2.3, genal=0, corner=0.3, widest=0.3, side=0.2, dome_fill=0.97, width=1.05, height=1.15),  # a smooth half-dome
    "lichas":       dict(length=0.8, front_exp=2.4, genal=0.5, genal_w=0.16, genal_tip=0.2, genal_spread=10, notch=0.1, widest=0.03, width=1.2),
    "odontopleura": dict(length=0.7, front_exp=2.6, genal=1.0, genal_w=0.10, genal_tip=0.15, genal_spread=40, genal_curve=-45, notch=0.05, widest=0.05, width=1.15),  # short, transverse, spines flung out
    "modocia":      dict(SP, length=0.9, genal=0.55, notch=0.1, dome_fill=0.78, width=1.2),          # the generalized ptychoparioid
    "triarthrus":   dict(length=0.8, front_exp=2.5, genal=0, corner=0.2, widest=0.1, side=0.1, dome_fill=0.92, width=1.05),      # short, rounded genal angles
    "harpes":       dict(length=1.25, genal=1.3, genal_w=0.30, genal_tip=0.55, genal_curve=-6, notch=0.55, widest=0.1, horn_h=0.2, width=1.5,
                         brim_w=0.34, brim_rise=0.10, brim_curve=0.4, brim_roll=0.07, brim_pits=3),  # the horseshoe
    "phacops":      dict(length=0.95, front_exp=2.4, genal=0, corner=0.25, widest=0.2, side=0.12, dome_fill=0.95, width=1.05),    # rounded shield, no spines
    "dalmanites":   dict(SP, length=1.05, front_exp=1.7, front_point=0.22, point_w=0.07, genal=0.7, genal_w=0.11, notch=0.1, width=1.2),      # ogival, an anterior point
    "encrinurus":   dict(length=0.85, front_exp=2.3, genal=0.35, genal_w=0.10, genal_tip=0.15, genal_spread=8, notch=0.08, widest=0.03, width=1.15),
    "deiphon":      dict(length=0.8, genal=1.1, genal_w=0.13, genal_tip=0.1, genal_spread=35, genal_curve=-35, notch=0.02, widest=0.03, dome_fill=0.7, width=1.1),  # cheeks reduced to spines
    "isotelus":     dict(length=1.1, front_exp=1.8, genal=0.3, genal_w=0.12, genal_tip=0.15, notch=0.1, widest=0.03, dome_fill=0.95, width=1.15),  # a subtriangular shovel
    "neoasaphus":   dict(length=0.95, front_exp=2.1, genal=0, corner=0.3, widest=0.25, side=0.15, dome_fill=0.95, width=1.05),
    "cryptolithus": dict(length=0.85, genal=2.6, genal_w=0.07, genal_tip=0.3, genal_spread=3, notch=0.03, widest=0.03, width=1.25,
                         brim_w=0.27, brim_rise=0.26, brim_curve=0.6, brim_pits=4),   # needles three heads long
    "cyclopyge":    dict(length=1.5, front_exp=2.2, genal=0, corner=0.3, widest=0.35, side=0.3, rear_taper=0.05, width=0.95),         # long and narrow
    "walliserops":  dict(SP, length=1.0, front_exp=1.9, genal=0.6, genal_spread=5, notch=0.1, width=1.2),
    "ceraurus":     dict(length=0.75, front_exp=2.6, genal=1.2, genal_w=0.12, genal_tip=0.1, genal_spread=22, genal_curve=28, notch=0.03, widest=0.03, width=1.15),  # spines sweeping out
    "ampyx":        dict(length=0.8, genal=2.4, genal_w=0.06, genal_tip=0.3, notch=0.03, widest=0.03, front_point=1.2, point_w=0.045, width=1.2),   # three needles
    "dikelocephalus": dict(SP, length=0.8, front_flat=0.4, genal=0.45, width=1.2),                    # truncate front
    "bristolia":    dict(SP, length=0.9, genal=0.9, genal_at=-0.5, genal_spread=28, genal_curve=-22, corner=0.15, width=1.15),   # genal spines advanced up the side
    "all_on":       dict(),                                                                          # the crescent's own silhouette
}

def names(): return list(STYLES)

def style(name):
    s = STYLES[name]; return dict(s.get("face", {})), dict(s.get("params", {}))

def apply(P, name):
    """P with the style's schema overrides (coerced), and the style's face overrides for body.build(..., face=)."""
    import schema
    face, params = style(name)
    return schema.coerce(dict(P, **params), base=P), face

def outline(name): return dict(OUTLINES.get(name, {}))

def shaped_head_of(name, base="textured"):
    """The head alone on its own silhouette (print mm): shaped_head's outline + this style's face, eyes, spines."""
    import body as BODY, presets as PR
    from joints import ball as BALL
    P, face = apply(PR.params(base), name); plans = BODY.placed_plans(P)
    J, *_ = BALL.geometry(P); K = BODY.iso_kit(P); PV = BODY.pivots(plans, J, K)
    H = BODY.shaped_head(P, plans, PV, K, J, outline(name), face); H.apply_scale(1.0 / K["s"]); return H, P

def head_of(name, base="textured"):
    """The finished head alone (print mm): isopod head piece + crescent band + this style's face, eyes, spines."""
    import body as BODY, presets as PR
    from joints import ball as BALL
    P, face = apply(PR.params(base), name); plans = BODY.placed_plans(P)
    J, *_ = BALL.geometry(P); K = BODY.iso_kit(P); PV = BODY.pivots(plans, J, K)
    H, _ = BODY.isopod_head(P, plans, J, PV, K, face=face); H.apply_scale(1.0 / K["s"]); return H, P

if __name__ == "__main__":
    import numpy as np, render as R, time
    crescent = "--crescent" in sys.argv                                 # the fixed crescent on the isopod head piece (the first version)
    want = [a for a in sys.argv[1:] if not a.startswith("--")] or names(); rows = []; os.makedirs(os.path.join(HERE, "photos"), exist_ok=True); os.makedirs(os.path.join(HERE, "stl", "heads"), exist_ok=True)
    for n in want:
        t0 = time.time()
        try:
            h, P = (head_of if crescent else shaped_head_of)(n)
        except Exception as ex:
            print(f"{n}: FAILED {type(ex).__name__}: {ex}", flush=True); continue
        ok = h.is_watertight and len(h.split(only_watertight=False)) == 1
        h.export(os.path.join(HERE, "stl", "heads", f"{n}.stl"))
        print(f"{n}: watertight single body {ok}, {h.extents.round(1).tolist()} mm, {time.time() - t0:.0f} s", flush=True)
        h = h.copy(); h.apply_translation((0, -h.bounds[0][1], 0))
        rows.append([(h, 32, -62, f"{n.upper()}" + (f"  (Gon p. {STYLES[n]['p']})" if STYLES[n]['p'] else "")), (h, 90, -90, "top")])
    for i in range(0, len(rows), 5):
        out = os.path.join(HERE, "photos", f"head_styles_{i // 5 + 1}.png"); R.sheet(rows[i:i + 5], out, W=720, H=470); print(out, flush=True)
