"""
joints/base.py: the template every joint module follows. Documentation, not a base class (the modules are plain
functions so a joint file is readable on its own). tests/test_joint_contract.py checks each joint against this.

A joint module defines:

    NAME            "pin", "flexi", ...
    MEASURED        True only for the instrument's joint. A measured joint may not change without a new instrument
                    version and a re-freeze of tests/references.
    BASE            "shell" or "solid": what the joint is cut into. The anatomy modules provide both.

    fits(P)         raise ValueError with a plain reason if no joint of this kind fits the animal (pitch too short..)
    pivot(P)        dict(z=..., y_beyond_plane=...): where the chain bends, for the blueprint and the site
    stop_deg(P)     the printed/built stop per joint, degrees

    cut(body, env, P, port, **opts) -> (body, T)
                    apply the joint to one edge of one part. body is the shell or solid (per BASE) in the plan
                    frame; env is the under-envelope (webs are clipped to it). port is an anatomy.Port. T is the
                    4x4 that maps the plan frame to the returned part's frame (identity unless the joint moved
                    the plate; the assembler puts ornaments through it). Options: bevel_deg (pin: the instrument
                    passes its own fixed bevel; None = the printed stop).

    overhang(P, part, S, port, opts) -> mesh or None
                    what this joint's cut removed that the assembler should put back and clear against the parts
                    behind (a print joint that trims to one pitch returns the pleurae beyond it). Pin: None.

    pose(P, angles_deg) -> [4x4 per part]
                    the chain bent by angles_deg[k] at joint k, head = identity. The instrument sweeps this.
    transforms_deg(P, theta) = pose(P, [theta] * n_joints)   (kept for the callers that bend uniformly)

    print_keepout(P, port) -> list of trimesh boxes   (optional)
                    regions a shell thickener must leave alone (barrels, bores, knuckles).
"""
