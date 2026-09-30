"""
anatomy/heads.py: every head, each in its own file, all on the same contract (tests/test_head_contract.py).

    classic   anatomy/head.py. The instrument's head: every frozen reference and the pre-registered sweep use it.
    crescent  anatomy/head_crescent.py (27 Sep 2026). The rear edge is one U that runs on as the horns' inner edge.

A head is a build option, like a joint, not a parameter: it is chosen by name (assemble's head=...), so adding one
changes no parameter hash. get(None) is the classic head.
"""
NAMES = ("classic", "crescent")

def get(name=None):
    if name in (None, "classic"):
        from anatomy import head as h
    elif name == "crescent":
        from anatomy import head_crescent as h
    else:
        raise KeyError(f"unknown head {name!r} (have: {', '.join(NAMES)})")
    return h

def is_head(mod):
    return mod.__name__ in ("anatomy.head", "anatomy.head_crescent")
