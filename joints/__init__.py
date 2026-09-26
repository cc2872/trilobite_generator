"""
joints: every way two parts can be connected, each in its own file, all on the same template (joints/base.py).

    pin    the measured joint. Interleaving knuckles on a pin with a ventral bevel stop. The instrument's ruler.
    flexi  print-in-place only. Thingiverse #3839472 form: convex lobe + through-bore, concave face + barrel on a
           round neck, 0.4 mm faces. Never measured.

get(name) returns the joint module. Adding a joint = adding a file here that satisfies joints/base.py.
"""
def get(name):
    if name == "pin":   from joints import pin as j
    elif name == "flexi": from joints import flexi as j
    else: raise KeyError(f"unknown joint {name!r} (have: pin, flexi)")
    return j
