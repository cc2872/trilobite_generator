"""tests/test_characters.py: schema.CHARACTERS reads and writes the guide's characters consistently."""
import json, glob, os, pytest, schema
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@pytest.mark.parametrize("key", sorted(schema.CHARACTERS))
def test_express_then_read_back(key):
    P = schema.express(schema.table_defaults(), key)
    assert key in schema.characters(P), key

def test_only_known_parameters():
    for k, c in schema.CHARACTERS.items():
        for p in c["set"]: assert p in schema.BY_KEY, (k, p)
        assert isinstance(c["p"], int) and c["name"]

def test_suture_types_are_exclusive():
    for k in ("proparian", "gonatoparian", "opisthoparian"):
        got = [c for c in schema.characters(schema.express(schema.table_defaults(), k)) if c.endswith("parian")]
        assert got == [k]

def test_eye_types_are_exclusive():
    for k in ("holochroalEyes", "schizochroalEyes", "eyesLost"):
        got = [c for c in schema.characters(schema.express(schema.table_defaults(), k)) if c in ("holochroalEyes", "schizochroalEyes", "eyesLost")]
        assert got == [k]

def test_every_preset_reads_without_error():
    for f in glob.glob(os.path.join(ROOT, "presets", "*.json")):
        assert isinstance(schema.characters(json.load(open(f))["params"]), list)
