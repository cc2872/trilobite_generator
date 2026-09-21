"""read_sheet_spec.py — recover a trilobite's full specification from a blueprint PNG.

The sheet embeds its spec (params + hash + schema + instrument) as a PNG tEXt chunk (blueprint.sheet). This reads it
back, so any saved/shared sheet is self-describing — no cache lookup or slider re-entry needed.

    python scripts/read_sheet_spec.py sheet.png                 # print the spec JSON
    python scripts/read_sheet_spec.py sheet.png params.json     # also write just the params to params.json
"""
import sys, json
from PIL import Image


def read_spec(png_path):
    """Return the embedded spec dict ({hash, schema, instrument, params}) or None if the PNG carries none."""
    info = Image.open(png_path).info          # PNG tEXt chunks land in .info / .text
    raw = info.get("trilobite")
    return json.loads(raw) if raw else None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/read_sheet_spec.py sheet.png [params.json]")
    spec = read_spec(sys.argv[1])
    if spec is None:
        sys.exit("no trilobite spec embedded in this PNG (older sheet, or not a blueprint)")
    print(json.dumps(spec, indent=2, sort_keys=True))
    if len(sys.argv) > 2:
        json.dump(spec.get("params", {}), open(sys.argv[2], "w"), indent=2, sort_keys=True)
        print(f"\n-> wrote params to {sys.argv[2]}", file=sys.stderr)
