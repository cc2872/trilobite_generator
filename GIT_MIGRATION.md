# From v1 (OpenCascade builder, dials site) to 6.0 — what changed, and how to commit it

## What this tree is
The same generator, rebuilt without OpenCascade. Height fields + Manifold booleans (`mesh.py`), the animal's parts
with their a·b·c cells (`parts.py`), instrument 2.1 in one file (`instrument.py`), schema 6.0 with the 3x3 CELLS
contract (`schema.py`), the sweep (`sweep.py`), the pre-registered analysis (`analyze_sweep.py`), the blueprint and
its gallery (`blueprint.py`), a Flask site with visual controls (`web/`), and a test suite that ties every reading to
values frozen from the old builder (`tests/`). The old builder and site are in `legacy/` for reference only.

## What changed from v1, in the order it matters
1. **Readings are the same.** proetida 22.81 / 22.81, corynexochida 22.66 / 22.58, harpetida 30.39 / 30.31 deg vs the
   frozen instrument-2.1 runs on the OpenCascade builder (`tests/references/`). Whole animals now build in 6-8 s
   instead of 100-240 s, with no retries and no grid jitter — the noise floor is the 0.1 deg bisection resolution.
2. **Two defects of the old builder are gone.** Its B-spline fit sat up to 1.6 mm below the exact surface on domed
   heads (phacopida 12 % lighter). Its retry loop hid a bevel-band bug that severed narrow segments' pleural tips
   (harpetida seg7); `parts._hinge_geometry` caps seg-seg bands 3 mm short of the local tip.
3. **Symmetric by construction.** Half grid mirrored; solid-eye heads are built as the right half + mirror.
4. **Schema 6.0.** 95 keys. Removed: skins, tubercles/ornament, seed, eyeElong. Added: prongs (tines, length, stem,
   splay, centre, width, curl) on head and tail; glabFront. `CELLS` maps every key to exactly one of frame / b1 / a1 /
   b2 / a2 / b3 / a3 / ruler; `migrate()` reports dead keys; presets are stamped `"schema": "6.0"`.
5. **Glabella is ovoid**, not a bar (rounded nose, bulged sides, soft rear corners).
6. **Instrument gates are stricter and honest.** A part must be ONE closed body; rest interference above 1 mm3 censors;
   the bevel actually built is recorded; the closed pose is classified (sphaeroidal / double / spiral / discoidal).
7. **The site** is the 3x3 grid with visual controls and the curl animation; `e_max` and the dials are gone. Same
   port (8765) so the trilomorph.org tunnel and the GitHub Actions image keep working.
8. **Deleted:** batch_v2, build_v2, build_parts_only, measure_from_stl, presets_resave, eyes, eye_solid, eye_blueprint,
   instrument2; `requirements.txt` has no OCC. Old outputs, scan STLs, skins and PNGs are not in the tree.

## How to commit — step by step
This replaces the working tree wholesale. Do it on a branch so v1 stays reachable by tag.

    git checkout main
    git tag v1-opencascade                      # the last v1 commit, kept forever
    git push origin v1-opencascade
    git checkout -b rewrite-6.0

    # 1. remove everything the old tree tracked except .git and .github
    git rm -r -q --cached . && git clean -fdx -e .git -e .github

    # 2. unzip the mega zip INTO the repo root (it contains a tg/ folder; move its contents up)
    unzip -q trilobite_generator_2026-09-11_mega.zip && rsync -a tg/ ./ && rm -rf tg

    # 3. large frozen references: keep them, but out of the diff noise
    git add tests/references                    # ~130 MB once; never regenerated (needs the old builder)
    git commit -m "tests: freeze instrument-2.1 references from the OpenCascade builder (10 Sep 2026)"

    # 4. the rewrite itself, one commit per layer so the history reads like the changelog
    git add mesh.py tests/test_mesh.py && git commit -m "mesh.py: height-field shells + Manifold booleans, symmetric by construction; hinge port"
    git add parts.py fields.py tests/test_parts.py tests/test_head.py && git commit -m "parts.py: head, rows, tail on the mesh builder; a/b/c cells; prongs; eye solid; ovoid glabella; local-width bevel band"
    git add schema.py presets tests/test_schema.py && git commit -m "schema 6.0: CELLS 3x3 contract, prongs, glabFront; ornament/skins/seed removed; presets migrated"
    git add instrument.py sweep.py analyze_sweep.py tests/test_regression.py tests/sanity_animals.py tests/noise_floor.py && git commit -m "instrument 2.1 on the mesh builder in one file; sweep presets/design/run; sanity animals; noise floor"
    git add blueprint.py && git commit -m "blueprint: gallery mode (absorbs sheet10)"
    git add web Dockerfile requirements.txt tests/test_web.py && git commit -m "web: Flask site on the 3x3 grid with visual controls and the curl animation; port 8765"
    git add legacy && git commit -m "legacy: OpenCascade builder, instrument2, v5 site, skins, sheet10 kept for reference"
    git add README.md CHANGES_2026-09-10.md GIT_MIGRATION.md PREREG_v1_sweep.md tests/README.md .gitignore && git commit -m "docs: README, changelog, migration notes, pre-registration"

    # 5. prove it, then merge and tag
    python -m pytest tests -q                   # ~8 min
    git checkout main && git merge --no-ff rewrite-6.0 -m "Rewrite 6.0: OpenCascade out, mesh builder + instrument 2.1 + 3x3 site"
    git tag -a generator-6.0 -m "generator 6.0 / instrument 2.1; readings reproduce v1 references"
    git push origin main --tags

    # 6. the pre-registration tag comes AFTER you resolve the [DECISION] lines in PREREG_v1_sweep.md
    git commit -am "prereg: decisions resolved" && git tag -a prereg-v1 -m "pre-registered sweep design" && git push --tags
    # then deposit the tagged tree on Zenodo and put the DOI in the pre-reg header

## After the push
GitHub Actions builds the image from the new Dockerfile (python:3.12-slim, no OCC); Watchtower on the lab box pulls it.
The container listens on 8765 as before. First request after a deploy builds the default preset (~8 s).
