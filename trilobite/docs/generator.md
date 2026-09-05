# trilobite_generator.html — interactive parametric generator

Open in a browser. Drag to rotate, scroll to zoom. Every control rebuilds the geometry.

- **Body length / width / segments** — sliders; the whole animal rescales, segments re-space.
- **Head length** — slider; the cephalon re-proportions (it gets genuinely longer, not just scaled),
  and the glabella stretches with it. **Glabella rise** is a second head slider.
- **Lenses per eye** — a whole-number field (+/− or type a value). The eyes are *generated* from this
  count, not a pasted asset: **Schizochroal** packs fewer large domes in dorsoventral files;
  **Holochroal** packs many small domes densely. 0 = blind.
- **Enrollment** — slider; the pin-hinge chain curls (head fixed, segments rotate about their hinges).
- **Legs** — checkbox, default on; one biramous pair per segment, curling with their segment.

`gen_preview.png` is a static render of the same geometry math (via `gen_preview.py`) at four settings —
proof the body, furrows, head-reshape, procedural eyes, legs and enrollment behave, independent of the
browser.

## Mapping to the sculpt-asset pipeline
This is the procedural control surface. To drive the *real sculpt meshes* the same way, the parameters
line up with `schema.py`: `length`, `width`, `segCount`, `cephFrac` (head length), `glabRise`, `enroll`
(via maxAngle), `eyeType`, and `legs`. Two pieces still to wire on the mesh side: a procedural eye
generator that emits N sculpted lenses (default = the extracted asset eye, N>0 = generated), and the
legs asset toggle. The head already re-proportions through `wrap.py`'s s-map.
