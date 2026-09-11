# Frozen instrument-2.1 targets (OpenCascade builder), 10 Sep 2026

| preset | bevel | limited_by | reason | class | theta/joint | total | s_tail | rest max mm3 |
|---|---|---|---|---|---|---|---|---|
| agnostida | 35.0 | invalid | rest_interference | censored | None | None | None | 10.58 |
| asaphida | 45.0 | invalid | rest_interference | censored | None | None | None | 9.0 |
| corynexochida | 45.0 | closed |  | sphaeroidal | 22.58 | 203.2 | -0.0162 | 0.0 |
| harpetida | 45.0 | closed |  | double | 30.31 | 303.1 | 0.9058 | 0.0 |
| phacopida | 45.0 | invalid | unsane_parts:seg0,seg7 | censored | None | None | None | None |
| proetida | 45.0 | closed |  | double | 22.81 | 228.1 | 0.2578 | 0.0 |
| ptychopariida | 45.0 | invalid | unsane_parts:seg9 | censored | None | None | None | None |
| redlichiida | 45.0 | invalid | unsane_parts:seg2 | censored | None | None | None | 0.0 |

Not frozen: **lichida** (OOMs on the BREP builder — its ~40 marginal spines fused into a spline shell), **odontopleurida**
(booleans fail on its spines). Both are recorded here as "BREP builder cannot build" — the mesh builder has no target to
match for them, only the requirement that it builds them watertight. Their first readings will be on the new builder.

Readings: three real (proetida closed/double, corynexochida closed/sphaeroidal, harpetida closed/double) and five invalid
(agnostida and asaphida rest interference; phacopida, ptychopariida, redlichiida unsane parts). Five of eight invalid on
the BREP builder is the case for the mesh builder in one table.
