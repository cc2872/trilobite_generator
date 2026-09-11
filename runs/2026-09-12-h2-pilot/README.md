# H2 pilot — 12 Sep 2026

- `design.csv` — the 60-animal LHS design (`sweep.py design --n-per-order 8 --n-control 20 --seed 7`); 100 rows, 60 non-control used.
- `rows_before_fix.csv` — all 60 animals, pre-fix builder. 73% invalid, 0 open.
- `rows_after_fix_53of60.csv` — 53 of the 60 (tool budget), post-fix builder. First open animals.
- `SUMMARY.txt` — before/after on the same 53.

`idx` in the rows files indexes `design.csv`. Real instrument, 121x61 print grid. Not a result; a pilot that
shows the instrument can now see the jam region. Finish the last 7 rows before tabulating per-order vs the 40% gate.
