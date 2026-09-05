# The enrollment study (planned)

Question: how far can a given body plan curl before its own anatomy stops it? Enrollment capacity as a
function of morphology, with the joint held constant.

Measures (skin.measure): e_max (largest collision-free uniform enrollment), free curl in degrees,
closure gap, class (none / partial / complete).

Pre-registered predictions: free curl falls with pleural spine length past a threshold set by sweep angle;
closure needs total curl near 360° and so scales with segment count (Esteve et al. 2017); macropygous forms
close more easily; large eyes cost head-joint curl unless forward; a terminal spine blocks completion.

Validation set: Phacops (complete enroller), calymenids and agnostids (complete), odontopleurids (spines out),
harpetids (poor/partial), olenellids (poor). Presets coded as parameter vectors; disagreements between the
model and the literature are findings.

Design: Latin hypercube over six primary axes (pygidium size, effacement, spinosity, eye size, outline
elongation, segment count), ~300 points, one row per point in a dataset (params, measures, instrument version).
Run overnight on a laptop. Then print three specimens — a champion, a poor one, a spiny one — and roll them.

References: Esteve et al. 2017 Palaeontology (enrolment kinematics from microCT); Beech et al. 2026 Proc B
(theoretical morphospace generator for harpid brims; the methodological template); NYT 2014 "Variations on a
Theme" (Gon III's 50 genera).
