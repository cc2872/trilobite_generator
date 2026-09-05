---
title: Trilobite Morphospace Generator
emoji: 🪲
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---

# Trilobite Morphospace Generator

A parametric, physically-enrollable trilobite. Every slider is one axis of the body plan
(Raup's coiling model, but for a segmented body that has to *roll up*). Built with
[build123d](https://github.com/gumyr/build123d); the joints are filament-pin hinges with
ventral stop bevels, and the collision check proves the stops for any setting.

`trilobite.py` is the generator (runs standalone, exports STLs); `app.py` is this UI.
