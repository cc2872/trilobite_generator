# Trilobite morphospace generator v4 — container for Render / Railway / Fly.io
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglu1-mesa libxrender1 libxext6 libxi6 libxcursor1 libxft2 libxinerama1 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir build123d trimesh manifold3d python-fcl numpy rtree matplotlib
COPY schema.py fields.py trilobite.py instrument.py trilobite_web.py index.html skins.py blueprint.py ./
COPY skins/ ./skins/
COPY presets/ ./presets/
ENV PORT=8080
EXPOSE 8080
CMD ["python", "trilobite_web.py"]
