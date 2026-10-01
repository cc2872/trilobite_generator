FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt flask
COPY . .
# Pre-bake the isopod model (the default build) for every preset so the deployed site never runs a geometry build
# on a visitor's request — each would be ~2-9 min and up to ~3 GB RAM. This step is resumable: if the build context
# already carried a warm web/cache it is instant; on a clean host it bakes all presets here (~40 min, needs ~3 GB RAM
# on the BUILD host — build the image where there is memory, then ship it). The cache keys are deterministic
# (param hash + ISOPOD_SIG), so a baked cache is served as-is at runtime.
RUN python web/warm_cache.py
ENV PORT=8765
EXPOSE 8765
CMD ["python", "web/app.py"]
