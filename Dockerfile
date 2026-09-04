FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglu1-mesa libxrender1 libxext6 libxi6 libxcursor1 libxft2 libxinerama1 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir build123d
COPY trilobite.py trilobite_web.py index.html ./
ENV PORT=8080
EXPOSE 8080
CMD ["python", "trilobite_web.py"]
