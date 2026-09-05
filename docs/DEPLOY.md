# Deploy (Render, Docker)

1. GitHub: a repo whose ROOT contains exactly the contents of website/ (Dockerfile, *.py, index.html,
   requirements.txt, assets/, tests/). Drag the `assets` FOLDER into the uploader so the folder is created.
   Stay on branch `main`; delete any other branch.
2. Render: New → Web Service → connect the repo → Language: Docker → Instance: Standard (2 GB). Free/Starter
   (512 MB) is killed by the CAD kernel. No environment variables needed (PORT is set by Render).
3. First build 5–10 min. Log ends with "Trilobite generator at http://localhost:10000" and the pill turns Live.
4. Verify: https://<name>.onrender.com/api/config contains "eyeType"; /api/health lists cached builds.
5. A build takes 1–2 min on Render. Meshes are served from memory; after a restart press Build again.

Failure signatures: a 15–40 s failed deploy = a COPY line can't find a file (usually assets/ missing or at the
wrong path). "Killed" in the log = out of memory → bigger instance. Gateway timeout on Build = move the build
to a background job with polling (not yet implemented).
