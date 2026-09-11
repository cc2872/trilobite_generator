FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt flask
COPY . .
EXPOSE 8765
CMD ["python", "web/app.py"]
