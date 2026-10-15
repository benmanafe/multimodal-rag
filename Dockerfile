FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so Docker caches this layer until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the code is baked in. Secrets come from .env at runtime; documents from a mounted ./data.
COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
