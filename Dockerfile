# Dockerfile for ml-spam-email-classifier

FROM python:3.11-slim

# Avoid Python buffering and cache
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install OS packages if needed (kept minimal here)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Train the spam classifier once at build time
RUN python -m src.train_spam_classifier

# Expose FastAPI port
EXPOSE 8000

# Default command: run API with uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
