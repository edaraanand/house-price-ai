FROM python:3.11-slim

# Python runtime settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy serving application
COPY serving/ ./serving/

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

# BentoML serving port
EXPOSE 3000

# Start BentoML server
CMD ["bentoml", "serve", "serving.service:HousePriceServing", "--host", "0.0.0.0", "--port", "3000"]

# FROM python:3.11-slim

# ENV PYTHONUNBUFFERED=1
# ENV PYTHONDONTWRITEBYTECODE=1

# WORKDIR /app

# # System packages needed by scientific Python packages
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#         build-essential \
#         curl && \
#     rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .

# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# COPY src/ ./src/

# CMD ["python", "src/train.py"]
