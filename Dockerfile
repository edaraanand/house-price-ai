FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY serving/ ./serving/
COPY src/ ./src/

EXPOSE 3000

CMD ["bentoml", "serve", "serving.service:HousingModelService", "--host", "0.0.0.0", "--port", "3000"]

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
