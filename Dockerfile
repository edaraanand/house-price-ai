# ---- Training / evaluation / promotion pipeline image ----
# Built and used by the Argo Workflows ML pipeline (train.py / evaluate.py /
# promote.py). Build with: docker buildx build --target pipeline ...
# This is a separate stage from the BentoML serving image below, so the
# default `docker build .` (no --target) is unaffected and still produces
# the serving image, matching the existing CI workflow.
FROM python:3.11-slim AS pipeline

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    SKLEARN_DATA_HOME=/home/appuser/scikit_learn_data

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY multi_train.py tune.py evaluate.py promote.py config.yaml ./

# Create non-root user and dataset cache directory.
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /home/appuser/scikit_learn_data \
    && chown -R appuser:appuser /app /home/appuser

USER appuser

# Pre-download the California Housing dataset into the image.
#
# This happens at IMAGE BUILD TIME, not during every tuning worker.
# RUN python - <<'PY'
# from sklearn.datasets import fetch_california_housing

# data = fetch_california_housing(
#     data_home="/home/appuser/scikit_learn_data",
#     download_if_missing=True,
# )

# print(
#     f"California Housing dataset ready: "
#     f"{data.data.shape}"
# )
# PY

# No CMD/ENTRYPOINT: the Argo Workflow steps supply the command
# (python train.py / python evaluate.py ... / python promote.py ...)


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
CMD ["bentoml", "serve", "serving.service:HousingModelService", "--host", "0.0.0.0", "--port", "3000"]
