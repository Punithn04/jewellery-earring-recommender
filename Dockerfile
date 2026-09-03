# Container image for the live FastAPI app (necklace picker + upload + Fashion-CLIP
# inference). Works on Google Cloud Run, Fly.io, Render, or any Docker host.
#
# The Fashion-CLIP + U^2-Net weights and the feature index are baked in at build
# time, so a cold start only has to *load* them (~15-30s), not download them.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    HF_HOME=/app/.cache/huggingface \
    U2NET_HOME=/app/.cache/u2net \
    DISABLE_REMBG=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY . .

# Download models + build outputs/index.pkl into the image
RUN python precompute.py

# From here on the models are baked in — never touch the network at runtime
# (an anonymous Hugging Face Hub check on cold start can otherwise stall 60-90s).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8080
# shell form so ${PORT} (set by Cloud Run / the platform) is expanded
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
