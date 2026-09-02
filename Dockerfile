# Hugging Face Spaces (Docker SDK) / any container host.
# Builds the feature index at image-build time so the first request is fast.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces expects a non-root user with uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    U2NET_HOME=/home/user/.u2net \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY --chown=user . .

# Pre-download Fashion-CLIP + U^2-Net and build outputs/index.pkl into the image
RUN python precompute.py

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
