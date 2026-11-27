# Self-hosted / RunPod GPU pod image for the tinygrad.js remote CUDA worker.
# The devel image provides NVRTC and the CUDA headers (curand_kernel.h, cuda_fp16.h, ...); the driver (libcuda) comes
# from the host through the NVIDIA container runtime (docker run --gpus all ...).
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*
RUN useradd --system --uid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin cudaworker
COPY tgjs_cuda_worker /opt/tgjs/tgjs_cuda_worker
ENV PYTHONPATH=/opt/tgjs CUDA_HOME=/usr/local/cuda TGJS_CUDA_HOST=0.0.0.0 TGJS_CUDA_PORT=8787
USER 10001
EXPOSE 8787
# TGJS_CUDA_TOKEN_SECRET and TGJS_CUDA_ALLOWED_ORIGINS must be provided at run time (never baked into the image)
CMD ["python3", "-m", "tgjs_cuda_worker", "serve"]
