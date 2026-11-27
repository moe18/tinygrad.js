# RunPod

The session protocol is stateful (device memory lives in the worker between requests), so the worker runs on a
**RunPod GPU Pod**, not on RunPod Serverless (whose request/response jobs can land on different workers).

1. Build and push the image: `docker build -t <registry>/tgjs-cuda-worker server/cuda-worker && docker push <registry>/tgjs-cuda-worker`
2. Create a Pod template with that image, **HTTP port 8787** exposed, and these environment variables
   (use RunPod *secrets* for the first one): `TGJS_CUDA_TOKEN_SECRET`, `TGJS_CUDA_ALLOWED_ORIGINS=https://your.app`,
   optionally `TGJS_CUDA_MAX_SESSIONS`, `TGJS_CUDA_MEMORY_BYTES`, `TGJS_CUDA_TIME_LIMIT_MS`.
3. Start a Pod; the endpoint is `https://<pod-id>-8787.proxy.runpod.net`.

Your RunPod API key is only used by you (or your backend) to manage pods. Browsers get short-lived session tokens from
your backend and talk to the pod's HTTPS proxy URL.
