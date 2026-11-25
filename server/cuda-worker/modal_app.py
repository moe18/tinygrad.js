"""Deploy the tinygrad.js remote CUDA worker on Modal:  modal deploy modal_app.py

Prerequisites (on your machine or CI, never in a browser):
  modal secret create tgjs-cuda-worker TGJS_CUDA_TOKEN_SECRET=<32+ random chars> TGJS_CUDA_ALLOWED_ORIGINS=https://your.app

Sessions live in the worker's memory, so all requests of a session must reach the same container: the function is
limited to one container that serves many concurrent requests. Raise TGJS_CUDA_MAX_SESSIONS / pick a bigger GPU to scale
up, or deploy several apps behind your own session-affine router to scale out. Modal API tokens stay with whoever deploys;
browsers only get short-lived session tokens from your backend (tgjs_cuda_worker.tokens.issue / issue-token.mjs).
"""
import os, subprocess
import modal

GPU = os.environ.get("TGJS_MODAL_GPU", "A10G")
image = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
         .env({"CUDA_HOME": "/usr/local/cuda"})
         .add_local_python_source("tgjs_cuda_worker"))
app = modal.App("tgjs-cuda-worker")

@app.function(image=image, gpu=GPU, secrets=[modal.Secret.from_name("tgjs-cuda-worker")], max_containers=1,
              scaledown_window=300, timeout=24 * 3600)
@modal.concurrent(max_inputs=64)
@modal.web_server(8787, startup_timeout=120)
def worker():
  subprocess.Popen(["python", "-m", "tgjs_cuda_worker", "serve", "--host", "0.0.0.0", "--port", "8787"])
