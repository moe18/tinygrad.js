# tinygrad.js remote CUDA worker

The reference server behind tinygrad.js's `cuda_remote` device. It runs your **original CUDA C++ source** on a real NVIDIA
GPU: NVRTC compiles it for the GPU's architecture and the CUDA driver API runs it. Use it when a kernel needs something the
browser backends can't provide, such as double precision on WebGPU, 64-bit atomics, native performance, or bit-exact
native results. It's standard-library Python only, talking to `libcuda` and `libnvrtc` through `ctypes`.

```
browser (tinygrad.js)  --HTTPS + session token-->  gateway (this server)  --pipes-->  session process (CUDA context) --> GPU
          ^                                              ^
          |  short-lived token                           |  TGJS_CUDA_TOKEN_SECRET (server side only)
     your backend  ------------------------------------- +
```

## Security model

- **No cloud credentials in browsers.** The worker itself needs none. Modal or RunPod API keys stay with whoever deploys
  it. Browsers hold only short-lived HMAC session tokens that **your backend** issues to signed-in users, either with
  `tgjs_cuda_worker.tokens.issue()` (Python) or `issue-token.mjs` (Node). The signing secret never leaves the servers.
- **Explicit opt-in.** tinygrad.js uses the worker only when `device: "cuda_remote"` is configured with an endpoint.
  `AUTO` device selection never picks a remote (possibly paid) GPU.
- **Isolation.** Each session is a separate child process with its own CUDA context and its own process group:
  - The process runs in an empty temp directory with a scrubbed environment.
  - It has resource limits (no core dumps, file and descriptor limits, a CPU-time cap).
  - When `TGJS_CUDA_RUN_AS_UID` is set and the gateway runs as root, it drops to that uid.
  - New device allocations are zeroed, so data from earlier sessions on the GPU can't be read.

  For untrusted users, also run the worker inside a container sandbox (for example gVisor with `--nvproxy`, or a
  dedicated VM) and on GPUs that host nothing else.
- **Limits.** Each session gets a memory limit and a per-launch (and per-memory-operation) time limit. A kernel that
  exceeds the time limit has its session process killed (`cudaErrorLaunchTimeout`). The worker also caps:
  - the number of sessions, and per-user session limits through the `maxSessions` token claim
  - source size, launches per session and request size
  - idle time (idle sessions are reaped)
  - grid and block dimensions, which are validated
- **Sessions are private to the token subject.** Other users get `404` for them.
- **CORS.** Only origins listed in `TGJS_CUDA_ALLOWED_ORIGINS` may call the worker from a browser.
- **Faults behave like native CUDA.** After an illegal address, assert or similar fault, the CUDA context is unusable.
  The session is terminated, and every later call fails with the original error.

## Run it

```bash
# self-hosted, on a machine with an NVIDIA driver + CUDA toolkit (NVRTC)
cd server/cuda-worker
export TGJS_CUDA_TOKEN_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(48))")
export TGJS_CUDA_ALLOWED_ORIGINS=https://your.app
python3 -m tgjs_cuda_worker probe                       # check that CUDA works: prints the device
python3 -m tgjs_cuda_worker serve --host 0.0.0.0 --port 8787
python3 -m tgjs_cuda_worker issue-token alice --ttl 900  # a token for testing
```

Docker (also the RunPod image): `docker build -t tgjs-cuda-worker server/cuda-worker && docker run --gpus all -p 8787:8787 -e TGJS_CUDA_TOKEN_SECRET -e TGJS_CUDA_ALLOWED_ORIGINS tgjs-cuda-worker`.

Modal: `modal deploy server/cuda-worker/modal_app.py` (see the docstring for the secret it expects). RunPod: see
`runpod/README.md`. Put the worker behind HTTPS; Modal and RunPod provide it.

| variable | default | meaning |
|---|---|---|
| `TGJS_CUDA_TOKEN_SECRET` | required | HMAC secret for session tokens (≥ 32 characters) |
| `TGJS_CUDA_ALLOWED_ORIGINS` | none | comma-separated browser origins allowed by CORS |
| `TGJS_CUDA_HOST` / `TGJS_CUDA_PORT` | `127.0.0.1` / `8787` | listen address |
| `TGJS_CUDA_MAX_SESSIONS` | 4 | concurrent sessions (one CUDA context each) |
| `TGJS_CUDA_MEMORY_BYTES` | 2 GiB | device memory per session |
| `TGJS_CUDA_TIME_LIMIT_MS` | 10000 | per kernel launch / memory operation |
| `TGJS_CUDA_SESSION_TTL_S` | 600 | idle sessions are closed after this |
| `TGJS_CUDA_MAX_LAUNCHES` | 100000 | launches per session |
| `TGJS_CUDA_RUN_AS_UID` / `_GID` | unset | drop session processes to this uid/gid (when started as root) |
| `CUDA_HOME`, `TGJS_CUDA_NVRTC` | `/usr/local/cuda` | where to find NVRTC and the CUDA headers |

Tokens can carry per-user caps, which can only lower the server's limits: `memoryBytes`, `timeLimitMs` and `maxSessions`.

## Use it from tinygrad.js

```js
import { createCudaContext } from "tinygrad.js/cuda";
const { token } = await (await fetch("/api/cuda-token")).json();      // your backend, after sign-in
const ctx = await createCudaContext({ device: "cuda_remote", remote: { endpoint: "https://cuda.your.app", token } });
```

Your backend's token route can be this small:

```js
import { issueSessionToken } from "./issue-token.mjs";
app.get("/api/cuda-token", requireUser, (req, res) => res.json({ token: issueSessionToken(process.env.TGJS_CUDA_TOKEN_SECRET, req.user.id, { ttlS: 900, memoryBytes: 1 << 30 }) }));
```

## Protocol

This is the protocol in `src/cuda/runtime/remote.ts`. All routes except `/v1/health` need `Authorization: Bearer <token>`.

| route | body | result |
|---|---|---|
| `POST /v1/sessions` | `{gpu, memoryLimit, timeLimitMs}` | `{session, device: {name, totalMem, major, minor, ...}, limits}` |
| `POST /v1/sessions/:s/alloc` | `{handle, bytes}` | `{}` |
| `PUT /v1/sessions/:s/mem/:handle?offset=N` | raw bytes | `{}` |
| `GET /v1/sessions/:s/mem/:handle?offset=N&bytes=M` | | raw bytes |
| `POST /v1/sessions/:s/free` | `{handle}` | `{}` |
| `POST /v1/sessions/:s/copy` | `{dst, dstOffset, src, srcOffset, bytes}` | `{}` |
| `POST /v1/sessions/:s/memset` | `{handle, offset, value, bytes}` | `{}` |
| `POST /v1/sessions/:s/launch` | `{source, kernel, options: {defines}, grid, block, sharedMem, args, timing}` | `{ms, printf}` or `{printf, error}` |
| `DELETE /v1/sessions/:s` | | `{}` |

In a launch, `kernel` is an NVRTC name expression: a template instantiation such as `"reduce<256>"` works without an
explicit instantiation. Each entry of `args` is `{ptr: {handle, offset}}` or `{scalar: {ctype, bytes: base64}}`, holding
the argument's little-endian bytes exactly as `cuLaunchKernel` expects them. Errors are
`{error: {name: "cudaError…", message}}`.

## Tests

`tests/cuda/remote.node.test.ts` starts this server as a real process. It covers tokens, CORS, the limits, the timeout
kill, sticky faults, session ownership and validation. Test machines have no NVIDIA GPU, so the tests swap in a
**test-only** engine (`tests/cuda/remote/wasm_engine.py`) that runs the kernels with tinygrad.js's WebAssembly backend.
`tgjs_cuda_worker/cuda.py`, the NVRTC and driver-API engine, can only run on NVIDIA hardware. Validate it there with
`python3 -m tgjs_cuda_worker probe` and the same tests, run against it: `TGJS_CUDA_TEST_NATIVE=1 npx playwright test --project=cuda-node tests/cuda/remote.node.test.ts`.
