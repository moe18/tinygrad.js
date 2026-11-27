/** Bundled into a test page by tests/cuda/*.browser.test.ts: exposes the CUDA runtime and the shared test cases. */
import * as cuda from "../../src/cuda/index.js";
import { CASES } from "./cases.js";
import { randomKernels } from "./random-kernels.js";
(window as unknown as Record<string, unknown>).cuda = cuda;
(window as unknown as Record<string, unknown>).CASES = CASES;
(window as unknown as Record<string, unknown>).randomKernels = randomKernels;
(window as unknown as Record<string, unknown>).cudaReady = true;
