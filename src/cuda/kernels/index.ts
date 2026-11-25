/**
 * The CUDA kernel library: real CUDA C sources compiled by tinygrad.js's compiler (they double as its test corpus).
 * Each entry is plain CUDA source that runs unmodified with nvcc as well.
 */
import * as F from "./fundamentals.js";
import * as N from "./nn.js";
import * as T from "./transformer.js";
import * as A from "./advanced.js";
import * as R from "./training.js";

export const sources = {
  vectorAdd: F.vectorAdd, matmul: F.matmul, transpose: F.transpose, reduction: F.reduction, scan: F.scan, histogram: F.histogram,
  activations: N.activations, softmax: N.softmax, norms: N.norms, conv: N.conv, embedding: N.embedding, crossEntropy: N.crossEntropy,
  attention: T.attention, rope: T.rope, kvcache: T.kvcache,
  quantized: A.quantized, sparse: A.sparse, moe: A.moe,
  optimizers: R.optimizers, mixedPrecision: R.mixedPrecision,
} as const;

export type KernelSourceName = keyof typeof sources;

/** One-line descriptions for the playground and docs. */
export const catalog: Record<KernelSourceName, string> = {
  vectorAdd: "vector_add, vector_add4 (float4), saxpy, elementwise, fill, scale",
  matmul: "matmul_naive, matmul_tiled<TILE> (shared memory), matmul_regtile<BM,BN,BK,TM,TN> (register blocking), matmul_batched",
  transpose: "transpose_naive, transpose_tiled (padded shared tile, coalesced)",
  reduction: "reduce_sum<BLOCK> (shared tree), reduce_sum_warp<BLOCK> (shuffles), row_max<BLOCK>",
  scan: "scan_block<BLOCK> + scan_add<BLOCK> (multi-block inclusive scan), scan_blelloch<BLOCK> (work-efficient)",
  histogram: "histogram256 (shared-memory atomics), histogram_float",
  activations: "relu/gelu/silu forward and backward, bias_act (fused bias + activation)",
  softmax: "softmax_rows<BLOCK> (warp shuffles), softmax_rows_bwd<BLOCK>",
  norms: "layernorm_fwd/bwd<BLOCK>, rmsnorm_fwd/bwd<BLOCK>",
  conv: "conv2d_fwd (direct), maxpool2d_fwd/bwd, avgpool2d_fwd",
  embedding: "embedding_fwd, embedding_bwd (atomic scatter-add)",
  crossEntropy: "cross_entropy_fwd_bwd<BLOCK> (fused softmax + NLL + gradient)",
  attention: "sdpa_naive (causal, grouped-query), flash_attention_fwd<BR,BC,DMAX> (online softmax, shared K/V tiles)",
  rope: "rope_apply, rope_bwd (rotary position embeddings)",
  kvcache: "kv_cache_append, add_layernorm<BLOCK> (fused residual + LayerNorm)",
  quantized: "int8_matmul, int8_matmul_dp4a, quantize_rows_int8<BLOCK>, int4_matmul (grouped scales), matmul_f16",
  sparse: "csr_spmv, csr_spmm",
  moe: "moe_topk (routing), moe_dispatch (capacity-limited scatter), moe_combine",
  optimizers: "sgd_step (momentum, nesterov, weight decay), adam_step (Adam/AdamW), sumsq<BLOCK> + clip_by_norm, accumulate",
  mixedPrecision: "fp16/bf16 casts, unscale_check (loss scaling overflow detection), master_step",
};
