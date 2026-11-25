/**
 * Device memory abstraction shared by all backends: allocations with ownership (which device/context created them),
 * byte sizes, element types for tensors, and lifetimes (freed allocations are poisoned so use-after-free is an error, not
 * silent corruption). Pointers are (allocation, byte offset) pairs, so `ptr.add(n)` works like C pointer arithmetic while
 * every kernel argument can still be validated against its allocation's bounds.
 */
import { CudaError } from "./errors.js";

export type DType = "float32" | "float16" | "bfloat16" | "float64" | "int8" | "uint8" | "int16" | "uint16" | "int32" | "uint32" | "int64" | "uint64" | "bool";
export const DTYPE_SIZE: Record<DType, number> = {
  float32: 4, float16: 2, bfloat16: 2, float64: 8, int8: 1, uint8: 1, int16: 2, uint16: 2, int32: 4, uint32: 4, int64: 8, uint64: 8, bool: 1,
};

let nextAllocId = 1;

export interface AllocationOwner { readonly id: number; readonly kind: string }

export class Allocation {
  readonly id = nextAllocId++;
  freed = false;
  /** backend-specific storage: a GPUBuffer (WebGPU), a byte address (WASM), a remote handle */
  constructor(readonly owner: AllocationOwner, readonly size: number, readonly native: unknown, readonly label = "") {}
  check(): void {
    if (this.freed) throw new CudaError("cudaErrorInvalidDevicePointer", `use of freed device memory (allocation #${this.id}${this.label ? ` '${this.label}'` : ""})`);
  }
}

export class DevicePtr {
  constructor(readonly alloc: Allocation, readonly offset = 0) {
    if (offset < 0 || offset > alloc.size) throw new CudaError("cudaErrorInvalidValue", `pointer offset ${offset} outside allocation of ${alloc.size} bytes`);
  }
  /** pointer arithmetic in bytes */
  add(bytes: number): DevicePtr { return new DevicePtr(this.alloc, this.offset + bytes); }
  /** bytes from this pointer to the end of its allocation */
  get remaining(): number { return this.alloc.size - this.offset; }
  get device(): AllocationOwner { return this.alloc.owner; }
  toString(): string { return `DevicePtr(#${this.alloc.id}+${this.offset}, ${this.remaining} bytes)`; }
}

export interface MemoryStats { allocations: number; bytesAllocated: number; peakBytes: number; bytesReserved: number; limit: number }

/** Tracks allocations of one device and enforces its memory limit. */
export class AllocationTracker {
  private live = new Set<Allocation>();
  bytes = 0;
  peak = 0;
  constructor(readonly owner: AllocationOwner, public limit: number) {}
  register(a: Allocation): void {
    this.live.add(a);
    this.bytes += a.size;
    this.peak = Math.max(this.peak, this.bytes);
  }
  reserve(size: number): void {
    if (!Number.isFinite(size) || size < 0 || !Number.isInteger(size)) throw new CudaError("cudaErrorInvalidValue", `invalid allocation size ${size}`);
    if (this.bytes + size > this.limit) throw new CudaError("cudaErrorMemoryAllocation", `allocating ${size} bytes would exceed the device memory limit (${this.bytes} of ${this.limit} bytes in use)`);
  }
  release(a: Allocation): void {
    if (!this.live.delete(a)) return;
    a.freed = true;
    this.bytes -= a.size;
  }
  validate(p: DevicePtr, bytes: number, what: string): void {
    p.alloc.check();
    if (p.alloc.owner !== this.owner) throw new CudaError("cudaErrorInvalidDevicePointer", `${what}: pointer belongs to another device (${p.alloc.owner.kind} #${p.alloc.owner.id})`);
    if (bytes < 0 || p.offset + bytes > p.alloc.size) throw new CudaError("cudaErrorInvalidValue", `${what}: ${bytes} bytes at offset ${p.offset} exceed the allocation of ${p.alloc.size} bytes`);
  }
  get count(): number { return this.live.size; }
  all(): Allocation[] { return [...this.live]; }
}

/** Typed host view of raw bytes for a dtype. */
export function typedView(dtype: DType, buf: ArrayBufferLike, byteOffset = 0, length?: number): ArrayBufferView {
  switch (dtype) {
    case "float32": return new Float32Array(buf, byteOffset, length);
    case "float64": return new Float64Array(buf, byteOffset, length);
    case "int8": return new Int8Array(buf, byteOffset, length);
    case "uint8": case "bool": return new Uint8Array(buf, byteOffset, length);
    case "int16": return new Int16Array(buf, byteOffset, length);
    case "uint16": case "float16": case "bfloat16": return new Uint16Array(buf, byteOffset, length);
    case "int32": return new Int32Array(buf, byteOffset, length);
    case "uint32": return new Uint32Array(buf, byteOffset, length);
    case "int64": return new BigInt64Array(buf, byteOffset, length);
    case "uint64": return new BigUint64Array(buf, byteOffset, length);
  }
}

export function bytesOf(v: ArrayBufferView): Uint8Array { return new Uint8Array(v.buffer, v.byteOffset, v.byteLength); }
