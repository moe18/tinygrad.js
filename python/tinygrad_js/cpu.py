"""The ``CPU`` device inside the browser.

Upstream tinygrad's ``CPU`` device compiles kernels to native code (clang / LLVM / its x86 assembler); none of those
exist in WebAssembly. Upstream code also refers to ``"CPU"`` directly in a few places (realizing device-less tensors,
loading state dicts from DISK, VALIDATE_WITH_CPU). To keep those paths working, this module (registered as
``tinygrad.runtime.ops_cpu`` by ``tinygrad_js.install``) provides ``CPU`` backed by tinygrad's own pure-Python
device implementation (``tinygrad.runtime.ops_python``). It really runs on the CPU, but interpreted: it is slow.
"""
from tinygrad.runtime.ops_python import PythonDevice

class CpuDevice(PythonDevice):
  pass
