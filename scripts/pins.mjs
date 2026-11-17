// Pinned upstream versions. Update together with docs/COMPATIBILITY.md and re-run the test suite.
export const PINS = {
  tinygrad: [
    {
      version: "0.14.0",
      file: "tinygrad-0.14.0-py3-none-any.whl",
      url: "https://files.pythonhosted.org/packages/py3/t/tinygrad/tinygrad-0.14.0-py3-none-any.whl",
      sha256: "d2a5bea33c637fdf105963632329876b3d3253d38295ac1ffd2ff81646d93be3",
    },
    {
      version: "0.12.0",
      file: "tinygrad-0.12.0-py3-none-any.whl",
      url: "https://files.pythonhosted.org/packages/py3/t/tinygrad/tinygrad-0.12.0-py3-none-any.whl",
      sha256: "275c01961e0959579ba99c6ce1ed70a2c2f1a408ed8cfb5c4de8ad1355e66611",
    },
  ],
  pyodide: {
    version: "314.0.8",
    url: "https://github.com/pyodide/pyodide/releases/download/314.0.8/pyodide-314.0.8.tar.bz2",
    sha256: "ce4ac9b450c22b1cee278ddda7c12ee5b83acb1f967d0675fff189be6739e574",
    coreFiles: ["pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "package.json", "pyodide.d.ts", "ffi.d.ts"],
  },
};
