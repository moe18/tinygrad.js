/// <reference types="vite/client" />
interface ImportMetaEnv { readonly VITE_PYODIDE_URL?: string }
declare module "*.py?raw" { const src: string; export default src; }
