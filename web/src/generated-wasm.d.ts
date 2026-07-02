/**
 * Ambient fallback for the GENERATED wasm-bindgen module (web/src/wasm/ is
 * git-ignored; ./web/build-wasm.sh produces it). TypeScript resolves the real
 * module when it exists (local dev after a wasm build) and falls back to this
 * wildcard declaration on a clean checkout (CI typecheck without a wasm build).
 * Runtime already handles the missing module: loadWasmFactory() catches the
 * failed dynamic import and returns null.
 */
declare module '*/wasm/rvface_wasm.js';
