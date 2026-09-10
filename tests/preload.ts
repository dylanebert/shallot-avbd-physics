import { plugin } from "bun";
import typegpu from "unplugin-typegpu/bun";

// TGSL function bodies are transpiled at load time: without this transform every kernel resolves
// with no metadata and CPU-called kernels return NaN. Exactly one instance may run, since a second
// pass re-wraps the emitted metadata. `.ts` only, matching the engine's own test preload.
plugin(typegpu({ include: /\.tsx?$/ }));
