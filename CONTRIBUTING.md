# Contributing

For anyone changing the solver, person or agent. Each API's contract is the JSDoc beside it. This page holds what the tree, the scripts and a failing check do not say.

```bash
bun install --frozen-lockfile
bun run check
bun run test
bun test ./tests/oracle.oracle.ts
bun test ./tests/corpus.oracle.ts
bun test ./tests/differential.gpu.ts
bun test ./tests/headless.gpu.ts
```

- `src/` is the plugin, the GPU step, the narrowphase and hull packing; `src/core.ts` is the public `/core` export. Shallot is imported only through its public package exports.
- `*.test.ts` files are the cheap Bun tier; each test's name is its claim and its timeout is its budget. The `test` script sets the default timeout to 250 ms.
- `*.oracle.ts` files are the CPU f64 oracle tier. `*.gpu.ts` files require a real WebGPU device; run each named file by path, and a missing device is a failure, not a skip. `tests/preload.ts` registers Shallot's Bun plugin for tests that evaluate TGSL kernels.
- `tests/` holds the f64 CPU oracle, a port of the reference C++, with committed gold vectors. The corpus and closed-form tests are the evidence. A GPU-authored buffer is proved by a real-device probe; a missing GPU seat is inconclusive, never green. A rendered claim, if one is ever admitted, uses Shallot's public `captureFrame` contract.
- `tests/character-sweep.test.ts` and the character rows of `tests/headless.gpu.ts` gate the engine's shipped character. They stay here until they move.
- `peerDependencies` is the compatibility range. The dev dependency and `bun.lock` carry the published Shallot prerelease pin used for checks. Shallot's `shallot()` plugin owns the Vite- and Bun-side TypeGPU transform and deduplication; TGSL tests register it through `tests/preload.ts`. Package states, linking and exit: [Shallot's CONTRIBUTING](https://github.com/dylanebert/shallot/blob/main/CONTRIBUTING.md#dependencies-and-releases).
- A release is a `v<version>` tag matching `package.json`. The release workflow runs `check` and `test` before publishing.
- Retired examples live in Git tags, listed in `ARCHIVE.md`.
