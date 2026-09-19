# Contributing

For anyone changing the solver, person or agent. Each API's contract is the JSDoc beside it. This page holds what the tree, the scripts and a failing check do not say.

```bash
bun install --frozen-lockfile
bun run check
bun run test
bun run test -- --integration --base <ref> --diff <ref>   # tests whose subject changed
bun run test -- --oracle "<claim>"   # one named oracle, never part of a sweep
bun run list
bun run workflow
```

- `src/` is the plugin, the GPU step, the narrowphase and hull packing; `src/core.ts` is the public `/core` export. Shallot is imported only through its public package exports.
- `tests/` holds the f64 CPU oracle, a port of the reference C++, with committed gold vectors. The corpus and closed-form tests are the evidence; a test that steps the solver is an integration test with a 20000 ms budget, and construction or closed-form tests are unit tests. A GPU-authored buffer is proved by a real-device probe; a missing GPU seat is inconclusive, never green. A rendered claim, if one is ever admitted, uses Shallot's public `captureFrame` contract.
- `tests/character-sweep.oracle.ts` and the character rows of `tests/headless.oracle.ts` gate the engine's shipped character. They stay here until they move.
- `peerDependencies` is the compatibility range. The dev dependency and `bun.lock` carry a full-SHA Git pin of Shallot until a stable release replaces it. Never link a second `typegpu`; a Vite consumer dedupes `@dylanebert/shallot` and `typegpu`. Package states, linking and exit: [Shallot's CONTRIBUTING](https://github.com/dylanebert/shallot/blob/main/CONTRIBUTING.md#pins-and-dependencies).
- A release is a `v<version>` tag matching `package.json`. The release workflow runs `check` and `test` before publishing; oracles are not part of the gate.
- Retired examples live in Git tags, listed in `ARCHIVE.md`.
