# Contributing

For anyone changing the solver, person or agent. Each API's contract is the JSDoc beside it. This page holds what the tree, the scripts and a failing check do not say.

```bash
bun install --frozen-lockfile
bun run check
bun run test
bun run test -- --integration --base <ref> --diff <ref>   # tests whose subject changed
bun run list
bun run workflow
```

- `src/` is the plugin, the GPU step, the narrowphase and hull packing; `src/core.ts` is the public `/core` export. Shallot is imported only through its public package exports.
- `tests/` holds the f64 CPU oracle, a port of the reference C++, with committed gold vectors. The corpus and closed-form tests are the evidence, and `bun run test` runs them. A CPU test under the unit budget is a unit test; one over it in the full sweep is an integration test whose subject names the solver files it steps. A GPU-authored buffer is proved by a real-device probe; a missing GPU seat is inconclusive, never green. A rendered claim, if one is ever admitted, uses Shallot's public `captureFrame` contract.
- `tests/character-sweep.test.ts` and the character rows of `tests/headless.test.ts` gate the engine's shipped character. They stay here until they move.
- `peerDependencies` is the compatibility range. The dev dependency and `bun.lock` carry a full-SHA Git pin of Shallot until a stable release replaces it. Never link a second `typegpu`; a Vite consumer dedupes `@dylanebert/shallot` and `typegpu`. Package states, linking and exit: [Shallot's CONTRIBUTING](https://github.com/dylanebert/shallot/blob/main/CONTRIBUTING.md#pins-and-dependencies).
- A release is a `v<version>` tag matching `package.json`. The release workflow runs `check` and `test` before publishing.
- Retired examples live in Git tags, listed in `ARCHIVE.md`.
