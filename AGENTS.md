# AVBD consumer contract

This repository is a published Shallot extension. Its stable compatibility contract stays in
`peerDependencies`; its own gates run against the qualified source-stage candidate
`github:dylanebert/shallot#0664218f465224397b80aeb604b51178ac71cfb2` in `devDependencies` and
`bun.lock`. Do not persist a `file:`, `link:`, short or moving Git selector, mutable dist-tag, or
local tarball. The candidate identity is the complete 40-hex SHA, not Bun's abbreviated display.

The supported states are:

- local co-development: record both HEADs, dirt and manifest/lock hashes; run `bun link` only in
the Shallot producer and `bun link @dylanebert/shallot --no-save` here; never link a second
`typegpu` copy. Vite consumers dedupe `@dylanebert/shallot` and `typegpu`.
- source staging: the committed full-SHA Git identity above, installed with a fresh empty cache and
`bun install --frozen-lockfile`.
- published exit: a future stable range and exact lock only after that release exists.

The normal entry is the installed carrier: `bun run list`, `bun run check`, `bun run test`, and
`bun run workflow`; integration selectors must be non-empty. These carrier gates do not replace
TypeScript, Biome, package-preflight, export, or product/oracle gates. CPU corpus and transforms
remain explicit unit/oracle evidence. GPU-authored buffers require a real-device probe; browser
presentation, if admitted later, uses Shallot's public `captureFrame` contract. Missing seats
are inconclusive.

Identity proof checks `package.json`, `bun.lock`, installed metadata, and the installed realpath.
A clean exit force-installs frozen from a newly empty cache, proves the installed package is not the
producer and leaves no symlink or local-directory residue, then reruns the focused gate. Package
preflight is a temporary artifact only, never a persisted stage. The public extension exports are
`@dylanebert/shallot-avbd-physics` and `/core`; Shallot imports use its public package exports only.

`tests/character-sweep.oracle.ts` and the character rows of `tests/headless.oracle.ts`
(`characterplugin` and `playerplugin`) gate the engine's shipped character. They stay owned here
until they move.
`tests/corpus.oracle.ts` and the closed-form rows are what survives.
Oracle rows that step the solver are `integration` with a 20000 ms budget, and construction or
closed-form rows are `unit`.
