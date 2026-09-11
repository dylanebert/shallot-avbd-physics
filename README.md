# shallot-avbd-physics

An AVBD (augmented vertex block descent) rigid-body solver for [Shallot](https://github.com/dylanebert/shallot) that runs on the GPU. It's an alternative to Shallot's built-in `PhysicsPlugin`, not an addition: pick one per project. It plugs in through the engine's published `physics/core` seam, so the same `Body`, `Spring` and `Joint` components drive either solver.

## Enabling it

```bash
bun add @dylanebert/shallot-avbd-physics
```

Name it in `shallot.json` in place of `Physics`:

```json
{
    "scene": "scenes/main.scene",
    "plugins": {
        "Avbd": "@dylanebert/shallot-avbd-physics"
    }
}
```

Or add `AvbdPlugin` to your plugin list in code. Read live poses with `Avbd.readBody`; `Avbd.step` is the raw escape hatch. It needs a WebGPU device, so a project without one should stay on the built-in solver.

## Layout

- `src/`: the plugin (`index.ts`), GPU step (`step.ts`), narrowphase (`collide.ts`) and hull packing
- `tests/`: the f64 CPU oracle, a port of the reference C++, with committed gold vectors
- `ARCHIVE.md`: retired examples, indexed by Git tag

## Developing

```bash
bun install --frozen-lockfile
bun run list                # installed Shallot carrier population
bun run workflow            # regenerate the hosted surface workflow
bun run check               # tsc + Biome + carrier declaration/drift checks
bun run test                # installed carrier unit sweep (src/)
bun run test:integration -- --base <parent> --diff <commit>
```

CPU oracle evidence is explicit by path, never part of the ordinary unit or integration sweeps:

```bash
bun test ./tests/math.oracle.ts ./tests/storage.oracle.ts ./tests/coloring.oracle.ts \
  ./tests/hull-pack.oracle.ts ./tests/reduce.oracle.ts ./tests/sat.oracle.ts \
  ./tests/hull.oracle.ts ./tests/oracle.oracle.ts ./tests/corpus.oracle.ts \
  ./tests/motor.oracle.ts ./tests/character.oracle.ts ./tests/character-sweep.oracle.ts \
  ./tests/rounded.oracle.ts
```

`tests/differential.oracle.ts` and `tests/headless.oracle.ts` are named GPU oracles and never enter the ordinary unit or integration sweeps. Run them directly by path to request that evidence. Their bodies need WebGPU, but the pinned carrier does not supply the `gpu` requirement, so the current direct commands refuse nonzero before the bodies run rather than passing or skipping. The committed `shallot.json` root manifest remains authoritative for the complete visible population.

### Against a local engine

`@dylanebert/shallot` is a peer dependency on the published range `^0.9.5`; the dev dependency is pinned to the exact Git commit `github:dylanebert/shallot#1ad4d7a500c13b4a3c0422a3b09584834d5e626d`. To test against an unreleased engine, link it and its `typegpu` too. The engine and the solver must share one typegpu instance, or struct schemas from one copy fail layout checks in the other:

```bash
# in your shallot checkout
bun link
cd node_modules/typegpu && bun link

# here
bun link @dylanebert/shallot
bun link typegpu
bun test src
```

Run `bun install --frozen-lockfile --force` here to replace the local links with the exact Shallot and TypeGPU dependencies recorded in `bun.lock`. This restores the Git-pinned Shallot carrier, not the published peer range. To remove the global link registrations too, run `bun unlink` in the Shallot checkout and in its `node_modules/typegpu` directory.

## Releasing

Bump `version` in `package.json`, commit, and push a matching `v<version>` tag. The release workflow runs the native `bun run check` and `bun run test` gates before publishing. Named oracle commands remain explicit evidence and are not part of the release gate.

## License

MIT
