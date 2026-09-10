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
- `examples/collapse`, `examples/sandbox`: showcase projects built on the solver

## Developing

```bash
bun install
bun run check               # tsc + Biome
bun test src                # kernel structure and CPU-callable math
bun test ./tests/*.oracle.ts
```

The `*.tier.ts` files and the examples' `gate` scripts need a real GPU.

### Against a local engine

`@dylanebert/shallot` is a peer and a dev dependency pinned to a published range. To test against an unreleased engine, link it and its `typegpu` too. The engine and the solver must share one typegpu instance, or struct schemas from one copy fail layout checks in the other:

```bash
# in your shallot checkout
bun link
cd node_modules/typegpu && bun link

# here
bun link @dylanebert/shallot
bun link typegpu
bun test src
```

Run `bun install` to go back to the published engine, and `bun unlink` in both engine locations to drop the registrations.

## Releasing

Bump `version` in `package.json`, commit, and push a matching `v<version>` tag. The release workflow checks, runs the oracle tests and publishes with the `NPM_TOKEN` repository secret.

## License

MIT
