# shallot-avbd-physics

An AVBD (augmented vertex block descent) rigid-body solver for [Shallot](https://github.com/dylanebert/shallot) that runs on the GPU. It is an alternative to Shallot's built-in `PhysicsPlugin`: pick one per project. The extension consumes Shallot's public physics API and exports its own public `./core` surface.

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

Or add `AvbdPlugin` to a plugin list in code. Read live poses with `Avbd.readBody`; `Avbd.step` is the raw escape hatch. It needs a WebGPU device, so a project without one should stay on the built-in solver.

## Layout and evidence

- `src/`: the plugin (`index.ts`), GPU step (`step.ts`), narrowphase (`collide.ts`) and hull packing
- `src/core.ts`: the public extension core export
- `tests/`: the f64 CPU oracle, a port of the reference C++, with committed gold vectors
- `ARCHIVE.md`: retired examples, indexed by Git tag

The CPU corpus and transforms are unit/oracle evidence; the migration does not manufacture a
population. GPU-authored buffers are named real-device probes and browser pixels have no admitted
AVBD population yet. If a future consumer makes a rendered claim, it must use Shallot's public
`captureFrame` contract rather than a local screenshot transport or CPU reconstruction.

## Installed carrier gates

The installed Shallot carrier owns the project population:

```bash
bun run list
bun run check
bun run test
bun run workflow
bun run test -- --integration --base <parent> --diff <commit>
```

The changed-subject selector must match at least one integration row. Named CPU/GPU evidence stays
explicit and is requested through the carrier's oracle selector when its declared premise exists;
missing GPU seats are inconclusive, never green.

## Package states

The stable extension compatibility range remains in `peerDependencies`. The committed development
identity is the qualified source-stage candidate
`github:dylanebert/shallot#70770cfc34d82fdd19cb705d8753bb6f093748d6` in both `package.json` and
`bun.lock`. A future stable release may replace it with an intentional stable range and fresh lock.

For local co-development, require Bun 1.4.2, record both repositories' HEAD/dirt and manifest/lock
hashes, run `bun link` only in the Shallot producer, and run
`bun link @dylanebert/shallot --no-save` here. Do not link a second `typegpu`; Vite consumers
configure `resolve.dedupe` for `@dylanebert/shallot` and `typegpu`. Exit with a fresh-cache
`bun install --force --frozen-lockfile`, prove the installed realpath is no longer the producer,
then rerun the focused gate. The manifest and lock must remain byte-identical.

A local `bun pm pack` of the extension is temporary package preflight, not a remote package state.
Inspect its source, integrity and installed metadata, prove the `/core` export and installed
`shallot` bin, then restore the candidate Git stage with the frozen install. Never persist a
`file:`/`link:` directory, short or moving Git ref, mutable dist-tag, or evidence-free tarball.

## Releasing

Bump `version` in `package.json`, commit, and push a matching `v<version>` tag. The release workflow
runs the native `bun run check` and `bun run test` gates before publishing. Named oracle commands
remain explicit evidence and are not part of the release gate.

## License

MIT
