# shallot-avbd-physics

an avbd (augmented vertex block descent) rigid-body solver for [shallot](https://github.com/dylanebert/shallot), on the gpu. an alternative to the built-in `PhysicsPlugin`: pick one per project.

```bash
bun add @dylanebert/shallot-avbd-physics
```

name it in `shallot.json` in place of `Physics`:

```json
{
    "scene": "scenes/main.scene",
    "plugins": {
        "Avbd": "@dylanebert/shallot-avbd-physics"
    }
}
```

or add `AvbdPlugin` to a plugin list in code. Read poses with `Avbd.readBody(state, eid)` and reach the raw solver through `Avbd.step(state)`. AVBD resources belong to that State; it needs WebGPU, so without a device stay on the built-in solver.

changing it: [`CONTRIBUTING.md`](CONTRIBUTING.md). mit.
