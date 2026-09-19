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

or add `AvbdPlugin` to a plugin list in code. read poses with `Avbd.readBody`; `Avbd.step` is the raw step. it needs a webgpu device; without one, stay on the built-in solver.

changing it: [`CONTRIBUTING.md`](CONTRIBUTING.md). mit.
