# shallot-avbd-physics

An AVBD (augmented vertex block descent) rigid-body solver core for [Shallot](https://github.com/dylanebert/shallot). The package exposes the low-level GPU step pipeline, collision kernels and hull packing for custom tools and the gym scenario. It no longer ships the `AvbdPlugin`: that adapter read per-field ECS buffers through Shallot's retired Slab API and is removed rather than rebuilt against a replacement table API.

```bash
bun add @dylanebert/shallot-avbd-physics
```

Import `PhysicsStep` and the supporting layouts from `@dylanebert/shallot-avbd-physics/core`. For a game plugin, use Shallot's built-in `PhysicsPlugin` until a table-native AVBD consumer is authored.

Changing it: [`CONTRIBUTING.md`](CONTRIBUTING.md). MIT.
