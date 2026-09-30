import { expect, setDefaultTimeout, test } from "bun:test";
import { build, type Plugin } from "@dylanebert/shallot";

setDefaultTimeout(1000);

async function bounded<T>(label: string, promise: PromiseLike<T>): Promise<T> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
        return await Promise.race([
            promise,
            new Promise<never>((_, reject) => {
                timer = setTimeout(() => reject(new Error(`${label} exceeded 750 ms`)), 750);
            }),
        ]);
    } finally {
        clearTimeout(timer);
    }
}

// The Slab-backed plugin was removed in engine-gpu-core stage 4. Its rebuild owns this claim;
// it must keep color/body dispatch counts on the GPU rather than restoring colorMirror views.
test.todo("avbd-rebuild: fixed count scheduling constructs no Uint32Array per tick and maps no GPU bytes", async () => {
    const peer = "bun-webgpu";
    await (await import(peer)).setupGlobals();
    const plugin = Reflect.get(await import("../src/index"), "AvbdPlugin") as Plugin | undefined;
    if (!plugin)
        throw new Error("AVBD rebuild must export AvbdPlugin to prove its fixed count scheduling");
    const adapter = await bounded("AVBD rebuild adapter", navigator.gpu.requestAdapter());
    if (!adapter) throw new Error("AVBD rebuild adapter unavailable");
    const device = await bounded("AVBD rebuild device", adapter.requestDevice());
    let maps = 0;
    let validation: GPUError | undefined;
    device.addEventListener("uncapturederror", (event) => {
        validation ??= event.error;
    });
    const create = device.createBuffer.bind(device);
    device.createBuffer = (descriptor) => {
        const buffer = create(descriptor);
        const map = buffer.mapAsync.bind(buffer);
        buffer.mapAsync = (...args) => {
            maps++;
            return map(...args);
        };
        return buffer;
    };
    const Original = Uint32Array;
    let views = 0;
    let app: Awaited<ReturnType<typeof build>> | undefined;
    try {
        app = await build({
            defaults: false,
            plugins: [plugin],
            device,
            scene: `<scene><a body="pos: 0 0 0; half-extents: 10 0.5 10; mass: 0" /><a body="pos: 0 2 0; half-extents: 0.5 0.5 0.5; mass: 1" /></scene>`,
        });
        globalThis.Uint32Array = new Proxy(Original, {
            construct(target, args, newTarget) {
                views++;
                return Reflect.construct(target, args, newTarget);
            },
        });
        new Uint32Array(1);
        expect(views).toBe(1);
        views = 0;
        maps = 0;
        for (let i = 0; i < 5; i++) app.state.step(1 / 60);
        await bounded("AVBD fixed count submissions", device.queue.onSubmittedWorkDone());
        if (validation) throw validation;
        expect(views).toBe(0);
        expect(maps).toBe(0);
    } finally {
        globalThis.Uint32Array = Original;
        app?.dispose();
        device.destroy();
    }
});
