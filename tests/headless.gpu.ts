// Headless GPU integration rows exercise Shallot's Character and Player plugins.
// Each row initializes bun-webgpu locally, so unit checks do not pay for GPU setup.

import { expect, test } from "bun:test";
import {
    build,
    Character,
    CharacterPlugin,
    InputPlugin,
    PhysicsPlugin,
    PlayerPlugin,
    RenderPlugin,
    readBody,
    type State,
} from "@dylanebert/shallot";

/** fixed ticks before probing the headless body pose. */
const TICKS = 5;

function probePhysicsPose(state: State, eid: number): NonNullable<ReturnType<typeof readBody>> {
    const pose = readBody(state, eid);
    if (!pose) throw new Error("physics body missing after build");
    return pose;
}

async function setupGpuPeer(): Promise<void> {
    const peerModule = "bun-webgpu";
    const peer = (await import(peerModule)) as { setupGlobals(): Promise<void> };
    await peer.setupGlobals();
}

/** every pose lane finite — the S1 bar: the build executed and the readback returned real values. */
function expectFinite(pose: {
    pos: readonly number[];
    quat: readonly number[];
    vel: readonly number[];
}): void {
    for (const lane of [...pose.pos, ...pose.quat, ...pose.vel]) {
        expect(Number.isFinite(lane), `expected a finite pose lane, got ${lane}`).toBe(true);
    }
}

test("gpu headless characterplugin sweeps headlessly", async () => {
    await setupGpuPeer();
    const app = await build({
        plugins: [PhysicsPlugin, CharacterPlugin],
        defaults: false,
        scene: `<scene>
                <a body="pos: 0 0 0; half-extents: 10 0.5 10; mass: 0" />
                <a
                    id="char"
                    body="pos: 0 3 0; shape: 2; half-extents: 0 0.6 0 0.3; mass: 0"
                    character
                />
            </scene>`,
    });
    // CharacterPlugin requires PhysicsPlugin and sweeps the authored capsule before the solve.
    const chars = [...app.state.query([Character])];
    expect(chars.length).toBe(1);
    for (let i = 0; i < TICKS; i++) app.state.step();
    const pose = probePhysicsPose(app.state, chars[0]);
    expectFinite(pose);
    // Five fixed ticks of world gravity move the un-driven character below its authored height.
    expect(pose.pos[1]).toBeLessThan(3);
    expect(pose.pos[1]).toBeGreaterThan(2.9);
    app.dispose();
}, 20_000);

test("gpu headless playerplugin composes headlessly", async () => {
    await setupGpuPeer();
    const app = await build({
        plugins: [RenderPlugin, InputPlugin, PhysicsPlugin, CharacterPlugin, PlayerPlugin],
        defaults: false,
        scene: `<scene>
                <a id="eye" camera transform="pos: 0 1.5 5" />
                <a
                    id="player"
                    body="pos: 0 1 0; shape: 2; half-extents: 0 0.6 0 0.3; mass: 0"
                    character
                    player="camera: @eye"
                />
                <a body="pos: 0 0 0; half-extents: 10 0.5 10; mass: 0" />
            </scene>`,
    });
    const chars = [...app.state.query([Character])];
    expect(chars.length).toBe(1);
    for (let i = 0; i < TICKS; i++) app.state.step();
    const pose = probePhysicsPose(app.state, chars[0]);
    expectFinite(pose);
    // The sweep depenetrates the capsule from its authored overlap to the geometry-derived rest height.
    expect(pose.pos[1]).toBeGreaterThan(1.35);
    expect(pose.pos[1]).toBeLessThan(1.45);
    app.dispose();
}, 20_000);
