// These GPU integration rows exercise the AVBD solver and Shallot's Character and Player plugins.
// AVBD poses come from its public solver buffer; Character and Player poses come from PhysicsPlugin.
//
// `CAPACITY` keeps AVBD's contact store within the adapter's storage-binding limit.
//
// Each row initializes bun-webgpu locally, so unit checks do not pay for GPU setup.

import { expect, test } from "bun:test";
import {
    Body,
    build,
    Character,
    CharacterPlugin,
    Compute,
    InputPlugin,
    MirrorPlugin,
    PhysicsPlugin,
    PlayerPlugin,
    probeBuffer,
    RenderPlugin,
    readBody,
    SlabPlugin,
    type State,
    TransformsPlugin,
} from "@dylanebert/shallot";

import { Avbd, AvbdPlugin } from "../src/index";
import { B_POS, B_QUAT, B_VELL } from "../src/step";

/** the headless entity capacity — the AVBD contact store fits under the adapter's storage limit. */
const CAPACITY = 8192;
/** fixed ticks before probing the headless body pose. */
const TICKS = 5;

/** the AVBD solver-owned pose of one body, read through Shallot's public `probeBuffer` seam. */
async function probeAvbdPose(
    eid: number,
): Promise<{ pos: number[]; quat: number[]; vel: number[] }> {
    const step = Avbd.step;
    const device = Compute.device;
    if (!step || !device) throw new Error("physics step or device missing after build");
    // the 32-byte read below assumes the quat column directly follows the pos column — guard the
    // contiguity instead of assuming it silently.
    if (B_QUAT !== B_POS + 1) throw new Error("pos/quat columns are no longer contiguous");
    // SoA cols-buffer: body eid's column c lives at `c * eidCap + eid`, one vec4 (16 B) per column.
    // pos (B_POS) + quat (B_QUAT) are contiguous; the linear velocity (B_VELL) is its own probe.
    const at = (col: number) => (col * step.eidCap + eid) * 16;
    const [pose, vel] = await Promise.all([
        probeBuffer(device, step.bodies, {
            offset: at(B_POS),
            size: 32,
            label: "headless-body-probe",
        }),
        probeBuffer(device, step.bodies, {
            offset: at(B_VELL),
            size: 16,
            label: "headless-vel-probe",
        }),
    ]);
    const f = new Float32Array(pose.bytes);
    const v = new Float32Array(vel.bytes);
    return {
        pos: [f[0], f[1], f[2]],
        quat: [f[4], f[5], f[6], f[7]],
        vel: [v[0], v[1], v[2]],
    };
}

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

// AVBD build, step, and public buffer readback
test("gpu headless avbdplugin builds, steps, and probes finite body poses at capacity 8192", async () => {
    await setupGpuPeer();
    const app = await build({
        plugins: [SlabPlugin, MirrorPlugin, AvbdPlugin],
        defaults: false,
        capacity: CAPACITY,
        scene: `<scene>
                <a body="pos: 0 0 0; half-extents: 10 0.5 10; mass: 0" />
                <a body="pos: 0 6 0; half-extents: 0.6 0.6 0.6; mass: 1" />
            </scene>`,
    });
    expect(Avbd.step).not.toBeNull();
    expect(Avbd.step!.eidCap).toBe(CAPACITY);
    // the falling box is the scene's only mass > 0 body; the ground is static.
    const box = [...app.state.query([Body])].find((eid) => Body.mass.get(eid) > 0);
    expect(box).toBeDefined();
    // one state.step() = one fixed tick (dt defaults to Time.FIXED_DT); the first tick's draw-group
    // pack seeds the box's slot, later ticks integrate it — and 4.9 m above contact is far outside
    // the 0.04 speculative band, so every tick is closed-form free fall.
    for (let i = 0; i < TICKS; i++) app.state.step();
    const pose = await probeAvbdPose(box!);
    expectFinite(pose);
    // closed-form gravity direction: after solved free-fall ticks the box sits strictly below its
    // authored start — a band derived from free-fall kinematics, never from the observed value.
    // The floor is the same derivation's other side, on the discrete scheme the solver actually
    // runs: symplectic Euler (velocity first, then position), whose closed form the oracle pins as
    // x_n = x0 + g·h²·n(n+1)/2 per integrated tick (tests/oracle.oracle.ts). The seeding
    // precondition is what fixes n: the first tick's draw-group pack seeds the box's slot (authored
    // pose, velocity zeroed) and lands no observable integration, so TICKS − 1 = 4 ticks integrate
    // — five ticks of free fall cover Δy = g·h²·n(n+1)/2 at g = 10, h = 1/60, n = 4 ≈ 0.0278 m
    // (the box starts 4.9 m above contact, far outside the 0.04 speculative band, so no contact
    // can have accelerated it), so the pose must still exceed 6 − 0.0278 ≈ 5.972 — assert > 5.9,
    // deliberately below the derived bound so the arm cannot flake on the derivation's own
    // precision (a floor is never tightened to the derived value, never fit to an observed
    // reading), while a zero readback (wrong eid or a broken readback seam) still reds.
    expect(pose.pos[1]).toBeLessThan(6);
    expect(pose.pos[1]).toBeGreaterThan(5.9);
    app.dispose();
}, 20_000);

test("gpu headless characterplugin sweeps headlessly at the same capacity", async () => {
    await setupGpuPeer();
    const app = await build({
        plugins: [SlabPlugin, MirrorPlugin, PhysicsPlugin, CharacterPlugin],
        defaults: false,
        capacity: CAPACITY,
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

test("gpu headless playerplugin composes headlessly at the same capacity", async () => {
    await setupGpuPeer();
    const app = await build({
        plugins: [
            SlabPlugin,
            TransformsPlugin,
            RenderPlugin,
            InputPlugin,
            MirrorPlugin,
            PhysicsPlugin,
            CharacterPlugin,
            PlayerPlugin,
        ],
        defaults: false,
        capacity: CAPACITY,
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
