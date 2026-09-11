// Named headless WebGPU build and lifecycle evidence for AVBD, Character, and Player. When a runner
// supplies a Lavapipe-backed WebGPU environment, these bodies compile the stack, step it, and read
// solver-owned poses through Shallot's public `probeBuffer` helper. They check finite readback,
// closed-form falling direction, and the grounded character rest band. They are not solver-parity or
// conformant-device evidence; Lavapipe is testing-only, and real-device parity needs separate evidence.
//
// `CAPACITY` is 8192 because the contact store uses about 3584 bytes per entity. Its roughly 28 MiB
// allocation fits under a 128 MiB storage-binding limit, while the engine default capacity of 65536
// would require roughly 235 MiB. The public `probeBuffer` helper owns command submission, staging,
// mapping, usage validation, and alignment. `Avbd.step.bodies` is the solver's SoA buffer, and the
// imported B_* indices keep probe offsets aligned with the solver layout.
//
// Request this oracle from the AVBD repository root with
// `bun test ./tests/headless.oracle.ts`. The current AVBD preloads install the TypeGPU transform and
// Shallot declaration carrier only; they do not install a software adapter. Each declaration requires
// `gpu`, and the pinned carrier has no GPU provider, so the command currently refuses nonzero before
// the bodies run rather than passing or skipping. Ordinary unit and integration sweeps exclude
// `.oracle.ts` files.

import { expect } from "bun:test";
import {
    Body,
    build,
    Character,
    CharacterPlugin,
    Compute,
    InputPlugin,
    MirrorPlugin,
    PlayerPlugin,
    probeBuffer,
    RenderPlugin,
    SlabPlugin,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { check } from "@dylanebert/shallot/harness/check";
import { Avbd, AvbdPlugin } from "../src/index";
import { B_POS, B_QUAT, B_VELL } from "../src/step";

/** the headless entity capacity — the contact store fits under lavapipe's 128 MiB binding ceiling. */
const CAPACITY = 8192;
/** fixed ticks to step before probing — five ticks of closed-form fall, no contact yet. */
const TICKS = 5;

/** the solver-owned pose of one body, read through the existing `probeBuffer` seam. */
async function probePose(eid: number): Promise<{ pos: number[]; quat: number[]; vel: number[] }> {
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

/** every pose lane finite — the S1 bar: the build executed and the readback returned real values. */
function expectFinite(pose: { pos: number[]; quat: number[]; vel: number[] }): void {
    for (const lane of [...pose.pos, ...pose.quat, ...pose.vel]) {
        expect(Number.isFinite(lane), `expected a finite pose lane, got ${lane}`).toBe(true);
    }
}

//  headless lavapipe physics (build + step + probeBuffer readback)
check(
    "AvbdPlugin builds, steps, and probes finite body poses at capacity 8192",
    {
        claim: "gpu headless avbdplugin builds, steps, and probes finite body poses at capacity 8192",
        size: "integration",
        requires: ["gpu"],
    },
    async () => {
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
        const pose = await probePose(box!);
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
    },
);

check(
    "CharacterPlugin sweeps headlessly at the same capacity",
    {
        claim: "gpu headless characterplugin sweeps headlessly at the same capacity",
        size: "integration",
        requires: ["gpu"],
    },
    async () => {
        const app = await build({
            plugins: [SlabPlugin, MirrorPlugin, AvbdPlugin, CharacterPlugin],
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
        // the kinematic character: the sweep (fixed group, before the solve) reads candidate poses
        // through the body Mirror and writes its pose via setKinematic — the whole readback-bounded
        // coupling, headless.
        const chars = [...app.state.query([Character])];
        expect(chars.length).toBe(1);
        for (let i = 0; i < TICKS; i++) app.state.step();
        const pose = await probePose(chars[0]);
        expectFinite(pose);
        // the sweep applies the world gravity (−10, the plugin's configured GRAVITY) to an un-driven
        // character, so it too falls from its authored 3 m — derived from the sweep contract, not
        // tuned. The floor is the same derivation's other side, on the discrete scheme the sweep
        // actually runs: symplectic Euler (velocity first, then position — the runtime twin of the
        // oracle's moveCharacter), whose closed form the oracle pins as x_n = x0 + g·h²·n(n+1)/2 per
        // integrated tick (tests/oracle.oracle.ts). The seeding precondition is what fixes n,
        // differently than the box: the sweep's velocity is persistent CPU-side state and integrates
        // from the first tick, but the first tick's draw-group pack seed overwrites the slot with the
        // authored pose (the readback shows 3 at tick 1; the sweep's own trajectory has already moved
        // once), so TICKS = 5 ticks of un-driven fall cover Δy = g·h²·n(n+1)/2 at g = 10, h = 1/60,
        // n = TICKS = 5 ≈ 0.0417 m (the capsule's bottom at 2.1 sits 1.6 m above the ground top at
        // 0.5, so no depenetration or contact can have moved it), so the pose must still exceed
        // 3 − 0.0417 ≈ 2.958 — assert > 2.9, deliberately below the derived bound so the arm cannot
        // flake on the derivation's own precision (a floor is never tightened to the derived value,
        // never fit to an observed reading), while a zero readback still reds.
        expect(pose.pos[1]).toBeLessThan(3);
        expect(pose.pos[1]).toBeGreaterThan(2.9);
        app.dispose();
    },
);

check(
    "PlayerPlugin composes headlessly at the same capacity",
    {
        claim: "gpu headless playerplugin composes headlessly at the same capacity",
        size: "integration",
        requires: ["gpu"],
    },
    async () => {
        const app = await build({
            plugins: [
                SlabPlugin,
                TransformsPlugin,
                RenderPlugin,
                InputPlugin,
                MirrorPlugin,
                AvbdPlugin,
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
        const pose = await probePose(chars[0]);
        expectFinite(pose);
        // the player is an un-driven character grounded at its closed-form rest height: authored at
        // y = 1 the capsule's bottom (0.9 below center) embeds in the ground (top at 0.5), and the
        // sweep depenetrates to rest at center y = 0.5 + 0.6 + 0.3 = 1.4 — derived from the authored
        // geometry, never from the observed value; the 0.05 band absorbs the depenetration residual.
        expect(pose.pos[1]).toBeGreaterThan(1.35);
        expect(pose.pos[1]).toBeLessThan(1.45);
        app.dispose();
    },
);
