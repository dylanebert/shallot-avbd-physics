import {
    Body,
    Compute,
    Joint,
    type Mirror,
    MirrorPlugin,
    mirror,
    type Plugin,
    SlabPlugin,
    Spring,
    type State,
    type System,
    Time,
} from "@dylanebert/shallot";
import { BVH_FEATURES } from "@dylanebert/shallot/bvh";
import {
    type BodyState,
    bodyTraits,
    Hulls,
    jointDefs,
    jointSignature,
    jointTraits,
    springDefs,
    springSignature,
    springTraits,
} from "@dylanebert/shallot/physics";
import { BeginFrameSystem, Render } from "@dylanebert/shallot/rendering";
import { PrepassSystem } from "@dylanebert/shallot/standard/rendering";
import { packHulls } from "./hull";
import { diffStamps } from "./recycle";
import { B_POS, B_QUAT, B_VELL, type Inputs, PENALTY_MIN, PhysicsStep } from "./step";

// AVBD physics — the rigid-body solver re-added to the lean engine, validated against the f64 oracle
// (tests/avbd). It registers the shared physics components through `physics/core` and runs its own
// constraint/step/compose systems; physics is unaware of it. The solver itself is `step.ts` (`PhysicsStep`), the SAT is `collide.ts`. Runs the full `warmstart`
// augmented-Lagrangian layer (λ accumulation + the conditional penalty ramp + friction + cross-frame
// persistence: the collide merges last frame's λ/k by feature key, γ decay).
//
// Storage is eid-indexed over `capacity`, persistent across frames — a body's solver state lives at its
// eid slot and survives spawn/despawn. The whole CPU side is firehose: NO per-entity iteration. Each
// frame a single GPU `pack` pass (PackSystem, draw group) scans capacity gated on the Body membership
// bit and (a) compacts the live eids into the dense→eid map (`eids[0]` = count, `eids[1+d]` = eid) the
// solver passes read, and (b) one-time-seeds any newly-spawned body's slot from its authored slabs
// (gated on a GPU `seeded` flag — existing bodies untouched). The fixed-group AvbdStepSystem
// solves from last frame's pack output (a 1-frame structural latency — a new body joins the solve next
// frame).
//
// Body / Transform contract (roadmap): `Body.excludes [Transform]` (substrate trait). `Body` carries
// pos/quat (spawn pose, then physics-owned) + mass/halfExtents/friction on slab; this backend owns the
// GPU pose after spawn. `AvbdComposeSystem` scatters the live pose into the `transforms`
// firehose (`compose` below), after the Transform compose and before the renderer reads geometry, so a
// `Body`+`Part` renders at the physics-owned pose. A CPU consumer reads the live pose through
// `Avbd.readBody(state, eid)` (pass it to the physics pick layer) or the raw `Avbd.step(state)?.bodies` escape hatch.

const GRAVITY = -10;
const ALPHA = 0.99;
// penalty ramp rate (Eq. 17) + warmstart decay (Eq. 19), the canonical AVBD set. The warmstart layer
// carries λ/k across frames, so the ramp converges from the persisted state — the canonical 1e4 holds
// a resting box to ~mg/k.
const BETA_LIN = 1e4;
// the joint angular penalty-ramp rate (Phase 6.2) — the canonical AVBD value; contacts
// ignore it, so it only matters once a scene authors joints (via Avbd.step(state)?.setJoints).
const BETA_ANG = 100;
const GAMMA = 0.999;
// solve iterations — a perf/robustness tradeoff knob, NOT a correctness gate: every fixed count explodes
// on *some* taller stack (iters=4 churns a 16384-body 12-layer pile, iters=10 a taller one), so the choice
// is a deliberate ship tradeoff, not a bug. Production ships 6: it settles a 10-storey stack (the collapse
// showcase wall) where 4 (the paper's count) under-converges and pancakes during the settle, ~1ms@1k on
// lovelace (0.996ms measured, vs 0.575 at 4). The f64 oracle + GPU gates VALIDATE at iters=10
// (corpus.oracle.ts, the gym seeded gates), where the math is proven correct independent of this ship
// value. Raise per-scene through that State's `Avbd.step(state)?.configure` for a known-harder pile. physics.md "f32 precision"
// / "iters is a free knob".
const ITERATIONS = 6;
// The eid-space and dispatch bound is the owning State.capacity at warm (slot 0 is the never-minted
// sentinel); each body owns a fixed pair block in the contact store.
// dispatched-color cap (Phase 4): the primal dispatches at most this many colors per iteration, so the
// dispatch count is bounded by the cap not the body count (avbd.md "Dispatch count"). The reference +
// webphysics both cap at 8; the convergence probe found realistic piles color in ≤4, well under it.
const MAX_COLORS = 8;

interface AvbdWorld {
    step: PhysicsStep | null;
    colorMirror: Mirror | null;
    bodyMirror: Mirror | null;
    cachedBuf: ArrayBuffer | null;
    cachedView: Float32Array | null;
    stamps: Map<number, number>;
    lastHullCount: number;
    springSig: number;
    jointSig: number;
}

const avbdWorldKey = Symbol("shallot-avbd.world");

function createAvbdWorld(): AvbdWorld {
    return {
        step: null,
        colorMirror: null,
        bodyMirror: null,
        cachedBuf: null,
        cachedView: null,
        stamps: new Map(),
        lastHullCount: -1,
        springSig: 0,
        jointSig: 0,
    };
}

function avbdWorld(state: State): AvbdWorld {
    return state.resource(avbdWorldKey, createAvbdWorld);
}

/** the AVBD state owned by one State. The raw step handle, readback and pose cache are never shared across Apps. */
export const Avbd = {
    step(state: State): PhysicsStep | null {
        return avbdWorld(state).step;
    },
    readBody(state: State, eid: number): BodyState | null {
        return readBody(state, eid);
    },
    setKinematic(
        state: State,
        eid: number,
        pos: readonly [number, number, number],
        quat: readonly [number, number, number, number],
        teleport?: boolean,
        vel?: readonly [number, number, number],
    ): void {
        avbdWorld(state).step?.setKinematic(eid, pos, quat, teleport, vel);
    },
    setVelocity(state: State, eid: number, vx: number, vy: number, vz: number): void {
        avbdWorld(state).step?.setVelocity(eid, vx, vy, vz);
    },
    gravity: GRAVITY,
    dt: Time.FIXED_DT,
};

// the frame-stale color-count readback, create-stamp diff and hull-upload stamp live in each AvbdWorld.

// the membership + authored slab sources the GPU pack gathers from — all stable, fixed-capacity
// buffers: the Body slab `.gpu` (allocated at SlabPlugin.warm) and the `membership` mirror (the
// draw-group `first` MembershipSystem). A draw-group consumer runs after both, so they're always up at
// the call site; a missing one is a wiring bug (SlabPlugin not a dependency), not a frame to skip.
function inputs(): Inputs {
    const membership = Compute.buffers.get("membership");
    const pos = Body.pos.gpu;
    const quat = Body.quat.gpu;
    const half = Body.halfExtents.gpu;
    const mass = Body.mass.gpu;
    const friction = Body.friction.gpu;
    const shape = Body.shape.gpu;
    if (!membership || !pos || !quat || !half || !mass || !friction || !shape) {
        throw new Error("[avbd] pack sources missing — declare SlabPlugin as a dependency");
    }
    return { membership, pos, quat, half, mass, friction, shape };
}

// the GPU firehose pack: scan capacity gated on Body membership → the dense→eid map + the one-time seed
// of any newly-spawned body. Draw group, so it runs after SlabSystem + MembershipSystem (both `first`
// in draw) have flushed the Body slabs + the membership mirror — the pack reads fresh GPU data. Submits
// its own encoder (works headless, no renderer needed); its writes are visible to the next fixed step +
// to `compose` (below, later submits). No CPU per-entity iteration — the membership scan + seed are GPU.
const PackSystem: System = {
    name: "pack",
    group: "draw",
    update(state: State) {
        const world = avbdWorld(state);
        const step = world.step;
        if (!step || !Compute.device) return;
        // the CPU pre-pack stamp diff: force a re-seed on every eid recycled to a new body since the last
        // pack (a same-update destroy+create the GPU pack's non-member seed reset can't see, diffStamps).
        for (const eid of diffStamps(state.query([Body]), (e) => state.stamp(e), world.stamps)) {
            step.reseed(eid);
        }
        // upload the convex-hull geometry the collide pass reads (ShapeKind.Hull bodies) when the registry
        // changed — before the pack seeds a hull body's slot, so its first solve frame reads valid geometry.
        if (Hulls.size !== world.lastHullCount) {
            world.lastHullCount = Hulls.size;
            step.setHulls(packHulls());
        }
        const encoder = Compute.device.createCommandEncoder({ label: "physics-pack" });
        step.pack(encoder, inputs());
        Compute.device.queue.submit([encoder.finish()]);
    },
};

// the bodies SoA columns `readBody` reads off the Mirror snapshot — imported from step.ts so the
// column indices can't drift from the solver's own layout

// the State-owned body-pose Mirror and cached view over its reused snapshot buffer (`snapshot.bytes` is the
// same ArrayBuffer across readbacks), so `readBody` does not re-wrap a Float32Array on every call.
function readBody(state: State, eid: number): BodyState | null {
    const world = avbdWorld(state);
    const s = world.step;
    if (!s) return null;
    const snap = world.bodyMirror?.snapshot;
    if (!snap) return null;
    if (world.cachedBuf !== snap.bytes) {
        world.cachedBuf = snap.bytes;
        world.cachedView = new Float32Array(snap.bytes);
    }
    const f = world.cachedView as Float32Array;
    const cap = s.eidCap;
    const po = (B_POS * cap + eid) * 4;
    const qo = (B_QUAT * cap + eid) * 4;
    const vo = (B_VELL * cap + eid) * 4;
    return {
        pos: [f[po], f[po + 1], f[po + 2]],
        quat: [f[qo], f[qo + 1], f[qo + 2], f[qo + 3]],
        vel: [f[vo], f[vo + 1], f[vo + 2]],
    };
}

const AvbdStepSystem: System = {
    name: "avbd-step",
    group: "fixed",
    update(state) {
        const world = avbdWorld(state);
        const s = world.step;
        if (!s || !Compute.device) return;
        // readback-bounded color loop (Phase 4.9 Lever 1) + direct color-loop dispatch (rung 0): bound the
        // primal's dispatched color count to the frame-stale used-color count and size the color loop's
        // direct dispatch off the frame-stale live count, both riding one snapshot ([0] = usedColors from
        // colorize, [1] = liveCount from packScan). No snapshot yet (first frames) → both keep the full
        // cap (the safe cold-start).
        if (world.colorMirror?.snapshot) {
            const counts = new Uint32Array(world.colorMirror.snapshot.bytes);
            s.boundColors(counts[0]);
            s.boundBodies(counts[1]);
        }
        const encoder = Compute.device.createCommandEncoder({ label: "physics-step" });
        s.record(encoder);
        Compute.device.queue.submit([encoder.finish()]);
    },
};

// the authored-constraint upload over the physics/core seam: re-derive the defs only when the authored
// signature changes. Reset in warm so a fresh step receives the authored set on its first frame.
const AvbdConstraintSystem: System = {
    name: "avbd-constraints",
    group: "fixed",
    before: [AvbdStepSystem],
    update(state) {
        const world = avbdWorld(state);
        const s = world.step;
        if (!s) return;
        const ss = springSignature(state);
        if (ss !== world.springSig) {
            world.springSig = ss;
            s.setSprings(springDefs(state));
        }
        const js = jointSignature(state);
        if (js !== world.jointSig) {
            world.jointSig = js;
            s.setJoints(jointDefs(state));
        }
    },
};

const AvbdComposeSystem: System = {
    name: "avbd-compose",
    group: "draw",
    after: [BeginFrameSystem],
    before: [PrepassSystem],
    update(state) {
        const step = avbdWorld(state).step;
        if (!step || !Render.encoder) return;
        const transforms = Compute.buffers.get("transforms");
        if (!transforms) return;
        step.compose(Render.encoder, transforms, state.time.fixedAlpha);
    },
};

/**
 * the AVBD rigid-body solver: installs `Body`/`Spring`/`Joint` and a GPU solver for GPU-resident scale. Use
 * it instead of `PhysicsPlugin`, never beside it.
 *
 * @example
 * ```
 * export const config: Config = { plugins: [AvbdPlugin], scene: "scenes/scene.scene" };
 * ```
 */
export const AvbdPlugin: Plugin = {
    name: "Avbd",
    components: { Body, Spring, Joint },
    systems: [AvbdConstraintSystem, AvbdStepSystem, PackSystem, AvbdComposeSystem],
    // MirrorPlugin: the readback-bounded color loop reads `step.colorCount` through a Mirror (above), and
    // `readBody` reads `step.bodies` through one.
    dependencies: [SlabPlugin, MirrorPlugin],
    // the LBVH broadphase's bounds reduction + radix sort prefer subgroup ops, falling back to an
    // LDS arm where absent (WebKit) — preferred, not required, so a no-subgroup device still runs
    // physics (bvh/core; createBvh reads `device.features` to pick the arm).
    preferredFeatures: BVH_FEATURES,
    traits: {
        Body: bodyTraits,
        Spring: springTraits,
        Joint: jointTraits,
    },

    initialize(state) {
        avbdWorld(state).step = null;
    },

    async warm(state: State) {
        if (!Compute.device) return;
        const world = avbdWorld(state);
        world.step?.destroy();
        world.colorMirror?.dispose();
        world.bodyMirror?.dispose();
        world.colorMirror = null;
        world.bodyMirror = null;
        world.lastHullCount = -1; // force a hull re-upload into the fresh step
        world.stamps.clear(); // a fresh step re-seeds every body; the stamp diff arms against this step's slots
        world.bodyMirror = null;
        world.cachedBuf = null;
        world.cachedView = null;
        // the membership gate templates the pack's per-eid skip test. `build` fixes every component's
        // bit up front, so `bit(Body)` is valid here; the State capacity is the eid range the pack walks.
        const { gen, mask } = state.membership.bit(Body);
        world.step = await PhysicsStep.create(Compute.device, state.capacity, state.capacity, {
            gen,
            mask,
        });
        world.bodyMirror = mirror(state, world.step.bodies);
        const transforms = Compute.buffers.get("transforms");
        if (transforms) await world.step.prepareCompose(transforms);
        // static per-step params — the live count is GPU-resident (the pack writes it), not a config field.
        world.step.configure({
            dt: Time.FIXED_DT,
            gravity: GRAVITY,
            alpha: ALPHA,
            penalty: PENALTY_MIN, // fresh-contact seed — ramps via betaLin, persists/decays via gamma
            betaLin: BETA_LIN,
            betaAng: BETA_ANG,
            gamma: GAMMA,
            iterations: ITERATIONS,
            maxColors: MAX_COLORS,
        });
        // mirror the used-color count for the readback-bounded color loop (allocated here, after
        // MirrorPlugin.initialize's Mirror.reset, so it survives the build).
        world.colorMirror = mirror(state, world.step.colorCount);
        world.springSig = 0;
        world.jointSig = 0;
    },

    dispose(state) {
        const world = avbdWorld(state);
        world.stamps.clear();
        world.colorMirror?.dispose();
        world.colorMirror = null;
        world.bodyMirror?.dispose();
        world.bodyMirror = null;
        world.cachedBuf = null;
        world.cachedView = null;
        world.step?.destroy();
        world.step = null;
    },
};

export default AvbdPlugin;
