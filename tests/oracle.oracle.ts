import { expect } from "bun:test";
import { check } from "@dylanebert/shallot/harness/check";
import { SPECULATIVE_DISTANCE } from "./collide";
import { joint } from "./joint";
import { COLLISION_MARGIN, PENALTY_MIN } from "./manifold";
import { length, type Quat, rotate, scale, sub, transform, type Vec3 } from "./math";
import { type Body, body, capsule, massOf, sphere } from "./rigid";
import { makeSolver, type Schedule, type Solver, step } from "./solver";
import { spring } from "./spring";

// The phase-0 standing gate: the TS oracle is the executable AVBD spec. Tolerances are derived
// + checked (the values in the comments were measured against the oracle, not guessed), and the
// gate holds closed-form rows (tightest, no reference) and the scheduler bridge. Scene invariants
// live in corpus.oracle.ts; the C++ fixture-parity rows were cut because their generated fixtures
// are not committed and cannot run here.

// total mechanical energy in the solver's own convention (world-frame diagonal inertia, matching
// the reference's MAng): KE_lin + KE_ang + PE. A dropped pile starts at rest at max height, so
// E(0) is the supremum; a dissipative implicit solver only loses energy.
function energy(s: Solver): number {
    let E = 0;
    for (const b of s.bodies) {
        if (b.mass <= 0) continue;
        const [vx, vy, vz] = b.velLin;
        const [wx, wy, wz] = b.velAng;
        E += 0.5 * b.mass * (vx * vx + vy * vy + vz * vz);
        E += 0.5 * (b.moment[0] * wx * wx + b.moment[1] * wy * wy + b.moment[2] * wz * wz);
        E += b.mass * -s.params.gravity * b.posLin[1];
    }
    return E;
}

const maxSpeed = (s: Solver): number => {
    let m = 0;
    for (const b of s.bodies) m = Math.max(m, length(b.velLin));
    return m;
};

check(
    "free-fall matches the exact discrete symplectic-Euler trajectory",
    {
        claim: "free-fall follows the exact symplectic-Euler trajectory",
        size: "integration",
        budget: 20000,
    },
    () => {
        // no ground ⇒ no contact: the body lands on its inertial target each step, giving
        // v_n = g·dt·n and x_n = x0 + g·dt²·n(n+1)/2 exactly. Measured max error 1.6e-12.
        const dt = 1 / 60;
        const g = -10;
        const y0 = 10;
        const s = makeSolver([body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, y0, 0])], {
            gravity: g,
            dt,
        });
        let maxErr = 0;
        for (let n = 1; n <= 120; n++) {
            step(s);
            const yExact = y0 + (g * dt * dt * (n * (n + 1))) / 2;
            const vExact = g * dt * n;
            maxErr = Math.max(
                maxErr,
                Math.abs(s.bodies[0].posLin[1] - yExact),
                Math.abs(s.bodies[0].velLin[1] - vExact),
            );
        }
        expect(maxErr).toBeLessThan(1e-9);
    },
);

check(
    "substeps = N is exactly N sub-steps of h = dt/N (the small-steps definition)",
    { claim: "substeps preserve the small-steps definition", size: "integration", budget: 20000 },
    () => {
        // one step at substeps=N must equal N manual steps at dt/N with substeps=1 — every dt-bearing
        // term (inertial init, M/h², BDF1 velocity, the velocity-sweep band) uses h, and the manifold
        // map persists across sub-steps exactly as across frames, so the two paths are the same f64 ops.
        // A 3-box stack on the ground exercises contacts + multi-pair warmstart carry between sub-steps.
        const scene = () => [
            body([20, 1, 20], 0, 0.5, [0, 0, 0]),
            body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, 1.0, 0]),
            body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0.02, 1.97, 0]),
        ];
        const N = 4;
        const internal = makeSolver(scene(), { substeps: N });
        const external = makeSolver(scene(), { substeps: 1, dt: 1 / 60 / N });
        for (let f = 0; f < 20; f++) {
            step(internal);
            for (let sub = 0; sub < N; sub++) step(external);
        }
        let maxErr = 0;
        for (let i = 1; i < 3; i++) {
            const a = internal.bodies[i];
            const b = external.bodies[i];
            maxErr = Math.max(
                maxErr,
                length(sub(a.posLin, b.posLin)),
                length(sub(a.velLin, b.velLin)),
            );
        }
        expect(maxErr).toBeLessThan(1e-12); // same f64 op sequence ⇒ bit-identical to round-off
    },
);

check(
    "resting box penetration = mg/(nc·K) (penalty layer, α=0)",
    {
        claim: "penalty resting penetration follows the mg over contact-stiffness law",
        size: "integration",
        budget: 20000,
    },
    () => {
        // Penalty layer with α=0: at rest the constraint is C = C0 and the dq terms vanish, so the
        // nc contact forces balance gravity as nc·K·|C0_n| = mg. With C0_n = (y−1) + MARGIN, the
        // penetration past the (1−MARGIN) margin-rest is mg/(nc·K). nc is read back, not assumed.
        const g = -10;
        const mg = 10; // m = 1, |g| = 10
        const restY = 1 - COLLISION_MARGIN; // the box-touching height the margin holds it at
        const rest = (K: number) => {
            const s = makeSolver(
                [
                    body([100, 1, 100], 0, 0.5, [0, 0, 0]),
                    body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, 1.5, 0]),
                ],
                { layer: "penalty", alpha: 0, penaltyStiffness: K, gravity: g },
            );
            for (let f = 0; f < 300; f++) step(s);
            const m = [...s.manifolds.values()][0];
            return {
                y: s.bodies[1].posLin[1],
                nc: m ? m.contacts.length : 0,
                speed: length(s.bodies[1].velLin),
            };
        };
        const a = rest(1000);
        const b = rest(2000);

        expect(a.speed).toBeLessThan(1e-6); // premise: the box is at rest
        expect(a.nc).toBeGreaterThan(0);
        expect(restY - a.y).toBeCloseTo(mg / (a.nc * 1000), 6); // measured rel error 2e-14
        expect(restY - b.y).toBeCloseTo((restY - a.y) / 2, 6); // mg/k signature: doubling K halves penetration
    },
);

check(
    "dual layer ramps the penalty to hold the box near the margin, not the seed floor",
    {
        claim: "dual penalty ramp holds a resting box at the collision margin",
        size: "integration",
        budget: 20000,
    },
    () => {
        // The Phase-2 gate (roadmap "resting penetration → 0"): λ accumulation + the within-frame
        // penalty ramp let the contact force balance gravity without a large fixed stiffness, so the
        // box rests near the margin where a *fixed* penalty at the same seed (PENALTY_MIN = 1) sinks
        // mg/(nc·1) ≈ O(1) deep (clean through the ground). The dual layer cold-starts λ each frame
        // (no cross-frame warmstart until Phase 3), so it needs the higher betaLin = 1e5 the engine
        // ships — the canonical 1e4 is calibrated for warmstart and under-ramps from cold here.
        const g = -10;
        const restY = 1 - COLLISION_MARGIN;
        const settle = (params: Parameters<typeof makeSolver>[1]) => {
            const s = makeSolver(
                [
                    body([100, 1, 100], 0, 0.5, [0, 0, 0]),
                    body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, 1.5, 0]),
                ],
                params,
            );
            for (let f = 0; f < 300; f++) step(s);
            return { y: s.bodies[1].posLin[1], speed: length(s.bodies[1].velLin) };
        };
        const dual = settle({
            layer: "dual",
            penaltyStiffness: PENALTY_MIN,
            betaLin: 1e5,
            alpha: 0.99,
            gravity: g,
        });
        const seedFloor = settle({
            layer: "penalty",
            penaltyStiffness: PENALTY_MIN,
            alpha: 0.99,
            gravity: g,
        });
        console.log(
            `[oracle] dual(b=1e5) rest y ${dual.y.toFixed(5)} (pen ${(restY - dual.y).toExponential(3)}), ` +
                `fixed-PENALTY_MIN rest y ${seedFloor.y.toFixed(3)} (pen ${(restY - seedFloor.y).toFixed(2)})`,
        );

        // dual rests AT the margin — |penetration| an order below the box scale, so the box center sits
        // in (restY − 1e-2, restY + 1e-2) = [0.98, 1.0]: between the mg/k penetration and the bare touch.
        // The speculative band (Phase 4.8.3) catches the box at the band, so it settles a hair above the
        // bare margin rest (measured −2.3e-4 past restY) rather than the overlap-only ~2e-3 below — either
        // way at the margin, not the deep seed floor, and never hovering above the touch (center > 1.0).
        const dualPen = restY - dual.y;
        expect(Math.abs(dualPen)).toBeLessThan(1e-2);
        // the fixed-PENALTY_MIN baseline sinks O(1) deep — the dual ramp recovers ~2 orders
        expect(restY - seedFloor.y).toBeGreaterThan(1);
    },
);

check(
    "static friction holds on a 30° ramp iff μ ≥ tan 30°",
    {
        claim: "static friction separates walkable and sliding ramp regimes",
        size: "integration",
        budget: 20000,
    },
    () => {
        // Coulomb cone: a box on a θ ramp stays static iff μ ≥ tan θ. Both surfaces friction = μ
        // (μ_eff = √(μ·μ) = μ). Warm up 60 frames to land on the ramp, then measure the slide —
        // below threshold it keeps sliding (kinetic), above it comes to rest. Threshold tan30 = 0.577.
        const angle = Math.PI / 6;
        const q: Quat = [0, 0, Math.sin(angle / 2), Math.cos(angle / 2)];
        const normal = rotate(q, [0, 1, 0]);
        const slideAfterLanding = (mu: number) => {
            const ramp = body([40, 1, 24], 0, mu, [0, 6, 0], [0, 0, 0], q);
            const pos: Vec3 = [
                ramp.posLin[0] + normal[0] * 1.05,
                ramp.posLin[1] + normal[1] * 1.05,
                ramp.posLin[2] + normal[2] * 1.05,
            ];
            const s = makeSolver(
                [
                    body([100, 1, 100], 0, mu, [0, 0, 0]),
                    ramp,
                    body([1, 1, 1], massOf([1, 1, 1], 1), mu, pos),
                ],
                { gravity: -10 },
            );
            for (let f = 0; f < 60; f++) step(s);
            const landed = [...s.bodies[2].posLin] as Vec3;
            for (let f = 0; f < 200; f++) step(s);
            return {
                slid: length(sub(s.bodies[2].posLin, landed)),
                endSpeed: length(s.bodies[2].velLin),
            };
        };

        const below = slideAfterLanding(0.45); // < tan30 → kinetic
        const above = slideAfterLanding(0.75); // > tan30 → static
        expect(below.slid).toBeGreaterThan(2); // measured 4.8
        expect(below.endSpeed).toBeGreaterThan(0.3); // measured 1.5 — still moving
        expect(above.slid).toBeLessThan(0.5); // measured 0.33
        expect(above.endSpeed).toBeLessThan(0.05); // measured 0.003 — at rest
    },
);

check(
    "speculative contact stops a fast in-band box at the surface (no penetration pop / tunnel)",
    {
        claim: "speculative contacts stop fast boxes inside the static band",
        size: "integration",
        budget: 20000,
    },
    () => {
        // Phase 4.8.3: the SAT generates a contact while the boxes are still separated by up to
        // SPECULATIVE_DISTANCE, carrying the true +gap in c0. The repulsion-only normal constraint then
        // limits the approach to close exactly that gap, so a body within the band at frame start lands
        // AT the surface in one step (Box2D / Firth speculative contacts). Overlap-only CD generates no
        // contact for a separated pair, so the box tunnels ~1.7 m below the ground in a single 1/60 s
        // step — the red this gate turns green (revert the testAxis abort to SPECULATIVE_DISTANCE to see).
        const g = -10;
        const touchingY = 1.0; // box-center height where the box (half 0.5) face meets the ground top (0.5)
        const gap = SPECULATIVE_DISTANCE * 0.75; // 0.03 — inside the band, so the contact fires
        const v = 100; // downward; v·dt ≈ 1.67 m ≫ the box, so overlap-only CD tunnels it in one step

        const s = makeSolver(
            [
                body([100, 1, 100], 0, 0.5, [0, 0, 0]),
                body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, touchingY + gap, 0], [0, -v, 0]),
            ],
            { gravity: g },
        );
        step(s);

        const box = s.bodies[1];
        const contacts = [...s.manifolds.values()][0]?.contacts.length ?? 0;
        // the speculative contact fired and caught the box AT the band: it stays near where it entered
        // (measured center ≈ 1.0296, gap barely closed under the α=0.99 stabilization) and does NOT pass
        // through. Overlap-only CD lands the center at ~ −0.64, deep through the ground — so > touchingY
        // − 0.05 is the tunnel discriminator; over later frames it settles to the margin rest (~0.99).
        expect(contacts).toBeGreaterThan(0);
        expect(box.posLin[1]).toBeGreaterThan(touchingY - 0.05); // > 0.95 — caught, not tunnelled (~ −0.64)
        expect(box.posLin[1]).toBeLessThan(touchingY + gap); // only closes the entry gap, never lifts off
        // the 100 m/s approach is killed to near zero in one step — the speculative contact absorbs it
        // (lands at contact, no penetration pop). measured |vy| ≈ 0.024 m/s; 1 m/s is the derived ceiling
        // (≪ the 100 m/s it entered with, so the box can't keep falling through the ground).
        expect(Math.abs(box.velLin[1])).toBeLessThan(1);
    },
);

check(
    "velocity sweep catches a fast box beyond the static band; a static box at the same gap is untouched",
    {
        claim: "velocity sweep catches fast boxes without sweeping static boxes",
        size: "integration",
        budget: 20000,
    },
    () => {
        // Phase 4.8.4: the static band (4.8.3) only generates a contact within SPECULATIVE_DISTANCE at
        // frame start, so a body crossing the whole contact between frames (v·dt ≫ the band) still
        // tunnels. The velocity sweep extends the per-axis SAT band by the closing displacement
        // max(0, dot(dRel, n)) (dRel = (vA−vB)·dt), so the frame-start SAT generates the swept contact
        // and the existing repulsion-only constraint limits the approach (Box2D/Bullet/Firth speculative
        // contacts). It is velocity-GATED: a static body at the same gap stays separated (the swept band
        // degenerates to 4.8.3 at vRel=0). Red without the sweep: gap 0.5 ≫ the 0.04 band ⇒ 0 contacts ⇒
        // the box free-falls ~1.67 m, ~ −0.17 deep through the ground spanning [−0.5, 0.5].
        const g = -10;
        const touchingY = 1.0; // box-center height where the box (half 0.5) face meets the ground top (0.5)
        const gap = 0.5; // ≫ SPECULATIVE_DISTANCE (0.04): the static band alone generates nothing
        const v = 100; // v·dt ≈ 1.67 m ≫ the gap, so the box would tunnel in one step without the sweep

        const s = makeSolver(
            [
                body([100, 1, 100], 0, 0.5, [0, 0, 0]),
                body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, touchingY + gap, 0], [0, -v, 0]),
            ],
            { gravity: g },
        );
        step(s);
        const box = s.bodies[1];
        const contacts = [...s.manifolds.values()][0]?.contacts.length ?? 0;
        // the swept contact fired and arrested the box: it stays near where it entered (closes only the
        // (1−α) stabilization fraction this step, ~0.005 m) well above the ground — it did NOT tunnel.
        expect(contacts).toBeGreaterThan(0);
        expect(box.posLin[1]).toBeGreaterThan(touchingY - 0.05); // caught above the surface, not tunnelled
        expect(box.posLin[1]).toBeLessThan(touchingY + gap + 1e-3); // only closes toward contact, never rises
        // the 100 m/s approach is killed in one step (measured |vy| ≈ 0.3 m/s ≪ 100 — it can't keep falling)
        expect(Math.abs(box.velLin[1])).toBeLessThan(1);

        // velocity-gated: the SAME 0.5 gap with a STATIC box is past the band → no swept contact (4.8.3).
        const stat = makeSolver(
            [
                body([100, 1, 100], 0, 0.5, [0, 0, 0]),
                body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, touchingY + gap, 0]),
            ],
            { gravity: g },
        );
        step(stat);
        const statContacts = [...stat.manifolds.values()][0]?.contacts.length ?? 0;
        expect(statContacts).toBe(0); // separated past the band, no contact generated
    },
);

{
    // Phase 6.1: the first non-contact Force — a soft distance constraint (spring.ts, a port of
    // spring.cpp). C = ‖pA − pB‖ − rest, force f = stiffness·C, no dual (finite stiffness ⇒ λ = 0).
    // The scene is the demo's sceneSpring stripped to the spring alone (no ground, irrelevant here):
    // a static anchor at y=14, a 2×2×2 block (mass 8) hanging from a center-anchored spring (k=100,
    // rest=4). Center anchors ⇒ rWorld = 0 ⇒ zero torque ⇒ a clean 1-DOF vertical oscillator, so the
    // closed forms are exact: static extension ext = m|g|/k, frequency ω = √(k/m).

    const k = 100;
    const rest = 4;
    const anchorY = 14;
    const g = -10;
    const dt = 1 / 60;
    const size: Vec3 = [2, 2, 2];
    const m = massOf(size, 1); // 8
    const ext = (m * Math.abs(g)) / k; // mg/k = 0.8 — the static spring extension past rest
    const yEq = anchorY - rest - ext; // 9.2 — the block's equilibrium height
    const omega = Math.sqrt(k / m); // √(k/m) = 3.5355 rad/s
    const wh = omega * dt;
    const decay = 1 / Math.sqrt(1 + wh * wh); // BDF1 per-step amplitude factor |λ|

    const hang = (yStart: number, vy = 0): Solver => {
        const anchor = body([1, 1, 1], 0, 0.5, [0, anchorY, 0]);
        const block = body(size, m, 0.5, [0, yStart, 0], [0, vy, 0]);
        const s = makeSolver([anchor, block], { gravity: g, dt });
        s.springs.push(spring(anchor, block, [0, 0, 0], [0, 0, 0], k, rest));
        return s;
    };

    check(
        "static extension is mg/k — the exact fixed point, and the attractor from a displaced start",
        {
            claim: "spring fixed point and attraction follow mg over stiffness",
            size: "integration",
            budget: 20000,
        },
        () => {
            // Phase 1 — the equilibrium is the AVBD fixed point. Placed at yEq the spring force k·ext = mg
            // exactly cancels gravity, so dx = 0 every step (no transient): a direct check that the force
            // MAGNITUDE is k·C with the equilibrium at extension mg/k. A wrong force law (wrong sign/scale)
            // moves the fixed point, so this discriminates. Measured drift: ~1e-13.
            const atEq = hang(yEq);
            let maxDrift = 0;
            for (let f = 0; f < 200; f++) {
                step(atEq);
                maxDrift = Math.max(maxDrift, Math.abs(atEq.bodies[1].posLin[1] - yEq));
            }
            expect(maxDrift).toBeLessThan(1e-9);
            expect(length(atEq.bodies[1].velLin)).toBeLessThan(1e-9);

            // Phase 2 — yEq is the ATTRACTOR. From y=8 (the demo start, displaced 1.2 below eq) the lightly
            // damped oscillation decays toward yEq. After F frames the analytic envelope is A0·decay^F; F=5000
            // ⇒ ~2e-4, far below the ~0.4 m a wrong equilibrium would sit at, so 1e-2 both passes and pins mg/k.
            const F = 5000;
            const settling = hang(8);
            for (let f = 0; f < F; f++) step(settling);
            const env = 1.2 * decay ** F; // ~2e-4
            expect(env).toBeLessThan(1e-3); // sanity: the run is long enough to have damped
            expect(Math.abs(settling.bodies[1].posLin[1] - yEq)).toBeLessThan(1e-2);
            expect(length(settling.bodies[1].velLin)).toBeLessThan(5e-3); // came to rest at eq
        },
    );

    check(
        "oscillates at ω = √(k/m) — the exact BDF1 discrete period",
        {
            claim: "spring oscillation follows the BDF1 discrete period",
            size: "integration",
            budget: 20000,
        },
        () => {
            // Phase 2 closed form: displaced and released, the block is a damped harmonic oscillator. BDF1 on
            // u'' = −ω²u has discrete eigenvalue arg(λ) = atan(ωh), so the position zero-crossings about yEq
            // sit EXACTLY π/atan(ωh) frames apart (a decaying cosine crosses zero independent of its envelope).
            // ⇒ period Tdisc = h·2π/atan(ωh), which is the continuous Tcont = 2π√(m/k) pulled +0.12% by the
            // ωh/atan(ωh) discretization factor. Both reference √(k/m); the measured period pins the frequency.
            const Tdisc = (dt * 2 * Math.PI) / Math.atan(wh); // 1.77926 s — the exact discrete prediction
            const Tcont = (2 * Math.PI) / omega; // 1.77715 s — the continuous closed form

            const ys: number[] = [];
            const s = hang(8);
            for (let f = 0; f < 1100; f++) {
                step(s);
                ys.push(s.bodies[1].posLin[1]);
            }
            // interpolated zero-crossings of u = y − yEq (linear root between bracketing samples)
            const crossings: number[] = [];
            for (let f = 1; f < ys.length; f++) {
                const u0 = ys[f - 1] - yEq;
                const u1 = ys[f] - yEq;
                if (u0 < 0 !== u1 < 0) crossings.push(f - 1 + u0 / (u0 - u1));
            }
            let sumHalf = 0;
            for (let i = 1; i < crossings.length; i++) sumHalf += crossings[i] - crossings[i - 1];
            const avgHalf = sumHalf / (crossings.length - 1); // frames per half-period
            const measuredT = 2 * avgHalf * dt;
            console.log(
                `[oracle/spring] period measured ${measuredT.toFixed(6)}s, Tdisc ${Tdisc.toFixed(6)}s, ` +
                    `Tcont ${Tcont.toFixed(6)}s, ${crossings.length} crossings`,
            );

            expect(crossings.length).toBeGreaterThan(20); // ~20 half-periods over 1100 frames — well sampled
            // crossings are exactly π/atan(ωh) frames apart, so the only error is the sub-frame interpolation
            // (~1e-6 rel, averaged over ~12 intervals) — measuredT matches Tdisc to <5e-4. And it lands within
            // the derived 0.12% of the continuous 2π√(m/k).
            expect(measuredT).toBeCloseTo(Tdisc, 3); // |Δ| < 5e-4 s
            expect(measuredT).toBeCloseTo(Tcont, 2); // |Δ| < 5e-3 s — the continuous limit, +0.12% off
        },
    );
}

{
    // Phase 6.2: the hard Force (joint.ts, a port of joint.cpp). Two stacked constraints — a linear anchor
    // pin (C = pA − pB) and an angular relative-orientation lock (C = (qA − qB)·torqueArm) — carrying
    // warmstartable λ + a per-iteration penalty ramp, with the rigid (∞-stiffness) form adding the explosive-
    // error stabilization C −= α·C₀. The two shipped types are configs of the SAME force: spherical
    // (stiffnessAng = 0 → the angular rows never activate, rotation free) and fixed (stiffnessAng = ∞). Without
    // a joint the body free-falls, so these gates pin the joint behavior, not gravity.

    const g = -10;
    const dt = 1 / 60;
    const size: Vec3 = [0.5, 0.5, 0.5];
    const m = massOf(size, 1); // 0.125

    check(
        "spherical-joint pendulum swings at the physical-pendulum period 2π√(I_pivot/(m·g·d))",
        {
            claim: "spherical joint pendulum follows the physical period",
            size: "integration",
            budget: 20000,
        },
        () => {
            // A bob pinned by a spherical joint d from its COM is a physical pendulum: gravity torques it about
            // the pin, rotation otherwise free (the spherical joint's linear+angular Jacobian coupling makes it
            // a pin). Small angle ⇒ θ'' = −ω²θ with ω² = m·g·d / I_pivot, I_pivot = I_com + m·d² (parallel axis
            // about the swing-plane z). BDF1 on that ODE crosses zero exactly π/atan(ωh) frames apart, so the
            // period is Tdisc = h·2π/atan(ωh) — the same discretization the spring test pins, and +0.26% of the
            // simple-pendulum 2π√(d/g) (the I_com/(m·d²) = 0.46% box-moment correction). The arm MUST coincide at
            // t=0: COM d below the pivot, rB = +d (local up to the pin), the body rotated by θ0 so rotate(qθ0, rB)
            // lands on the pivot — an offset start instead injects energy (the next test's footgun).
            const d = 3;
            const H = 8;
            const theta0 = 0.05;
            const iCom = (m * (size[0] * size[0] + size[1] * size[1])) / 12; // about z
            const iPivot = iCom + m * d * d;
            const omega = Math.sqrt((m * Math.abs(g) * d) / iPivot);
            const wh = omega * dt;
            const Tdisc = (dt * 2 * Math.PI) / Math.atan(wh); // 3.45046 s
            const Tsimple = 2 * Math.PI * Math.sqrt(d / Math.abs(g)); // 2π√(L/g) = 3.44144 s

            const qz: Quat = [0, 0, Math.sin(theta0 / 2), Math.cos(theta0 / 2)];
            const pivot = body(size, 0, 0.5, [0, H, 0]);
            const bob = body(
                size,
                m,
                0.5,
                [d * Math.sin(theta0), H - d * Math.cos(theta0), 0],
                [0, 0, 0],
                qz,
            );
            const s = makeSolver([pivot, bob], { gravity: g, dt });
            s.joints.push(joint(pivot, bob, [0, 0, 0], [0, d, 0]));
            // the arm coincides at t=0 — a clean hanging small-angle pendulum, not the energy-injecting offset
            const armErr = length(
                sub(
                    transform(pivot.posLin, pivot.posAng, [0, 0, 0]),
                    transform(bob.posLin, bob.posAng, [0, d, 0]),
                ),
            );
            expect(armErr).toBeLessThan(1e-12);

            const ang: number[] = [];
            for (let f = 0; f < 1400; f++) {
                step(s);
                ang.push(Math.atan2(bob.posLin[0], H - bob.posLin[1])); // swing angle from vertical
            }
            const crossings: number[] = [];
            for (let f = 1; f < ang.length; f++) {
                const a0 = ang[f - 1];
                const a1 = ang[f];
                if (a0 < 0 !== a1 < 0) crossings.push(f - 1 + a0 / (a0 - a1));
            }
            let sumHalf = 0;
            for (let i = 1; i < crossings.length; i++) sumHalf += crossings[i] - crossings[i - 1];
            const measuredT = (2 * sumHalf * dt) / (crossings.length - 1);

            // small angle stays small (no energy injected — the arm started coincident), so the linear period holds
            expect(Math.max(...ang.map(Math.abs))).toBeLessThan(theta0 * 1.01); // measured max 0.0500
            expect(crossings.length).toBeGreaterThan(10); // ~14 half-periods over 1400 frames
            // crossings sit exactly π/atan(ωh) apart, so the only error is sub-frame interpolation — measuredT
            // matches Tdisc to < 5e-4 (measured 2.6e-4) and the simple form to < 0.5% (measured 0.27%)
            expect(measuredT).toBeCloseTo(Tdisc, 3); // |Δ| < 5e-4 s
            expect(measuredT).toBeCloseTo(Tsimple, 1); // |Δ| < 0.05 s — the box-moment-corrected 2π√(L/g)
        },
    );

    check(
        "a spherical joint leaves rotation free; a fixed joint locks it (a static anchor pins a body rigid)",
        {
            claim: "spherical and fixed joints separate free and locked rotation",
            size: "integration",
            budget: 20000,
        },
        () => {
            // Same 2-body rig — a dynamic box pinned to a static anchor at a 2 m arm — under gravity. The angular
            // row triple is the ONLY difference: spherical (stiffnessAng 0) lets the box swing down (rotation free
            // → a pendulum); fixed (stiffnessAng ∞) locks the box's orientation to the anchor's, and with the
            // linear pin that leaves zero DOF → the box is held rigid in place. Isolates the fixed joint's angular
            // rows from the shared linear ones.
            const H = 5;
            const d = 2;
            const run = (stiffAng: number): Body => {
                const anchor = body(size, 0, 0.5, [0, H, 0]);
                const b = body(size, m, 0.5, [d, H, 0]);
                const s = makeSolver([anchor, b], { gravity: g, dt });
                s.joints.push(
                    joint(anchor, b, [0, 0, 0], [-d, 0, 0], Number.POSITIVE_INFINITY, stiffAng),
                );
                for (let f = 0; f < 240; f++) step(s);
                return b;
            };
            const free = run(0); // spherical
            const locked = run(Number.POSITIVE_INFINITY); // fixed

            expect(free.posLin[1]).toBeLessThan(H - 1); // swings down ~2 m (measured 3.00, a drop of ~2)
            expect(Math.abs(free.posAng[2])).toBeGreaterThan(0.5); // rotated freely (measured |qz| 0.73)
            expect(length(sub(locked.posLin, [d, H, 0]))).toBeLessThan(1e-2); // held rigid (measured 2.5e-3)
            expect(Math.abs(locked.posAng[2])).toBeLessThan(1e-2); // no z-rotation (measured 5e-4)
        },
    );

    check(
        "a finite-intermediate stiffnessAng settles to a pinned rest, not free-fall (mirroring physics's 1000-case)",
        {
            claim: "finite joint angular stiffness pins and settles the body",
            size: "integration",
            budget: 20000,
        },
        () => {
            // S2 grant arm: a legitimate finite-positive stiffnessAng (1000) still builds a joint under AVBD —
            // the over-refusal check no refusal arm can show. Mirrors Shallot physics's stiffnessAng = 1000 settle
            // arm ("an intermediate stiffnessAng settles rather than oscillates"), scored
            // against the CPU oracle here. The setup matches the spherical/fixed test above — a dynamic box
            // pinned to a static anchor at a 2 m arm under gravity — with the angular stiffness set to the
            // finite-intermediate 1000 (not 0 = spherical, not ∞ = fixed). Gravity torques the box about the
            // pin; the finite angular spring resists, and the dissipative BDF1 solver settles it to the
            // gravity-balanced equilibrium — pinned (not free-falling like the spherical case) and settled
            // (not oscillating like a free pendulum). The angular deflection at stiffnessAng 1000 is tiny (the
            // penalty ramp + torqueArm scaling make the effective stiffness ≫ the gravity torque), so the
            // observable signature is the PIN + SETTLE, not a large rotation: the body stays near its spawn
            // pose and comes to rest, while the spherical case swings down ~2 m at ~4.7 m/s.
            //
            // witnessed red by mutation: exit code 1 — removing the angular constraint block from stampJoint
            // (the `if (lengthSq(j.penaltyAng) > 0)` guard, setting penaltyAng to [0,0,0] so the rows never
            // stamp) makes the joint behave as spherical: the body swings down to pos[1] ≈ 3.0 at vel ≈ 4.7,
            // failing the pinned (|pos − [d,H,0]| < 0.1) and settled (vel < 0.05) assertions. The arm thus pins
            // the angular constraint's load-bearingness, not a restatement of the linear pin.
            const H = 5;
            const d = 2;
            const anchor = body(size, 0, 0.5, [0, H, 0]);
            const b = body(size, m, 0.5, [d, H, 0]);
            const s = makeSolver([anchor, b], { gravity: g, dt });
            s.joints.push(joint(anchor, b, [0, 0, 0], [-d, 0, 0], Number.POSITIVE_INFINITY, 1000));
            for (let f = 0; f < 240; f++) step(s);

            // pinned: the body stays near its spawn pose (measured |pos − [2, 5, 0]| ≈ 1.2e-3), not swung
            // down like the spherical case (which lands at pos[1] ≈ 3.0, a ~2 m drop). The linear pin is
            // rigid (∞ stiffnessLin), so this asserts the joint was created and the linear rows are active.
            expect(length(sub(b.posLin, [d, H, 0]))).toBeLessThan(0.1);
            // settled: the body is at rest (measured vel ≈ 2e-4, angVel ≈ 5e-5), not oscillating like a free
            // pendulum (which is still moving at ~4.7 m/s after 240 frames). The finite angular spring + BDF1
            // damping converge to the gravity-balanced equilibrium.
            expect(length(b.velLin)).toBeLessThan(0.05);
            expect(length(b.velAng)).toBeLessThan(0.05);
            // angular constraint active: the body's rotation is far below the spherical case's ~1.62 rad
            // (measured 5e-4 rad). A spherical joint (stiffnessAng 0) would have rotated the body ~93°; the
            // finite angular spring holds it near its initial orientation. The threshold (0.1 rad ≈ 6°) is
            // far below the spherical case and far above the measured deflection, so it cleanly separates.
            const q0: Quat = [0, 0, 0, 1]; // identity — the body's spawn orientation
            const dot =
                q0[0] * b.posAng[0] +
                q0[1] * b.posAng[1] +
                q0[2] * b.posAng[2] +
                q0[3] * b.posAng[3];
            const rotation = 2 * Math.acos(Math.min(1, Math.abs(dot)));
            expect(rotation).toBeLessThan(0.1);
        },
    );

    check(
        "a grossly non-coincident joint fails loudly at construction (the rope-explosion footgun)",
        {
            claim: "non-coincident joints refuse at construction",
            size: "integration",
            budget: 20000,
        },
        () => {
            // Reproduces the legacy rope bug + proves the guard catches it. A rigid joint whose anchors start far
            // apart recovers spurious velocity through BDF1 as it corrects (measured: a 4.2 m mismatch injects +34%
            // energy, 6.6 m/s). joint() throws when the anchors exceed the bodies' combined reach, so the mistake
            // surfaces at build time, not as an explosion 50 frames in. A coincident (or ≤ reach) start constructs
            // fine — the α-stabilization absorbs a small offset.
            const pivot = body(size, 0, 0.5, [0, 8, 0]);
            const bob = body(size, m, 0.5, [0, 5, 0]); // hangs 3 m below, identity orientation
            expect(() => joint(pivot, bob, [0, 0, 0], [0, 3, 0])).not.toThrow(); // rB = +y → anchors meet
            expect(() => joint(pivot, bob, [0, 0, 0], [-3, 0, 0])).toThrow(/coincident/); // 4.2 m mismatch → loud
        },
    );

    check(
        "a joint between two non-dynamic bodies fails loudly at construction (the both-static energy guard)",
        {
            claim: "all-static joints refuse while static anchors remain valid",
            size: "integration",
            budget: 20000,
        },
        () => {
            // A joint no dynamic body can resolve — both endpoints mass ≤ 0 (static/kinematic) — is never satisfied
            // by the primal (both skip it, solver.ts), so its dual penalty + λ ramp unbounded: the joint analog of
            // the contact all-static dual guard (manifold.ts updateDual). The harm surfaces when such an endpoint is
            // later released (made dynamic) — the accumulated huge λ yanks it. joint() rejects it at construction;
            // the GPU jointInit deactivates + bumps counters[1]. A joint with ONE dynamic body (a grab / pendulum to
            // a static anchor) is the LEGITIMATE case and must NOT be rejected. Anchors coincident so the rope guard
            // can't mask the both-static one. (Red before the guard: this construction does not throw.)
            const staticA = body(size, 0, 0.5, [0, 8, 0]);
            const staticB = body(size, 0, 0.5, [0, 8, 0]);
            const dyn = body(size, m, 0.5, [0, 8, 0]);
            expect(() => joint(staticA, staticB, [0, 0, 0], [0, 0, 0])).toThrow(/non-dynamic/);
            expect(() => joint(staticA, dyn, [0, 0, 0], [0, 0, 0])).not.toThrow(); // static anchor + dynamic = fine
        },
    );

    check(
        "the grab dangles from a world anchor without injecting energy; a kinematic-anchor BODY flails",
        {
            claim: "world-anchor grabs damp without kinematic-anchor energy injection",
            size: "integration",
            budget: 20000,
        },
        () => {
            // avbd-demo3d's mouse-drag grab pins the box to a WORLD-space anchor (a = null, rA the cursor point) by a
            // soft SPHERICAL joint (joint.cpp `bodyA == null`): the box dangles (rotation free), and with NO anchor
            // body there's no anchor↔box contact. Dragging pumps the pendulum; holding the anchor still does no work,
            // so the box can only DISSIPATE (a damped pendulum) — energy never grows, never NaN, never the chaotic
            // flail the report feared. Contrast: a kinematic-anchor BODY (the old workaround) embeds a sphere
            // in the box surface whose CONTACT keeps shoving it, so that one stays agitated — why the grab joints to
            // the world (no body), not a kinematic anchor.
            const dragThenHold = (
                worldAnchor: boolean,
            ): { early: number; late: number; finite: boolean } => {
                const box = body([0.8, 0.8, 0.8], 0.4, 0.5, [0, 3, 0]); // held in the air (no ground)
                const arm: Vec3 = [0.4, 0, 0]; // grabbed at a +x face-surface point → offset, so it swings
                const hit: Vec3 = [0.4, 3, 0];
                const bodies: Body[] = [box];
                let j: ReturnType<typeof joint>;
                if (worldAnchor) {
                    j = joint(null, box, [...hit] as Vec3, arm, 5000, 0); // world anchor — no body, no contact
                } else {
                    const anchor = sphere(0.05, 0, 0, [...hit] as Vec3); // a kinematic anchor BODY → a contact
                    bodies.push(anchor);
                    j = joint(anchor, box, [0, 0, 0], arm, 5000, 0);
                }
                const s = makeSolver(bodies, { gravity: -10, iterations: 4 });
                s.joints.push(j);
                const setAnchor = (p: Vec3): void => {
                    if (worldAnchor) {
                        j.rA = p; // mutate the world anchor point (the reference's `drag->rA = …`)
                    } else {
                        const a = bodies[1];
                        a.velLin = scale(sub(p, a.posLin), 60);
                        a.posLin = p;
                    }
                };
                const base: Vec3 = [0.4, 3, 0];
                // whip the anchor around a fast circle (radius 0.5, ~4 m/s), then hold it dead still for 6 s
                for (let f = 0; f < 240; f++) {
                    const t = f / 60;
                    setAnchor([
                        base[0] + Math.cos(t * 8) * 0.5 - 0.5,
                        base[1],
                        base[2] + Math.sin(t * 8) * 0.5,
                    ]);
                    step(s);
                }
                const hold: Vec3 = worldAnchor
                    ? ([...j.rA] as Vec3)
                    : ([...bodies[1].posLin] as Vec3);
                let early = 0;
                let late = 0;
                let finite = true;
                for (let f = 0; f < 360; f++) {
                    setAnchor(hold);
                    step(s);
                    const w = length(box.velAng);
                    if (f >= 60 && f < 120) early = Math.max(early, w); // peak swing in hold window [1s, 2s]
                    if (f >= 240 && f < 300) late = Math.max(late, w); //  …and in window [4s, 5s]
                    if (!box.posLin.every(Number.isFinite)) finite = false;
                }
                return { early, late, finite };
            };

            const world = dragThenHold(true);
            expect(world.finite).toBe(true); // never explodes
            expect(world.early).toBeGreaterThan(0.1); // it DANGLES — a spherical joint leaves rotation free
            expect(world.late).toBeLessThan(world.early); // and DAMPS: the held pendulum dissipates, no injection
            expect(world.early).toBeLessThan(15); // bounded swing, never a chaotic blow-up

            // the kinematic-anchor BODY stays more agitated than the world anchor (its embedded-sphere contact keeps
            // feeding energy in) — the reason the grab joints to the world, not a body. Measured world ≈ 2, kin ≈ 4.
            const kin = dragThenHold(false);
            expect(kin.late).toBeGreaterThan(world.late);
        },
    );

    // a small settling stack: ground + 4 boxes. Adjacent boxes share a contact.
    const smallStack = (): Body[] => {
        const bs = [body([100, 1, 100], 0, 0.5, [0, 0, 0])];
        for (let i = 0; i < 4; i++)
            bs.push(body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, 1.0 + i, 0]));
        return bs;
    };
    const restOf = (schedule: Schedule): Vec3[] => {
        const s = makeSolver(smallStack());
        for (let f = 0; f < 250; f++) step(s, schedule);
        return s.bodies.map((b) => [...b.posLin] as Vec3);
    };

    check(
        "colored with reverse-rank colors is bit-identical to sequential",
        {
            claim: "reverse-rank coloring preserves sequential results",
            size: "integration",
            budget: 20000,
        },
        () => {
            // colors[i] = n−1−i gives every body its own color; ascending color order then visits bodies
            // n−1 … 0 — the same reverse-creation order sequential GS uses, and a one-body color commits
            // immediately, so the deferred-commit colored path reduces to sequential exactly. Measured: 0.
            const seq = restOf({ kind: "sequential" });
            const n = smallStack().length;
            const colored = restOf({
                kind: "colored",
                colors: Array.from({ length: n }, (_, i) => n - 1 - i),
            });
            let maxDiff = 0;
            for (let i = 0; i < seq.length; i++)
                maxDiff = Math.max(maxDiff, length(sub(seq[i], colored[i])));
            expect(maxDiff).toBeLessThan(1e-12);
        },
    );

    check(
        "a valid coloring reaches the same rest as sequential (coloring-preserves-solution)",
        {
            claim: "valid coloring preserves the sequential settled solution",
            size: "integration",
            budget: 20000,
        },
        () => {
            // alternate boxes by height — no same-color pair shares a contact, so colored GS converges to
            // the sequential fixed point (transient differs, rest does not). Measured 8e-6.
            const seq = restOf({ kind: "sequential" });
            const colored = restOf({ kind: "colored", colors: [0, 1, 0, 1, 0] });
            let maxDiff = 0;
            for (let i = 0; i < seq.length; i++)
                maxDiff = Math.max(maxDiff, length(sub(seq[i], colored[i])));
            expect(maxDiff).toBeLessThan(1e-3);
        },
    );
}

// The Phase-3 warmstart crux at the spec level: the reference's force-list manifold persistence — `initManifold` merges this frame's
// contacts onto last frame's by feature key, carrying λ/k with γ decay (manifold.ts). These rows
// close the documented gap —
// a churning scene (flipping feature keys) and the positive "warmstart converges tighter" property
// that proves the persisted state actually does work. The GPU reconstructs this merge (step.ts);
// the gym `pile` stack-warmstart gate verifies GPU == this oracle on the real device.

// a 5-box chain on the ground — the bottom contact carries the whole stack, the regime the paper
// flags as wanting more iterations (a "series of connections"). Warmstart accumulates λ/k across
// frames, so it converges the chain in fewer per-frame iterations than a cold reset.
const chain = (): Body[] => {
    const bs = [body([100, 1, 100], 0, 0.5, [0, 0, 0])];
    for (let i = 0; i < 5; i++)
        bs.push(body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, 1.0 + i, 0]));
    return bs;
};
const restY = 1 - COLLISION_MARGIN; // the bottom box's box-touching height

const settle = (layer: "dual" | "warmstart", iterations: number) => {
    const s = makeSolver(chain(), {
        layer,
        iterations,
        betaLin: 1e4, // canonical — the same ramp for both, so warmstart's edge is persistence alone
        alpha: 0.99,
        gamma: 0.999,
        gravity: -10,
    });
    for (let f = 0; f < 240; f++) step(s);
    return { pen: restY - s.bodies[1].posLin[1], speed: maxSpeed(s) };
};

check(
    "warmstart converges the chain in fewer iterations than a cold reset",
    {
        claim: "warmstart settles a chain with fewer iterations than cold reset",
        size: "integration",
        budget: 20000,
    },
    () => {
        // At the canonical 1e4 ramp and 4 iters, the cold `dual` layer (λ/k reset every frame) leaves the
        // load-bearing bottom contact deeply penetrated — 4 iters can't propagate the 5-box load down a
        // cold chain. Warmstart carries the ramped λ/k across frames, so the same 4 iters converge far
        // tighter — and at least as tight as 10 cold iters (the "fewer iterations" property). Measured:
        // cold-4it pen ~0.73 m, warmstart-4it ~2.6e-3 m (~280× tighter, and < cold-10it).
        const cold4 = settle("dual", 4);
        const warm4 = settle("warmstart", 4);
        const cold10 = settle("dual", 10);
        console.log(
            `[oracle/warmstart] chain bottom penetration — cold(dual,4it) ${cold4.pen.toExponential(2)} m, ` +
                `warmstart(4it) ${warm4.pen.toExponential(2)} m, cold(dual,10it) ${cold10.pen.toExponential(2)} m`,
        );
        expect(warm4.speed).toBeLessThan(0.05); // premise: warmstart settles
        expect(warm4.pen).toBeGreaterThan(0); // still touching (positive penetration)
        expect(warm4.pen).toBeLessThan(cold4.pen * 0.5); // ≥2× tighter than equal-iter cold — persistence works
        expect(warm4.pen).toBeLessThan(cold10.pen + 1e-3); // 4 warmstart iters ≥ 10 cold ones (fewer iterations)
        expect(warm4.pen).toBeLessThan(1e-2); // and converged near the margin
    },
);

check(
    "churning contacts (tipping box): energy non-increasing, no λ blow-up, settles flat",
    {
        claim: "churning contact keys remain finite and settle the tipping box",
        size: "integration",
        budget: 20000,
    },
    () => {
        // A box tilted ~40° about z, dropped from rest onto the ground. It lands on an edge, tips, and
        // settles flat — the contact feature flips (edge-edge key → face manifold keys) as it tips,
        // churning the keys the warmstart merge is keyed on. A bad merge (λ carried onto the wrong
        // feature, the legacy instability) injects energy; the gate is that mechanical energy never
        // exceeds the drop's E0 (the supremum — the box starts at rest) within round-off, stays finite,
        // and the box settles flat (proving the edge→face churn actually happened, then persisted).
        const a = 0.7; // ~40° about z
        const q: Quat = [0, 0, Math.sin(a / 2), Math.cos(a / 2)];
        const s = makeSolver(
            [
                body([100, 1, 100], 0, 0.6, [0, 0, 0]),
                body([1, 1, 1], massOf([1, 1, 1], 1), 0.6, [0, 2.5, 0], [0, 0, 0], q),
            ],
            { layer: "warmstart", gravity: -10 },
        );
        const E0 = energy(s);
        let maxExcess = 0;
        let finite = true;
        for (let f = 0; f < 300; f++) {
            step(s);
            if (!s.bodies[1].posLin.every(Number.isFinite)) finite = false;
            maxExcess = Math.max(maxExcess, (energy(s) - E0) / Math.abs(E0));
        }
        console.log(
            `[oracle/warmstart] tipping-box max energy excess ${maxExcess.toExponential(2)}`,
        );
        expect(finite).toBe(true);
        // E0 is the supremum (dropped from rest); a dissipative solver only loses energy. The penalty
        // spring loads on impact but never lifts the box above its drop height, so E(t) ≤ E0 holds to a
        // small slack. Measured ~1e-3; 1e-2 is a derived margin (a real blow-up is O(1)+ within frames).
        expect(maxExcess).toBeLessThan(1e-2);
        // settled flat on a face: y near the box-touching rest, at rest
        expect(maxSpeed(s)).toBeLessThan(0.05);
        expect(s.bodies[1].posLin[1]).toBeGreaterThan(0.9);
        expect(s.bodies[1].posLin[1]).toBeLessThan(1.05);
    },
);

// The headline correctness fix: a kinematic body (a character, mass ≤ 0, moved by its controller)
// pushed into an immovable surface (a static wall, also mass ≤ 0). Neither body's primal runs
// (both mass ≤ 0), so the contact constraint C is never satisfied — it stays penetrating frame
// after frame. The reference's dual update (solver.cpp:230 runs updateDual on EVERY force) then
// ramps that contact's penalty `k += βLin·|C|` every iteration, every frame, with nothing ever
// moving to relax C — the escalating constraint force the legacy stack blew up on. The fix is at the dual update: a contact NO dynamic body can resolve
// must not ramp. Validated here (red without the gate: the penalty escalates), bounded with it.

const maxNormalPenalty = (s: Solver): number => {
    let p = 0;
    for (const m of s.manifolds.values()) for (const c of m.contacts) p = Math.max(p, c.penalty[0]);
    return p;
};

check(
    "a kinematic capsule held into a static wall does NOT ramp the penalty (both mass ≤ 0)",
    {
        claim: "all-static kinematic contact penalties do not ramp",
        size: "integration",
        budget: 20000,
    },
    () => {
        // A static box wall + a capsule character (mass 0 = kinematic) overlapping its +x face. Both are
        // mass ≤ 0, so the primal skips both — the capsule stays put, penetrating, and the contact can
        // never close. Without the gate the dual ramps this unsolvable contact's penalty unbounded
        // (the legacy escalation); with it the penalty holds at the PENALTY_MIN seed (the decay clamps
        // a never-ramped contact to the floor). The bodies don't move either way (no primal), so this
        // isolates the ramp pathology, not a position effect.
        const wall = body([1, 4, 4], 0, 0.5, [0, 1, 0]); // static, +x face at x = 0.5
        const char = capsule(0.5, 0.3, 0, 0.5, [0.6, 1, 0]); // kinematic (mass 0), surface ~0.2 m into the wall
        const s = makeSolver([wall, char], { layer: "warmstart", gravity: -10 });

        let penAt5 = 0;
        for (let f = 0; f < 300; f++) {
            step(s);
            if (f === 5) penAt5 = maxNormalPenalty(s);
        }
        const penFinal = maxNormalPenalty(s);
        console.log(
            `[oracle/§6.4] kin-vs-static normal penalty: frame 5 ${penAt5.toExponential(2)}, ` +
                `frame 300 ${penFinal.toExponential(2)} (seed ${PENALTY_MIN})`,
        );

        // premise: the contact exists and the capsule is genuinely penetrating (else there's nothing to
        // ramp and the test is vacuous)
        expect([...s.manifolds.values()].some((m) => m.contacts.length > 0)).toBe(true);
        // the invariant: the penalty never climbs off the seed floor (a never-ramped contact decays to
        // PENALTY_MIN). Without the fix it escalates into the 1e5–1e10 range — `< 2` cleanly separates the
        // bounded seed from any ramp. It must also not grow between an early and a late frame.
        expect(penFinal).toBeLessThan(PENALTY_MIN + 1);
        expect(penFinal).toBeLessThanOrEqual(penAt5 + 1e-6);
    },
);

check(
    "a DYNAMIC box on the same wall still ramps — the gate is specific to the all-static contact",
    { claim: "dynamic contacts retain their penalty ramp", size: "integration", budget: 20000 },
    () => {
        // The fix must not silence a real resting contact. A dynamic box settling on a static ground has a
        // dynamic body, so its dual ramp is untouched: the penalty climbs well off the seed to hold mg
        // (the Phase-2 behavior). This pins that the gate keys on "no dynamic body", not "is static
        // involved" — a dyn–static contact (one dynamic) keeps ramping.
        const s = makeSolver(
            [
                body([100, 1, 100], 0, 0.5, [0, 0, 0]), // static ground
                body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [0, 1.5, 0]), // dynamic box
            ],
            { layer: "warmstart", gravity: -10 },
        );
        for (let f = 0; f < 300; f++) step(s);
        const pen = maxNormalPenalty(s);
        console.log(
            `[oracle/§6.4] dyn-vs-static normal penalty ${pen.toExponential(2)} (ramps to hold mg)`,
        );
        expect(pen).toBeGreaterThan(PENALTY_MIN + 1); // a real contact ramps
        expect(length(s.bodies[1].velLin)).toBeLessThan(1e-2); // and rests
    },
);
