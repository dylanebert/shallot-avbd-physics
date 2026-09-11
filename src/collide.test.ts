import { expect } from "bun:test";
import { ShapeKind } from "@dylanebert/shallot";
import { check } from "@dylanebert/shallot/harness/check";
import { Hulls } from "@dylanebert/shallot/physics";
import * as d from "typegpu/data";
import { boxHull, collideHull as collideHullOracle, coneHull } from "../tests/hull";
import { collideHull, polyMake, withCpuHullData } from "./collide";
import { packHulls } from "./hull";

check(
    "packed-hull TGSL graph matches the f64 oracle",
    { claim: "packed-hull TGSL result matches the independent f64 cone/cube oracle" },
    () => {
        const cone = coneHull(0.4, 1, 8);
        const cube = boxHull([1, 1, 1]);
        const coneId = Hulls.register({ name: "__sat_cpu_cone8__", ...cone });
        const cubeId = Hulls.register({ name: "__sat_cpu_cube__", ...cube });
        const quat = d.vec4f(0, 0, 0, 1);
        const oracle = collideHullOracle(
            cone,
            [0, 0.8, 0],
            [0, 0, 0, 1],
            cube,
            [0, 0, 0],
            [0, 0, 0, 1],
        );
        const got = withCpuHullData(packHulls(), () =>
            collideHull(
                polyMake(ShapeKind.Hull, d.vec3f(0, 0.8, 0), quat, d.vec3f(), coneId),
                polyMake(ShapeKind.Hull, d.vec3f(), quat, d.vec3f(), cubeId),
                d.vec3f(),
            ),
        );

        expect(got.count).toBe(oracle.contacts.length);
        const byFeature = new Map<number, number>();
        for (let i = 0; i < got.count; i++) byFeature.set(got.feat[i] >>> 0, i);
        expect([...byFeature.keys()].sort((a, b) => a - b)).toEqual(
            oracle.contacts.map((contact) => contact.feature >>> 0).sort((a, b) => a - b),
        );
        for (const contact of oracle.contacts) {
            const i = byFeature.get(contact.feature >>> 0);
            expect(i).toBeDefined();
            for (let lane = 0; lane < 3; lane++) {
                expect(got.rA[i as number][lane]).toBeCloseTo(contact.rA[lane], 6);
                expect(got.rB[i as number][lane]).toBeCloseTo(contact.rB[lane], 6);
            }
        }
    },
);

check(
    "packed-hull CPU data scope rejects asynchronous callbacks",
    { claim: "withCpuHullData rejects an asynchronous callback" },
    () => {
        expect(() => withCpuHullData(new Uint32Array(), () => Promise.resolve())).toThrow(
            "withCpuHullData callback must be synchronous",
        );
    },
);
