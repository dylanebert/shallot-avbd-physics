import { expect } from "bun:test";
import { check } from "@dylanebert/shallot/harness/check";
import { diffStamps } from "./recycle";

// diffStamps is the CPU pre-pack stamp diff the AVBD PackSystem runs to catch a same-update destroy+create
// realias. The seed path itself is GPU-only (the pack shader), so this bounded truth table pins the
// bookkeeping that decides which eids the plugin reseeds.
check(
    "diffStamps stamp-diff truth table",
    { claim: "stamp-diff reseeds exactly the realiased eids" },
    () => {
        const cases = [
            {
                name: "first-seen eid is a fresh spawn",
                seen: new Map<number, number>(),
                stamps: new Map([[5, 1]]),
                eids: [5],
                expected: [],
                recorded: [5, 1] as const,
            },
            {
                name: "unchanged stamp is settled",
                seen: new Map([[5, 1]]),
                stamps: new Map([[5, 1]]),
                eids: [5],
                expected: [],
                recorded: null,
            },
            {
                name: "changed stamp is a realias",
                seen: new Map([[5, 1]]),
                stamps: new Map([[5, 2]]),
                eids: [5],
                expected: [5],
                recorded: [5, 2] as const,
            },
            {
                name: "mixed set only reseeds recycled eid",
                seen: new Map([
                    [3, 7],
                    [4, 7],
                ]),
                stamps: new Map([
                    [2, 1],
                    [3, 7],
                    [4, 8],
                ]),
                eids: [2, 3, 4],
                expected: [4],
                recorded: [2, 1] as const,
            },
        ];

        for (const row of cases) {
            const actual = diffStamps(row.eids, (eid) => row.stamps.get(eid) ?? 0, row.seen);
            expect(actual, row.name).toEqual(row.expected);
            if (row.recorded !== null)
                expect(row.seen.get(row.recorded[0]), row.name).toBe(row.recorded[1]);
        }

        const changed = new Map([[5, 2]]);
        const stamps = new Map([[5, 2]]);
        expect(diffStamps([5], (eid) => stamps.get(eid) ?? 0, changed)).toEqual([]);
    },
);
