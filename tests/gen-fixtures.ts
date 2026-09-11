// Regenerates the dense AVBD parity fixtures by building and running the reference
// C++ harness in the explicitly selected avbd-demo3d checkout. Emits two param sets
// the oracle and GPU validate against:
//   canonical/ — 10 iters, betaLin 1e4, alpha 0.99 (reference defaults) — correctness
//   budget/    — 4 iters, betaLin 1e5                                    — perf tuning
//
// Output lands in `tests/fixtures/{canonical,budget}/dense-<scene>.json`, which is
// gitignored (regenerable). Run before the fixture-parity gates in oracle.test.ts.
//
// Usage: AVBD_DEMO3D_DIR=/absolute/path bun run tests/gen-fixtures.ts [frames]
//        (default 600). Requires: g++ on PATH.

import { spawnSync } from "child_process";
import { existsSync, mkdirSync, statSync } from "fs";
import { isAbsolute, resolve } from "path";

const frames = Number(Bun.argv[2] ?? 600);
const refDir = Bun.env.AVBD_DEMO3D_DIR;
const fixtureRoot = resolve(import.meta.dir, "fixtures");

if (!refDir) {
    console.error("AVBD_DEMO3D_DIR is required");
    process.exit(1);
}
if (!isAbsolute(refDir)) {
    console.error(`AVBD_DEMO3D_DIR must be absolute: ${refDir}`);
    process.exit(1);
}
if (!existsSync(refDir)) {
    console.error(`reference dir missing: ${refDir}`);
    process.exit(1);
}
if (!statSync(refDir).isDirectory()) {
    console.error(`reference path is not a directory: ${refDir}`);
    process.exit(1);
}

const sources = [
    "source/solver.cpp",
    "source/rigid.cpp",
    "source/force.cpp",
    "source/manifold.cpp",
    "source/collide.cpp",
    "source/joint.cpp",
    "source/spring.cpp",
    "harness-dense.cpp",
];
const binary = resolve(fixtureRoot, ".bin", "harness-dense");
mkdirSync(resolve(fixtureRoot, ".bin"), { recursive: true });

console.log("[gen-fixtures] building harness-dense");
const build = spawnSync("g++", ["-std=c++17", "-O2", ...sources, "-I", "source", "-o", binary], {
    cwd: refDir,
    stdio: "inherit",
});
if (build.status !== 0) process.exit(build.status ?? 1);

// [outSubdir, iterations, betaLin]; "" args fall back to the solver defaults.
const sets: [string, string[]][] = [
    ["canonical", []],
    ["budget", ["4", "100000"]],
];

for (const [name, params] of sets) {
    const outDir = resolve(fixtureRoot, name);
    mkdirSync(outDir, { recursive: true });
    console.log(`[gen-fixtures] ${name}: ${frames} frames -> ${outDir}`);
    const run = spawnSync(binary, [String(frames), outDir, ...params], { stdio: "inherit" });
    if (run.status !== 0) process.exit(run.status ?? 1);
}
