import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";

const CANDIDATE_SHA = "0664218f465224397b80aeb604b51178ac71cfb2";
const CANDIDATE_TAG = `dylanebert-shallot-${CANDIDATE_SHA.slice(0, 7)}`;

interface CommandResult {
    code: number;
    stdout: string;
    stderr: string;
}

function run(command: string[], cwd: string): CommandResult {
    const result = Bun.spawnSync(command, {
        cwd,
        stdout: "pipe",
        stderr: "pipe",
    });
    return {
        code: result.exitCode ?? 1,
        stdout: result.stdout.toString(),
        stderr: result.stderr.toString(),
    };
}

function runChecked(command: string[], cwd: string): CommandResult {
    const result = run(command, cwd);
    if (result.code !== 0) {
        throw new Error(
            `${command.join(" ")} failed with ${result.code}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`,
        );
    }
    return result;
}

function jsonFile(path: string): Record<string, unknown> {
    return JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
}

check(
    "packed AVBD identity and public boundary work in a scratch consumer",
    {
        claim: "packed AVBD identity and public boundary work in a scratch consumer",
        size: "integration",
        subject: [
            "package.json",
            "bun.lock",
            "AGENTS.md",
            "README.md",
            "shallot.json",
            "tests/package-preflight.test.ts",
            ".github/workflows/test-surface.yml",
        ],
        budget: 20000,
    },
    () => {
        const root = process.cwd();
        const manifest = jsonFile(resolve(root, "package.json"));
        const lock = readFileSync(resolve(root, "bun.lock"), "utf8");
        const devDependencies = manifest.devDependencies as Record<string, unknown>;
        if (
            devDependencies["@dylanebert/shallot"] !== `github:dylanebert/shallot#${CANDIDATE_SHA}`
        ) {
            throw new Error("package.json does not carry the qualified full-SHA Shallot candidate");
        }
        if (
            !lock.includes(
                `@dylanebert/shallot@github:dylanebert/shallot#${CANDIDATE_SHA.slice(0, 7)}`,
            )
        ) {
            throw new Error("bun.lock does not carry the qualified Shallot candidate");
        }

        const packDir = mkdtempSync("/tmp/shallot-avbd-pack-");
        const scratchDir = mkdtempSync("/tmp/shallot-avbd-consumer-");
        try {
            const packed = runChecked(
                [process.execPath, "pm", "pack", "--destination", packDir],
                root,
            );
            const artifactName = packed.stdout
                .split("\n")
                .map((line) => line.trim())
                .find((line) => line.endsWith(".tgz"));
            if (artifactName === undefined)
                throw new Error("bun pm pack did not report an artifact");
            const artifact = resolve(packDir, artifactName);
            const archive = runChecked(["tar", "-tzf", artifact], root).stdout;
            for (const entry of [
                "package/package.json",
                "package/src/core.ts",
                "package/src/index.ts",
            ]) {
                if (!archive.split("\n").includes(entry))
                    throw new Error(`package preflight omitted ${entry}`);
            }
            const packedManifest = JSON.parse(
                runChecked(["tar", "-xOf", artifact, "package/package.json"], root).stdout,
            ) as Record<string, unknown>;
            if (
                packedManifest.name !== manifest.name ||
                packedManifest.version !== manifest.version
            ) {
                throw new Error("package preflight identity differs from the project manifest");
            }
            const packedExports = packedManifest.exports as Record<string, unknown>;
            if (packedExports["./core"] !== "./src/core.ts")
                throw new Error("package preflight omitted the public /core export");

            writeFileSync(
                resolve(scratchDir, "package.json"),
                `${JSON.stringify(
                    {
                        name: "shallot-avbd-scratch-consumer",
                        private: true,
                        type: "module",
                        packageManager: "bun@1.4.2",
                        dependencies: {
                            "@dylanebert/shallot-avbd-physics": `file:${artifact}`,
                            "@dylanebert/shallot": `github:dylanebert/shallot#${CANDIDATE_SHA}`,
                            typegpu: "~0.12.5",
                        },
                    },
                    null,
                    2,
                )}\n`,
            );
            runChecked([process.execPath, "install"], scratchDir);
            rmSync(resolve(scratchDir, "node_modules"), { recursive: true, force: true });
            runChecked([process.execPath, "install", "--frozen-lockfile"], scratchDir);

            const probe = `
import { createRequire } from "node:module";
import { existsSync, realpathSync } from "node:fs";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { AvbdPlugin } from "@dylanebert/shallot-avbd-physics";
import { Avbd as CoreAvbd, MAX_CONTACTS } from "@dylanebert/shallot-avbd-physics/core";
import avbd from "@dylanebert/shallot-avbd-physics/package.json";
import shallot from "@dylanebert/shallot/package.json";
import typegpu from "typegpu/package.json";

if (typeof AvbdPlugin !== "object" || typeof CoreAvbd !== "object" || MAX_CONTACTS !== 4)
  throw new Error("public AVBD exports are not usable");
if (avbd.name !== "@dylanebert/shallot-avbd-physics")
  throw new Error("installed AVBD package has the wrong identity");
if (shallot.version !== "0.10.0") throw new Error("installed Shallot has the wrong version");
const shallotPath = realpathSync(resolve(import.meta.dir, "node_modules/@dylanebert/shallot"));
const typegpuPath = realpathSync(resolve(import.meta.dir, "node_modules/typegpu"));
const resolvedTypegpu = realpathSync(createRequire(resolve(shallotPath, "package.json")).resolve("typegpu/package.json"));
if (resolvedTypegpu !== resolve(typegpuPath, "package.json"))
  throw new Error("Shallot resolved a second TypeGPU instance");
if (typegpu.version !== "0.12.5") throw new Error("installed TypeGPU has the wrong version");
const tag = await Bun.file(resolve(shallotPath, ".bun-tag")).text();
if (tag.trim() !== "${CANDIDATE_TAG}") throw new Error("installed Shallot is not the qualified candidate");
const bin = resolve(import.meta.dir, "node_modules/.bin/shallot");
if (!existsSync(bin)) throw new Error("installed Shallot bin is missing");
const result = spawnSync(process.execPath, [bin, "list"], { cwd: import.meta.dir, encoding: "utf8" });
if (result.status !== 0) throw new Error("installed Shallot bin failed: " + result.stderr);
`;
            writeFileSync(resolve(scratchDir, "probe.ts"), probe);
            runChecked([process.execPath, "probe.ts"], scratchDir);
        } finally {
            rmSync(packDir, { recursive: true, force: true });
            rmSync(scratchDir, { recursive: true, force: true });
        }
    },
);
