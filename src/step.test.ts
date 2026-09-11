import { expect } from "bun:test";
import { check } from "@dylanebert/shallot/harness/check";
import { Compute, precompile, precompileAll, requestGPU } from "@dylanebert/shallot/runtime";
import { PhysicsStep } from "./step";

// Bun's CPU test process does not provide WebGPU's numeric enums. The recording device below needs the
// same values that the public GPU API supplies when the production constructor creates its buffers.
if (typeof GPUBufferUsage === "undefined")
    Object.assign(globalThis, {
        GPUBufferUsage: {
            MAP_READ: 1,
            MAP_WRITE: 2,
            COPY_SRC: 4,
            COPY_DST: 8,
            INDEX: 16,
            VERTEX: 32,
            UNIFORM: 64,
            STORAGE: 128,
            INDIRECT: 256,
            QUERY_RESOLVE: 512,
        },
    });
if (typeof GPUShaderStage === "undefined")
    Object.assign(globalThis, { GPUShaderStage: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 } });

// These lifecycle checks use a recording GPU stub. They exercise the public construction and compose
// contract without dispatching a real device or treating generated WGSL text as product evidence.
const stub = (): GPUDevice =>
    ({
        features: new Set(["subgroups"]),
        limits: { maxStorageBufferBindingSize: 1 << 30 },
        queue: { writeBuffer() {} },
        createBuffer: (desc: GPUBufferDescriptor) => ({ ...desc, destroy() {} }),
        createBindGroupLayout: (desc: unknown) => desc,
        createBindGroup: (desc: unknown) => desc,
        createPipelineLayout: (desc: unknown) => desc,
        createShaderModule: (desc: unknown) => desc,
        createComputePipeline: (desc: unknown) => desc,
        createComputePipelineAsync: async (desc: unknown) => desc,
    }) as unknown as GPUDevice;

check(
    "PhysicsStep scopes labels for a second instance",
    { claim: "a second PhysicsStep receives unique precompile labels" },
    async () => {
        const saved = { ...Compute };
        try {
            const device = stub();
            await requestGPU(device);

            await PhysicsStep.create(device, 64, 64);
            expect(() => precompile("phys-aabb", () => true)).toThrow(/duplicate/);

            await PhysicsStep.create(device, 64, 64);
            expect(() => precompile("phys-2-aabb", () => true)).toThrow(/duplicate/);
            expect(() => precompile("phys-2-collide-box", () => true)).toThrow(/duplicate/);
            expect(() => precompile("phys-2-joint-dual", () => true)).toThrow(/duplicate/);
        } finally {
            Object.assign(Compute, saved);
        }
    },
);

check(
    "PhysicsStep compose uses its instance scope",
    { claim: "compose uses the precompile scope of its PhysicsStep instance" },
    async () => {
        const saved = { ...Compute };
        try {
            const device = stub();
            await requestGPU(device);
            const first = await PhysicsStep.create(device, 64, 64);
            const second = await PhysicsStep.create(device, 64, 64);

            // The stub buffer carries no schema for the indirect dispatch to read. Swallow that dispatch,
            // never a duplicate label: the collision this guards against surfaces at preparation.
            const composeOnce = async (step: PhysicsStep) => {
                const transforms = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE });
                const encoder = {
                    beginComputePass: () => ({ end() {} }),
                } as unknown as GPUCommandEncoder;
                await step.prepareCompose(transforms);
                try {
                    step.compose(encoder, transforms);
                } catch (err) {
                    if (String(err).includes("duplicate precompile")) throw err;
                }
            };

            await composeOnce(first);
            expect(() => precompile("phys-compose", () => true)).toThrow(/duplicate/);
            await composeOnce(second);
            expect(() => precompile("phys-2-compose", () => true)).toThrow(/duplicate/);
        } finally {
            Object.assign(Compute, saved);
        }
    },
);

check(
    "PhysicsStep compose preserves late validation failure",
    { claim: "compose refuses before late validation settles and preserves its failure" },
    async () => {
        const saved = { ...Compute };
        let rejectFence!: (error: Error) => void;
        const fence = new Promise<void>((_, reject) => {
            rejectFence = reject;
        });
        let lateFence: Promise<void> | undefined;
        try {
            const device = {
                ...stub(),
                queue: {
                    writeBuffer() {},
                },
                pushErrorScope() {},
                popErrorScope: async () => null,
            } as unknown as GPUDevice;
            await requestGPU(device);
            const bound = {
                $name() {
                    return this;
                },
                with() {
                    return this;
                },
                initAsync: () => lateFence ?? Promise.resolve(),
            };
            Object.assign(Compute, {
                root: {
                    createComputePipeline: () => bound,
                    createBindGroup: () => ({}),
                    unwrap: (value: unknown) => value,
                },
            });
            const step = await PhysicsStep.create(device, 64, 64);
            await precompileAll();
            lateFence = fence;

            const transforms = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE });
            const preparing = step.prepareCompose(transforms);
            const encoder = {
                beginComputePass: () => {
                    throw new Error("compose encoded before validation settled");
                },
            } as unknown as GPUCommandEncoder;
            expect(() => step.compose(encoder, transforms)).toThrow(
                "await prepareCompose(transforms)",
            );

            rejectFence(new Error("late compose fence failed"));
            const failure = await preparing.catch((error: unknown) => error);
            expect(failure).toMatchObject({
                name: "GpuDiagnosticError",
                label: "phys-compose",
            });
            expect(String(failure)).toContain("late compose fence failed");
            expect(() => step.compose(encoder, transforms)).toThrow("late compose fence failed");
        } finally {
            Object.assign(Compute, saved);
        }
    },
);

check(
    "PhysicsStep creation propagates late precompile failure",
    { claim: "public PhysicsStep creation awaits and propagates late precompile failure" },
    async () => {
        const saved = { ...Compute };
        let physicsAllocated = false;
        let physicsFences = 0;
        let rejectFirst!: (error: Error) => void;
        let releaseRest!: () => void;
        const firstFence = new Promise<void>((_, reject) => {
            rejectFirst = reject;
        });
        const restFence = new Promise<void>((resolve) => {
            releaseRest = resolve;
        });
        try {
            const base = stub();
            const createBuffer = base.createBuffer.bind(base);
            const device = {
                ...base,
                queue: {
                    writeBuffer() {},
                },
                pushErrorScope() {},
                popErrorScope: async () => null,
                createBuffer(descriptor: GPUBufferDescriptor) {
                    if (descriptor.label === "phys-bodies") physicsAllocated = true;
                    return createBuffer(descriptor);
                },
            } as unknown as GPUDevice;
            await requestGPU(device);
            await precompileAll();
            const bound = {
                $name() {
                    return this;
                },
                with() {
                    return this;
                },
                initAsync() {
                    if (!physicsAllocated) return Promise.resolve();
                    physicsFences++;
                    return physicsFences === 1 ? firstFence : restFence;
                },
            };
            Object.assign(Compute, {
                root: {
                    createComputePipeline: () => bound,
                    createBindGroup: () => ({}),
                    unwrap: (value: unknown) => value,
                },
            });

            const creation = PhysicsStep.create(device, 64, 64);
            while (!physicsAllocated) await Promise.resolve();
            let returned = false;
            void creation.then(
                () => {
                    returned = true;
                },
                () => {
                    returned = true;
                },
            );
            await Promise.resolve();
            await Promise.resolve();
            expect(returned).toBe(false);

            rejectFirst(new Error("late physics fence failed"));
            while (physicsFences < 2) await Promise.resolve();
            await Promise.resolve();
            expect(returned).toBe(false);

            releaseRest();
            await expect(creation).rejects.toMatchObject({
                name: "GpuDiagnosticError",
                label: "phys-aabb",
                message: expect.stringContaining("late physics fence failed"),
            });
        } finally {
            Object.assign(Compute, saved);
        }
    },
);
