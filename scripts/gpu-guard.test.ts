import { expect, test } from "bun:test";
import { resolve } from "node:path";
import {
  boundedQuery,
  gpuFields,
  guard,
  journalRows,
  parseOptions,
  type GuardIO,
  type Options,
  type QueryResult,
} from "./gpu-guard";
import { parseGPUCSV } from "./gpu-guard-state";

type ExitStatus = { code: number | null; signal: string | null };
type QueryPlan = { code?: number | null; stdout?: string; stderr?: string; error?: Error };
type JournalPlan = QueryPlan & { afterCursor?: string | null };

const BOOT_ID = "aaaa-bbbb";
const OTHER_BOOT_ID = "cccc-dddd";
const JOURNAL_BOOT_ID = "aaaabbbb";
const CURSOR = "safeboot123";
const GPU_UUID = "GPU-aaaa-bbbb";
const GPU_ROW = `${GPU_UUID},00000000:00:10.0,610.57.04,50,0,10,12288,20,210,405`;
const EXPECTED_GPU = parseGPUCSV(GPU_ROW)[0]!;

function deferred<T>() {
  let resolvePromise!: (value: T) => void;
  const promise = new Promise<T>(resolve => {
    resolvePromise = resolve;
  });
  return { promise, resolve: resolvePromise };
}

function options(overrides: Partial<Options> = {}): Options {
  return {
    out: resolve("test-output"),
    seconds: 1,
    intervalMs: 1000,
    queryMs: 250,
    maxTemperatureC: 83,
    runAuthorized: false,
    command: [],
    ...overrides,
  };
}

function queryOk(stdout = "", stderr = ""): QueryPlan {
  return { code: 0, stdout, stderr };
}

function journalLine(cursor: string, message: string, bootId = JOURNAL_BOOT_ID, timestamp = "1") {
  return JSON.stringify({
    __CURSOR: cursor,
    MESSAGE: message,
    __REALTIME_TIMESTAMP: timestamp,
    _BOOT_ID: bootId,
  });
}

function journalPlans(...rows: string[]): JournalPlan[] {
  const initial = rows.length ? `${rows.join("\n")}\n` : "";
  return [
    { afterCursor: null, ...queryOk(initial) },
    { afterCursor: CURSOR, ...queryOk("") },
    { afterCursor: CURSOR, ...queryOk("") },
    { afterCursor: CURSOR, ...queryOk("") },
    { afterCursor: CURSOR, ...queryOk("") },
  ];
}

function gpuRow(overrides: Partial<{
  uuid: string;
  pciBusId: string;
  driverVersion: string;
  temperatureC: number;
  utilizationPercent: number;
  memoryUsedMiB: number;
  memoryTotalMiB: number;
  powerW: number;
  smClockMHz: number;
  memoryClockMHz: number;
}> = {}) {
  return [
    overrides.uuid ?? GPU_UUID,
    overrides.pciBusId ?? "00000000:00:10.0",
    overrides.driverVersion ?? "610.57.04",
    String(overrides.temperatureC ?? 50),
    String(overrides.utilizationPercent ?? 0),
    String(overrides.memoryUsedMiB ?? 10),
    String(overrides.memoryTotalMiB ?? 12288),
    String(overrides.powerW ?? 20),
    String(overrides.smClockMHz ?? 210),
    String(overrides.memoryClockMHz ?? 405),
  ].join(",");
}

function computeRow(pid: number, processName = "python-worker", uuid = GPU_UUID) {
  return `${uuid},${pid},${processName}`;
}

class FakeIO implements GuardIO {
  readonly events: Record<string, unknown>[] = [];
  readonly queryCalls: Array<{ command: string[]; timeoutMs: number }> = [];
  readonly startedCommands: string[][] = [];
  readonly sleepCalls: number[] = [];
  readonly pgids = new Map<number, number>();
  readonly child = deferred<ExitStatus>();

  time = 0;
  stopCount = 0;
  bootIds: string[];
  journals: JournalPlan[];
  gpus: QueryPlan[];
  computes: QueryPlan[];
  childPid = 123;
  onSleep?: (ms: number, io: FakeIO) => void;

  constructor({
    bootIds = [],
    journals = journalPlans(journalLine(CURSOR, "kernel: boot ok")),
    gpus = [
      queryOk(gpuRow()),
      queryOk(gpuRow({ utilizationPercent: 10, memoryUsedMiB: 20 })),
      queryOk(gpuRow({ temperatureC: 49, utilizationPercent: 5, memoryUsedMiB: 11 })),
    ],
    computes = [queryOk(""), queryOk(""), queryOk("")],
  }: {
    bootIds?: string[];
    journals?: JournalPlan[];
    gpus?: QueryPlan[];
    computes?: QueryPlan[];
  } = {}) {
    this.bootIds = [...bootIds];
    this.journals = [...journals];
    this.gpus = [...gpus];
    this.computes = [...computes];
  }

  resolveChild(exit: ExitStatus = { code: 0, signal: null }) {
    this.child.resolve(exit);
  }

  async query(command: string[], timeoutMs: number): Promise<QueryResult> {
    this.queryCalls.push({ command: [...command], timeoutMs });
    const isGPU = command[0] === "/usr/bin/nvidia-smi" && command[1] === `--query-gpu=${gpuFields}`;
    const isCompute =
      command[0] === "/usr/bin/nvidia-smi" &&
      command[1] === "--query-compute-apps=gpu_uuid,pid,process_name";
    const isJournal = command[0] === "/usr/bin/journalctl";

    let plan: QueryPlan | undefined;
    if (isJournal) {
      const next = this.journals.shift();
      if (!next) throw new Error(`unexpected journal query: ${command.join(" ")}`);
      const afterCursor = command.find(arg => arg.startsWith("--after-cursor="));
      const actualAfter = afterCursor ? afterCursor.slice("--after-cursor=".length) : null;
      if ((next.afterCursor ?? null) !== actualAfter) {
        throw new Error(`expected journal after-cursor ${String(next.afterCursor)} but got ${String(actualAfter)}`);
      }
      plan = next;
    } else if (isGPU) {
      plan = this.gpus.shift();
      if (!plan) throw new Error(`unexpected gpu query: ${command.join(" ")}`);
    } else if (isCompute) {
      plan = this.computes.shift();
      if (!plan) throw new Error(`unexpected compute query: ${command.join(" ")}`);
    } else {
      throw new Error(`unexpected query command: ${command.join(" ")}`);
    }

    if (plan.error) throw plan.error;
    return { code: plan.code ?? 0, stdout: plan.stdout ?? "", stderr: plan.stderr ?? "" };
  }

  bootId(): string {
    return this.bootIds.length ? this.bootIds.shift()! : BOOT_ID;
  }

  now(): number {
    return this.time;
  }

  sleep(ms: number): Promise<void> {
    this.time += ms;
    this.sleepCalls.push(ms);
    this.onSleep?.(ms, this);
    return new Promise(resolveSleep => queueMicrotask(resolveSleep));
  }

  processGroup(pid: number): number {
    return this.pgids.get(pid) ?? pid;
  }

  record(event: Record<string, unknown>): void {
    this.events.push(event);
  }

  start(command: string[]) {
    this.startedCommands.push([...command]);
    return {
      pid: this.childPid,
      ended: this.child.promise,
      stop: async () => {
        this.stopCount++;
      },
    };
  }
}

function eventTypes(io: FakeIO) {
  return io.events.map(event => String(event.type));
}

function commandCount(io: FakeIO, predicate: (command: string[]) => boolean) {
  return io.queryCalls.filter(call => predicate(call.command)).length;
}

test("parseOptions parses defaults and authorized workload commands", () => {
  const parsed = parseOptions([
    "--out",
    "logs/run-a",
    "--seconds",
    "12",
    "--interval-ms",
    "500",
    "--query-ms",
    "1500",
    "--max-temperature",
    "81",
    "--uuid",
    GPU_UUID,
    "--run-authorized",
    "--",
    process.execPath,
    "-e",
    "console.log('ok')",
  ]);

  expect(parsed).toEqual({
    out: resolve("logs/run-a"),
    seconds: 12,
    intervalMs: 500,
    queryMs: 1500,
    maxTemperatureC: 81,
    uuid: GPU_UUID,
    runAuthorized: true,
    command: [process.execPath, "-e", "console.log('ok')"],
  });
  expect(parseOptions(["--out", "logs/defaults"])).toEqual({
    out: resolve("logs/defaults"),
    seconds: 30,
    intervalMs: 1000,
    queryMs: 2000,
    maxTemperatureC: 83,
    runAuthorized: false,
    command: [],
  });
});

test("parseOptions rejects invalid combinations and values", () => {
  for (const [args, pattern] of [
    [["--seconds", "1"], /--out NEW_DIR is required/],
    [["--out", "x", "--run-authorized"], /workload requires both --run-authorized and -- command/],
    [["--out", "x", "--", "echo", "hi"], /workload requires both --run-authorized and -- command/],
    [["--out", "x", "--seconds", "0"], /outside 1\.\.3600/],
    [["--out", "x", "--interval-ms", "249"], /outside 250\.\.5000/],
    [["--out", "x", "--query-ms", "99"], /outside 100\.\.5000/],
    [["--out", "x", "--max-temperature", "39"], /outside 40\.\.83/],
    [["--out", "x", "--uuid", "bad-uuid"], /invalid GPU UUID/],
    [["--out", "x", "--bogus"], /unknown option --bogus/],
    [["--out", "x", "--seconds", "-1"], /expected an unsigned integer/],
    [["--out", "x", "--uuid"], /missing value for --uuid/],
  ] as const) {
    expect(() => parseOptions(args as string[])).toThrow(pattern);
  }
});

test("journalRows parses matching boot rows and rejects malformed input", () => {
  expect(
    journalRows(
      [
        journalLine("cur-1", "kernel: boot ok", JOURNAL_BOOT_ID, "100"),
        journalLine("cur-2", "nvrm: loading", JOURNAL_BOOT_ID, "200"),
      ].join("\n"),
      BOOT_ID,
    ),
  ).toEqual([
    { cursor: "cur-1", message: "kernel: boot ok", timestamp: "100" },
    { cursor: "cur-2", message: "nvrm: loading", timestamp: "200" },
  ]);

  for (const text of [
    journalLine("cur-1", "kernel: boot ok", "differentboot"),
    JSON.stringify({ __CURSOR: "cur-1", MESSAGE: 5, __REALTIME_TIMESTAMP: "100", _BOOT_ID: JOURNAL_BOOT_ID }),
    "not-json",
  ]) {
    expect(() => journalRows(text, BOOT_ID)).toThrow();
  }
});

test("guard monitor mode samples deterministically, keeps original baseline, and never starts workload", async () => {
  const io = new FakeIO({
    gpus: [
      queryOk(gpuRow({ utilizationPercent: 0, memoryUsedMiB: 10 })),
      queryOk(gpuRow({ temperatureC: 70, utilizationPercent: 95, memoryUsedMiB: 4096 })),
      queryOk(gpuRow({ temperatureC: 45, utilizationPercent: 1, memoryUsedMiB: 12 })),
    ],
  });

  const result = await guard(options(), io);

  expect(result.ok).toBeTrue();
  expect(result.mode).toBe("monitor");
  expect(result.samples).toBe(3);
  expect(result.baseline).toEqual({ bootId: BOOT_ID, gpu: EXPECTED_GPU });
  expect(result.elapsedMs).toBe(1000);
  expect(io.startedCommands).toEqual([]);
  expect(io.stopCount).toBe(0);
  expect(io.sleepCalls).toEqual([1000]);
  expect(eventTypes(io)).toEqual([
    "start",
    "journal",
    "sample",
    "compute-processes",
    "journal",
    "sample",
    "compute-processes",
    "journal",
    "journal",
    "sample",
    "compute-processes",
    "journal",
    "result",
  ]);
  expect(io.queryCalls[0]?.command).toEqual([
    "/usr/bin/journalctl",
    "-k",
    "--boot=0",
    "--no-pager",
    "--output=json",
    "--quiet",
  ]);
  expect(commandCount(io, command => command[0] === "/usr/bin/journalctl")).toBe(5);
  expect(commandCount(io, command => command[0] === "/usr/bin/nvidia-smi" && command[1] === `--query-gpu=${gpuFields}`)).toBe(3);
  expect(commandCount(io, command => command[0] === "/usr/bin/nvidia-smi" && command[1] === "--query-compute-apps=gpu_uuid,pid,process_name")).toBe(3);
});

test("guard fails closed on boot id change", async () => {
  const io = new FakeIO({
    bootIds: [BOOT_ID, BOOT_ID, BOOT_ID, OTHER_BOOT_ID],
    computes: [queryOk("")],
  });

  const result = await guard(options(), io);

  expect(result.ok).toBeFalse();
  expect(String(result.fault)).toContain("BootIdChanged");
  expect(result.samples).toBe(2);
  expect(io.startedCommands).toEqual([]);
});

test("guard fails closed when the observed GPU UUID changes or disappears", async () => {
  const io = new FakeIO({
    gpus: [queryOk(gpuRow()), queryOk(gpuRow({ uuid: "GPU-dead-beef" }))],
    computes: [queryOk("")],
  });

  const result = await guard(options(), io);

  expect(result.ok).toBeFalse();
  expect(String(result.fault)).toContain("UUIDChanged");
  expect(result.samples).toBe(1);
});

test("guard fails closed on driver and temperature changes", async () => {
  for (const [label, nextRow, expectedFault] of [
    ["driver", gpuRow({ driverVersion: "999.0" }), "DriverVersionChanged"],
    ["temperature", gpuRow({ temperatureC: 83 }), "TemperatureExceeded"],
  ] as const) {
    const io = new FakeIO({
      gpus: [queryOk(gpuRow()), queryOk(nextRow)],
      computes: [queryOk("")],
    });

    const result = await guard(options(), io);

    expect(result.ok, label).toBeFalse();
    expect(String(result.fault), label).toContain(expectedFault);
    expect(result.samples, label).toBe(2);
  }
});

test("guard surfaces query failures and timeout-style rejections", async () => {
  {
    const io = new FakeIO({
      gpus: [queryOk(gpuRow()), { code: 9, stdout: "", stderr: "driver busy" }],
      computes: [queryOk("")],
    });
    const result = await guard(options(), io);
    expect(result.ok).toBeFalse();
    expect(String(result.fault)).toContain("query failed (9)");
  }

  {
    const io = new FakeIO({
      gpus: [queryOk(gpuRow()), { error: new Error("query timeout: nvidia-smi") }],
      computes: [queryOk("")],
    });
    const result = await guard(options(), io);
    expect(result.ok).toBeFalse();
    expect(String(result.fault)).toContain("query timeout: nvidia-smi");
  }
});

test("guard blocks prior-boot Xid before workload start", async () => {
  const io = new FakeIO({
    journals: journalPlans(journalLine(CURSOR, "NVRM: Xid (PCI:0000:00:10): 79, GPU has encountered a fatal error")),
  });

  const result = await guard(options({ runAuthorized: true, command: ["cmd", "--flag"] }), io);

  expect(result.ok).toBeFalse();
  expect(String(result.fault)).toContain("kernel fault: Xid");
  expect(io.startedCommands).toEqual([]);
  expect(io.stopCount).toBe(0);
});

test("guard blocks malformed journal rows before workload start", async () => {
  const io = new FakeIO({
    journals: [{ afterCursor: null, ...queryOk("not-json\n") }],
  });

  const result = await guard(options({ runAuthorized: true, command: ["cmd"] }), io);

  expect(result.ok).toBeFalse();
  expect(String(result.fault)).toContain("SyntaxError");
  expect(io.startedCommands).toEqual([]);
});

test("guard blocks competitor before workload start", async () => {
  const io = new FakeIO({
    computes: [queryOk(computeRow(777, "other-job"))],
  });

  const result = await guard(options({ runAuthorized: true, command: ["cmd"] }), io);

  expect(result.ok).toBeFalse();
  expect(String(result.fault)).toContain("competing GPU process 777");
  expect(io.startedCommands).toEqual([]);
  expect(io.stopCount).toBe(0);
});

test("guard stops workload on journal Xid after launch", async () => {
  const io = new FakeIO({
    journals: [
      { afterCursor: null, ...queryOk(`${journalLine(CURSOR, "kernel: boot ok")}\n`) },
      { afterCursor: CURSOR, ...queryOk("") },
      { afterCursor: CURSOR, ...queryOk(`${journalLine("cur-2", "NVRM: Xid (PCI:0000:00:10): 79, GPU has encountered a fatal error", JOURNAL_BOOT_ID, "2")}\n`) },
    ],
  });

  const result = await guard(options({ runAuthorized: true, command: ["train", "--one-step"] }), io);

  expect(result.ok).toBeFalse();
  expect(String(result.fault)).toContain("kernel fault: Xid");
  expect(io.startedCommands).toEqual([["train", "--one-step"]]);
  expect(io.stopCount).toBe(1);
  expect(eventTypes(io)).toContain("workload-start");
  expect(eventTypes(io)).toContain("workload-group-stopped");
});

test("guard stops workload when a different process group uses the GPU after launch", async () => {
  const io = new FakeIO({
    computes: [queryOk(""), queryOk(computeRow(777, "other-job"))],
  });
  io.pgids.set(777, 456);

  const result = await guard(options({ runAuthorized: true, command: ["train"] }), io);

  expect(result.ok).toBeFalse();
  expect(String(result.fault)).toContain("competing GPU process 777");
  expect(io.startedCommands).toEqual([["train"]]);
  expect(io.stopCount).toBe(1);
});

test("guard allows workload-owned GPU processes", async () => {
  const io = new FakeIO({
    computes: [queryOk(""), queryOk(computeRow(777, "owned-worker")), queryOk("")],
  });
  io.pgids.set(777, 123);
  io.onSleep = () => io.resolveChild({ code: 0, signal: null });

  const result = await guard(options({ runAuthorized: true, command: ["train"] }), io);

  expect(result.ok).toBeTrue();
  expect(result.mode).toBe("workload");
  expect(result.exit).toEqual({ code: 0, signal: null });
  expect(result.samples).toBe(3);
  expect(io.startedCommands).toEqual([["train"]]);
  expect(io.stopCount).toBe(1);
});

test("guard reports nonzero child exit after final checks", async () => {
  const io = new FakeIO();
  io.onSleep = () => io.resolveChild({ code: 2, signal: null });

  const result = await guard(options({ runAuthorized: true, command: ["train"] }), io);

  expect(result.ok).toBeFalse();
  expect(String(result.fault)).toContain("workload exit code=2 signal=null");
  expect(result.exit).toEqual({ code: 2, signal: null });
  expect(io.stopCount).toBe(1);
  expect(commandCount(io, command => command[0] === "/usr/bin/journalctl")).toBe(5);
  expect(commandCount(io, command => command[0] === "/usr/bin/nvidia-smi" && command[1] === `--query-gpu=${gpuFields}`)).toBe(3);
});

test("guard stops the workload group when the deadline expires", async () => {
  const io = new FakeIO({
    computes: [queryOk(""), queryOk("")],
  });

  const result = await guard(options({ runAuthorized: true, command: ["train"] }), io);

  expect(result.ok).toBeFalse();
  expect(String(result.fault)).toContain("workload deadline exceeded");
  expect(io.startedCommands).toEqual([["train"]]);
  expect(io.stopCount).toBe(1);
  expect(result.exit).toBeUndefined();
});

test("guard runs final health checks even when a fast child exits successfully", async () => {
  const io = new FakeIO({
    gpus: [queryOk(gpuRow()), queryOk(gpuRow({ utilizationPercent: 20 })), queryOk(gpuRow({ driverVersion: "611.00.00" }))],
  });
  io.onSleep = () => io.resolveChild({ code: 0, signal: null });

  const result = await guard(options({ runAuthorized: true, command: ["train"] }), io);

  expect(result.ok).toBeFalse();
  expect(String(result.fault)).toContain("DriverVersionChanged");
  expect(result.exit).toEqual({ code: 0, signal: null });
  expect(io.stopCount).toBe(1);
  expect(commandCount(io, command => command[0] === "/usr/bin/nvidia-smi" && command[1] === `--query-gpu=${gpuFields}`)).toBe(3);
});

test("guard runs final journal checks even when a fast child exits successfully", async () => {
  const io = new FakeIO({
    journals: [
      { afterCursor: null, ...queryOk(`${journalLine(CURSOR, "kernel: boot ok")}\n`) },
      { afterCursor: CURSOR, ...queryOk("") },
      { afterCursor: CURSOR, ...queryOk("") },
      { afterCursor: CURSOR, ...queryOk(`${journalLine("cur-2", "NVRM: Xid (PCI:0000:00:10): 79, GPU has encountered a fatal error", JOURNAL_BOOT_ID, "2")}\n`) },
    ],
  });
  io.onSleep = () => io.resolveChild({ code: 0, signal: null });

  const result = await guard(options({ runAuthorized: true, command: ["train"] }), io);

  expect(result.ok).toBeFalse();
  expect(String(result.fault)).toContain("kernel fault: Xid");
  expect(result.exit).toEqual({ code: 0, signal: null });
  expect(io.stopCount).toBe(1);
  expect(commandCount(io, command => command[0] === "/usr/bin/journalctl")).toBe(4);
});

test("boundedQuery captures stdout stderr and exit code", async () => {
  const result = await boundedQuery(
    [process.execPath, "-e", "process.stdout.write('out'); process.stderr.write('err'); process.exit(7);"] ,
    1000,
  );

  expect(result).toEqual({ code: 7, stdout: "out", stderr: "err" });
});

test("boundedQuery rejects timeouts within a bounded wall time", async () => {
  const started = Date.now();
  let error: unknown;
  try {
    await boundedQuery([process.execPath, "-e", "setInterval(() => {}, 100)"], 100);
  } catch (e) {
    error = e;
  }

  expect(String(error)).toContain("query timeout");
  expect(Date.now() - started).toBeLessThan(2000);
});

test("boundedQuery rejects output beyond the configured limit", async () => {
  let error: unknown;
  try {
    await boundedQuery(
      [process.execPath, "-e", "process.stdout.write('x'.repeat(1024)); setTimeout(() => {}, 1000)"] ,
      1000,
      32,
    );
  } catch (e) {
    error = e;
  }

  expect(String(error)).toContain("query output limit");
});
