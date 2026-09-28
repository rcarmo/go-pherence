import { expect, test } from "bun:test";
import {
  evaluateHealth,
  faultFromJournal,
  parseComputeCSV,
  parseGPUCSV,
  type GPUBaseline,
  type GPUSample,
} from "./gpu-guard-state";

const baselineGpu: GPUSample = {
  uuid: "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
  pciBusId: "00000000:65:00.0",
  driverVersion: "550.54.14",
  temperatureC: 70,
  utilizationPercent: 42,
  memoryUsedMiB: 1024,
  memoryTotalMiB: 24576,
  powerW: 210.5,
  smClockMHz: 1800,
  memoryClockMHz: 7000,
};

const baseline: GPUBaseline = {
  bootId: "boot-a",
  gpu: baselineGpu,
};

test("parseGPUCSV parses valid rows and unsupported optional values", () => {
  expect(parseGPUCSV([
    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, 00000000:65:00.0, 550.54.14, 72, 91, 1200, 24576, 215.5, 1800, 7000",
    "GPU-1111-2222, 0000:17:00.0, 550.54.14, 45, 0, 100, 4096, N/A, [Not Supported], N/A",
  ].join("\n"))).toEqual([
    {
      uuid: "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
      pciBusId: "00000000:65:00.0",
      driverVersion: "550.54.14",
      temperatureC: 72,
      utilizationPercent: 91,
      memoryUsedMiB: 1200,
      memoryTotalMiB: 24576,
      powerW: 215.5,
      smClockMHz: 1800,
      memoryClockMHz: 7000,
    },
    {
      uuid: "GPU-1111-2222",
      pciBusId: "0000:17:00.0",
      driverVersion: "550.54.14",
      temperatureC: 45,
      utilizationPercent: 0,
      memoryUsedMiB: 100,
      memoryTotalMiB: 4096,
      powerW: null,
      smClockMHz: null,
      memoryClockMHz: null,
    },
  ]);
});

test("parseGPUCSV returns empty list for empty text", () => {
  expect(parseGPUCSV("")).toEqual([]);
  expect(parseGPUCSV(" \n\r\n ")).toEqual([]);
});

test("parseGPUCSV fails closed on malformed rows", () => {
  for (const text of [
    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,00000000:65:00.0,550.54.14,72,91,1200,24576,215.5,1800",
    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,00000000:65:00.0,550.54.14,N/A,91,1200,24576,215.5,1800,7000",
    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,00000000:65:00.0,550.54.14,151,91,1200,24576,215.5,1800,7000",
    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,00000000:65:00.0,550.54.14,72,101,1200,24576,215.5,1800,7000",
    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,00000000:65:00.0,550.54.14,72,91,25000,24576,215.5,1800,7000",
    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,00000000:65:00.0,550.54.14,72,91,1200,0,215.5,1800,7000",
    "GPU-not-hex,00000000:65:00.0,550.54.14,72,91,1200,24576,215.5,1800,7000",
    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,bad-pci,550.54.14,72,91,1200,24576,215.5,1800,7000",
    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,00000000:65:00.0,550.54.14,NaN,91,1200,24576,215.5,1800,7000",
    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,00000000:65:00.0,550.54.14,72,91,1200,24576,1e309,1800,7000",
  ]) {
    expect(() => parseGPUCSV(text)).toThrow();
  }
});

test("parseComputeCSV parses process names with commas and empty text", () => {
  expect(parseComputeCSV("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, 4242, python, worker, --serve")).toEqual([
    { uuid: "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", pid: 4242, processName: "python, worker, --serve" },
  ]);
  expect(parseComputeCSV("\n \r\n")).toEqual([]);
});

test("parseComputeCSV rejects malformed pid uuid and missing process name", () => {
  for (const text of [
    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, not-a-pid, python",
    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, 12.5, python",
    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, 0, python",
    "bad-uuid, 12, python",
    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, 12",
    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, 12,   ",
  ]) {
    expect(() => parseComputeCSV(text)).toThrow();
  }
});

test("faultFromJournal flags conservative GPU faults but ignores ordinary boot messages", () => {
  expect(faultFromJournal("GPU recovery action changed from None to NodeReboot")).toBe("NodeReboot");
  expect(faultFromJournal("NVRM: GPU at PCI:0000:65:00: GPU has fallen off the bus.")).toBe("GPUFallenOffBus");
  expect(faultFromJournal("NVRM: Xid (PCI:0000:65:00): 79, GPU has encountered a fatal error")).toBe("Xid");
  expect(faultFromJournal("nvrm: fatal: unable to communicate with device")).toBe("NVRMFault");
  expect(faultFromJournal("NVRM: loading NVIDIA UNIX x86_64 Kernel Module  550.54.14")).toBeNull();
  expect(faultFromJournal("kernel: pci 0000:65:00.0: enabling device (0000 -> 0003)")).toBeNull();
});

test("evaluateHealth accepts matching healthy samples without mutating inputs", () => {
  const sample = Object.freeze({ ...baselineGpu, temperatureC: 82.9 });
  const currentBaseline = Object.freeze({
    bootId: "boot-a",
    gpu: Object.freeze({ ...baselineGpu }),
  });
  expect(evaluateHealth(sample, "boot-a", currentBaseline, 83)).toBeNull();
  expect(sample.temperatureC).toBe(82.9);
  expect(currentBaseline.gpu.driverVersion).toBe("550.54.14");
});

test("evaluateHealth fails closed on policy threshold and identity changes", () => {
  expect(evaluateHealth({ ...baselineGpu, temperatureC: 83 }, "boot-a", baseline, 83)).toBe("TemperatureExceeded");
  expect(evaluateHealth({ ...baselineGpu }, "boot-b", baseline, 83)).toBe("BootIdChanged");
  expect(evaluateHealth({ ...baselineGpu, uuid: "GPU-1111-2222" }, "boot-a", baseline, 83)).toBe("UUIDChanged");
  expect(evaluateHealth({ ...baselineGpu, pciBusId: "0000:17:00.0" }, "boot-a", baseline, 83)).toBe("PCIBusIdChanged");
  expect(evaluateHealth({ ...baselineGpu, driverVersion: "555.12" }, "boot-a", baseline, 83)).toBe("DriverVersionChanged");
});

test("evaluateHealth validates sample baseline boot id and policy inputs", () => {
  expect(evaluateHealth({ ...baselineGpu, memoryTotalMiB: 0 }, "boot-a", baseline, 83)).toBe("MalformedSample");
  expect(evaluateHealth(baselineGpu, "", baseline, 83)).toBe("MalformedBootId");
  expect(evaluateHealth(baselineGpu, "boot-a", { bootId: "boot-a", gpu: { ...baselineGpu, utilizationPercent: Number.NaN } }, 83)).toBe("MalformedBaseline");
  expect(evaluateHealth(baselineGpu, "boot-a", baseline, Number.NaN)).toBe("MalformedPolicy");
});
