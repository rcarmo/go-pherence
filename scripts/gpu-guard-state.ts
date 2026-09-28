export type GPUSample = {
  uuid: string;
  pciBusId: string;
  driverVersion: string;
  temperatureC: number;
  utilizationPercent: number;
  memoryUsedMiB: number;
  memoryTotalMiB: number;
  powerW: number | null;
  smClockMHz: number | null;
  memoryClockMHz: number | null;
};

export type GPUComputeProcess = {
  uuid: string;
  pid: number;
  processName: string;
};

export type GPUBaseline = {
  bootId: string;
  gpu: GPUSample;
};

export type GPUJournalFault = "NodeReboot" | "GPUFallenOffBus" | "Xid" | "NVRMFault";

export type GPUHealthFault =
  | "MalformedSample"
  | "MalformedBaseline"
  | "MalformedBootId"
  | "MalformedPolicy"
  | "BootIdChanged"
  | "UUIDChanged"
  | "PCIBusIdChanged"
  | "DriverVersionChanged"
  | "TemperatureExceeded";

const GPU_UUID_RE = /^GPU-[0-9A-Fa-f]+(?:-[0-9A-Fa-f]+)*$/;
const PCI_BUS_ID_RE = /^[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f]$/;
const DECIMAL_RE = /^(?:\d+(?:\.\d+)?|\.\d+)$/;
const PID_RE = /^\d+$/;
const OPTIONAL_UNAVAILABLE = new Set(["N/A", "[Not Supported]"]);

function assert(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim() !== "";
}

function parseStrictNumber(raw: string, field: string, lineNumber: number): number {
  const value = raw.trim();
  assert(DECIMAL_RE.test(value), `line ${lineNumber}: invalid ${field}`);
  const parsed = Number(value);
  assert(Number.isFinite(parsed), `line ${lineNumber}: invalid ${field}`);
  return parsed;
}

function parseOptionalNumber(raw: string, field: string, lineNumber: number): number | null {
  const value = raw.trim();
  if (OPTIONAL_UNAVAILABLE.has(value)) return null;
  return parseStrictNumber(value, field, lineNumber);
}

function validateRanges(sample: GPUSample, label: string): string | null {
  if (!GPU_UUID_RE.test(sample.uuid.trim())) return label;
  if (!PCI_BUS_ID_RE.test(sample.pciBusId.trim())) return label;
  if (sample.driverVersion.trim() === "") return label;
  if (!Number.isFinite(sample.temperatureC) || sample.temperatureC < 0 || sample.temperatureC > 150) return label;
  if (!Number.isFinite(sample.utilizationPercent) || sample.utilizationPercent < 0 || sample.utilizationPercent > 100) return label;
  if (!Number.isFinite(sample.memoryUsedMiB) || sample.memoryUsedMiB < 0) return label;
  if (!Number.isFinite(sample.memoryTotalMiB) || sample.memoryTotalMiB <= 0) return label;
  if (sample.memoryUsedMiB > sample.memoryTotalMiB) return label;
  for (const value of [sample.powerW, sample.smClockMHz, sample.memoryClockMHz]) {
    if (value !== null && (!Number.isFinite(value) || value < 0)) return label;
  }
  return null;
}

function sampleFault(value: unknown, label: "MalformedSample" | "MalformedBaseline"): "MalformedSample" | "MalformedBaseline" | null {
  if (!isRecord(value)) return label;
  if (!isNonEmptyString(value.uuid)) return label;
  if (!isNonEmptyString(value.pciBusId)) return label;
  if (!isNonEmptyString(value.driverVersion)) return label;
  if (typeof value.temperatureC !== "number") return label;
  if (typeof value.utilizationPercent !== "number") return label;
  if (typeof value.memoryUsedMiB !== "number") return label;
  if (typeof value.memoryTotalMiB !== "number") return label;
  if (!(typeof value.powerW === "number" || value.powerW === null)) return label;
  if (!(typeof value.smClockMHz === "number" || value.smClockMHz === null)) return label;
  if (!(typeof value.memoryClockMHz === "number" || value.memoryClockMHz === null)) return label;
  return validateRanges(value as GPUSample, label);
}

function baselineFault(value: unknown): "MalformedBaseline" | null {
  if (!isRecord(value)) return "MalformedBaseline";
  if (!isNonEmptyString(value.bootId)) return "MalformedBaseline";
  return sampleFault(value.gpu, "MalformedBaseline");
}

export function parseGPUCSV(text: string): GPUSample[] {
  const trimmed = text.trim();
  if (trimmed === "") return [];
  return trimmed.split(/\r?\n/).filter(line => line.trim() !== "").map((line, index) => {
    const lineNumber = index + 1;
    const fields = line.split(",");
    assert(fields.length === 10, `line ${lineNumber}: expected 10 fields, got ${fields.length}`);
    const sample: GPUSample = {
      uuid: fields[0].trim(),
      pciBusId: fields[1].trim(),
      driverVersion: fields[2].trim(),
      temperatureC: parseStrictNumber(fields[3], "temperatureC", lineNumber),
      utilizationPercent: parseStrictNumber(fields[4], "utilizationPercent", lineNumber),
      memoryUsedMiB: parseStrictNumber(fields[5], "memoryUsedMiB", lineNumber),
      memoryTotalMiB: parseStrictNumber(fields[6], "memoryTotalMiB", lineNumber),
      powerW: parseOptionalNumber(fields[7], "powerW", lineNumber),
      smClockMHz: parseOptionalNumber(fields[8], "smClockMHz", lineNumber),
      memoryClockMHz: parseOptionalNumber(fields[9], "memoryClockMHz", lineNumber),
    };
    const fault = validateRanges(sample, "MalformedSample");
    assert(fault === null, `line ${lineNumber}: invalid GPU sample`);
    return sample;
  });
}

export function parseComputeCSV(text: string): GPUComputeProcess[] {
  const trimmed = text.trim();
  if (trimmed === "") return [];
  return trimmed.split(/\r?\n/).filter(line => line.trim() !== "").map((line, index) => {
    const lineNumber = index + 1;
    const fields = line.split(",");
    assert(fields.length >= 3, `line ${lineNumber}: expected at least 3 fields, got ${fields.length}`);
    const uuid = fields[0].trim();
    const pidText = fields[1].trim();
    const processName = fields.slice(2).join(",").trim();
    assert(GPU_UUID_RE.test(uuid), `line ${lineNumber}: invalid gpu_uuid`);
    assert(PID_RE.test(pidText), `line ${lineNumber}: invalid pid`);
    const pid = Number(pidText);
    assert(Number.isSafeInteger(pid) && pid > 0, `line ${lineNumber}: invalid pid`);
    assert(processName !== "", `line ${lineNumber}: invalid process_name`);
    return { uuid, pid, processName };
  });
}

export function faultFromJournal(message: string): GPUJournalFault | null {
  if (/GPU recovery action/i.test(message)) return "NodeReboot";
  if (/GPU has fallen off the bus/i.test(message)) return "GPUFallenOffBus";
  if (/\bXid\b/i.test(message)) return "Xid";
  if (/\bNVRM\b/i.test(message) && /\b(?:error|fatal)\b/i.test(message)) return "NVRMFault";
  return null;
}

export function evaluateHealth(
  sample: GPUSample,
  bootId: string,
  baseline: GPUBaseline,
  maxTemperatureC = 83,
): GPUHealthFault | null {
  const currentFault = sampleFault(sample, "MalformedSample");
  if (currentFault !== null) return currentFault;
  const currentBootId = typeof bootId === "string" ? bootId.trim() : "";
  if (currentBootId === "") return "MalformedBootId";
  const baselineIssue = baselineFault(baseline);
  if (baselineIssue !== null) return baselineIssue;
  if (!Number.isFinite(maxTemperatureC) || maxTemperatureC < 0 || maxTemperatureC > 150) return "MalformedPolicy";
  const baselineBootId = baseline.bootId.trim();
  const baselineGpu = baseline.gpu;
  if (currentBootId !== baselineBootId) return "BootIdChanged";
  if (sample.uuid !== baselineGpu.uuid) return "UUIDChanged";
  if (sample.pciBusId !== baselineGpu.pciBusId) return "PCIBusIdChanged";
  if (sample.driverVersion !== baselineGpu.driverVersion) return "DriverVersionChanged";
  if (sample.temperatureC >= maxTemperatureC) return "TemperatureExceeded";
  return null;
}
