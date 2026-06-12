// Package tcm is a driver for the SpaceMIT K3 TCM (Tightly-Coupled Memory), the
// on-chip SRAM scratchpad. It memory-maps and acquires per-core TCM regions via
// ioctl (8 cores x 384 KB).
//
// Note: TCM is uncached for the CPU/RVV, so it is only useful as a DMA staging
// buffer, not as a general compute scratchpad — see research/aicpu-whisper for
// the measurements behind that conclusion.
package tcm
