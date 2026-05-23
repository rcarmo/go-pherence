// Package spacemitort contains integration scaffolding for SpacemiT's ONNX
// Runtime Execution Provider on K-series RISC-V boards.
//
// This backend is deliberately a bridge to the vendor-supported ONNX Runtime
// path rather than a direct Linlon/AIPU backend. On K3/Bianbu the practical AI
// stack is SpaceMITExecutionProvider + libspacemit_ep + libspine_tcm + /dev/tcm;
// there is no public /dev/aipu-style Linlon V5 SDK exposed to applications.
//
// The first milestone is capability detection and option plumbing. Actual ORT
// session execution should live behind build tags so normal builds do not take
// a hard cgo dependency on SpacemiT's riscv64-only libraries.
package spacemitort
