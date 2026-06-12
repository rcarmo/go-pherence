// Package ime2 implements low-level int8 matrix-multiply kernels for the
// SpaceMIT K3 IME (Integrated Matrix Engine), using the X60 RVV vmadot
// instructions, plus a generic barrier-based WorkerPool that fans GEMM work
// across cores.
//
// This is a leaf kernel package: it does not depend on the higher-level
// inference engine (aicpu).
package ime2
