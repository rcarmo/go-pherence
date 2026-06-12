// Package aipool provides the aicpu worker pool: a TCM-aware,
// barrier-synchronized pool that fans quantized GEMM work (AIGemmSpec) across
// the K3's cores, with optional TCM B-wave activation staging.
//
// It is distinct from ime2.WorkerPool, which is a simpler generic GEMM pool;
// aipool adds the engine-specific scheduling and TCM staging.
package aipool
