// Package inference provides mid-level numeric building blocks layered on top
// of the ime2 kernels: activation quantization, RMSNorm, and Q4_K / INT8
// mat-vec routines (serial, parallel, and WorkerPool-backed).
//
// It is a thin numeric layer; the full pure-Go transformer runtime lives in
// k3engine.
package inference
