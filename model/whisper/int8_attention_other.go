//go:build !riscv64

package whisper

func attnInt8Head(scores, outh, qh, kh, vh []float32, seqQ, seqKV, headDim int, scale float32,
	mq, nk, kpad int, qi8 []int8, qs []float32, ki8 []int8, ks []float32, cqk []int32, qp, kp []int8,
	vhT []float32, vti8 []int8, vts []float32, sPad []float32, si8 []int8, ss []float32, cout []int32, sp, vtp []int8) {
	// Non-RISC-V builds never enable attnInt8; this stub only keeps generic
	// Whisper tests buildable on development hosts.
}
