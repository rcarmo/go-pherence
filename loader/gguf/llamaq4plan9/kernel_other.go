//go:build !amd64

package llamaq4plan9

import "fmt"

func Available() bool { return false }

func DotQ4_0x8Q8_0x4CompilerPlan9(q4, q8 []byte, blocks int, out *[32]float32) error {
	return fmt.Errorf("llama Q4_0x8 Plan 9 kernel is unavailable")
}

func ProjectQ4_0x8Q8_0StageRows(q4, q8 []byte, rowBase, rowGroups, rows, tokens, blocks int, out []float32) error {
	return fmt.Errorf("llama Q4_0x8 Plan 9 kernel is unavailable")
}

func ProjectQ4_0x8Q8_0Stage(q4, q8 []byte, rows, tokens, blocks int, out []float32) error {
	return fmt.Errorf("llama Q4_0x8 Plan 9 kernel is unavailable")
}
