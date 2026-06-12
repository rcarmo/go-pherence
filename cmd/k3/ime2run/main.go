// Command ime2run runs pure-Go IME2/RVV inference on the SpaceMIT K3.
// The implementation lives in backends/spacemit/aicpu.
package main

import "github.com/rcarmo/go-pherence/backends/spacemit/aicpu"

func main() { aicpu.Run() }
