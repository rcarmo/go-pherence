// Command ime2run runs pure-Go IME2/RVV inference on the SpaceMIT K3.
// The implementation lives in backends/spacemit/k3engine.
package main

import "github.com/rcarmo/go-pherence/backends/spacemit/k3engine"

func main() { k3engine.Run() }
