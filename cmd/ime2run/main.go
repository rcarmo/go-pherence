// Command ime2run runs pure-Go IME2/RVV inference on the SpaceMIT K3.
// The implementation lives in backends/spacemit/k3.
package main

import k3 "github.com/rcarmo/go-pherence/backends/spacemit/k3"

func main() { k3.Run() }
