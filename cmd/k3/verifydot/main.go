package main

import (
	"fmt"
	"os"
	"os/exec"
)

func main() {
	// Just run ime2run with a flag
	cmd := exec.Command(os.Args[0])
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	fmt.Println("Use /tmp/ime2fast for benchmarks")
}
