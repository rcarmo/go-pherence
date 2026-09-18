//go:build linux && riscv64

package aicpu

import (
	"fmt"
	"os"
	"testing"
)

// Linux/RISC-V alone does not imply K3 IME instructions or AI-core registration.
func TestMain(m *testing.M) {
	if os.Getenv("GO_PHERENCE_TEST_K3") != "1" {
		fmt.Println("SKIP: native K3 tests require GO_PHERENCE_TEST_K3=1 on a K3 board")
		os.Exit(0)
	}
	if _, err := os.Stat("/proc/set_ai_thread"); err != nil {
		fmt.Fprintln(os.Stderr, "K3 tests requested but /proc/set_ai_thread is unavailable:", err)
		os.Exit(1)
	}
	os.Exit(m.Run())
}
