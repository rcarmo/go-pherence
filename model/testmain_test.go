package model

import (
	"os"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func TestMain(m *testing.M) {
	code := m.Run()
	nvidia.Shutdown()
	os.Exit(code)
}
