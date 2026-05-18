package model

import (
	"os"
	"testing"

	gpu "github.com/rcarmo/go-pherence/backends/cuda"
)

func TestMain(m *testing.M) {
	code := m.Run()
	gpu.Shutdown()
	os.Exit(code)
}
