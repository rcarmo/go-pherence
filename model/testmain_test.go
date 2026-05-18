package model

import (
	"os"
	"testing"

	cuda "github.com/rcarmo/go-pherence/backends/cuda"
)

func TestMain(m *testing.M) {
	code := m.Run()
	cuda.Shutdown()
	os.Exit(code)
}
