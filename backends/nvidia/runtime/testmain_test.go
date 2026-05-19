package nvidia

import (
	"os"
	"testing"
)

func TestMain(m *testing.M) {
	code := m.Run()
	Shutdown()
	os.Exit(code)
}
