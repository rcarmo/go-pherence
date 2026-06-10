package testexec

import (
	"os"
	"testing"
)

// HelperProcess lets command tests re-exec their own test binary and dispatch
// to main after a "--" separator when GO_WANT_HELPER_PROCESS=1 is set.
func HelperProcess(t *testing.T, main func()) {
	t.Helper()
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	for i, arg := range os.Args {
		if arg == "--" {
			os.Args = append([]string{os.Args[0]}, os.Args[i+1:]...)
			main()
			return
		}
	}
	os.Exit(2)
}
