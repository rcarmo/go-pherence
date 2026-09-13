package main

import (
	"runtime"
	"strings"
	"testing"
)

func TestWorkerFlagValidation(t *testing.T) {
	old := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(old)
	for _, tt := range []struct {
		name string
		args []string
		want string
	}{
		{"negative", []string{"-gemm-workers", "-1"}, "gemm-workers must"},
		{"too-many", []string{"-gemm-workers", "65"}, "gemm-workers must"},
		{"wrong-mode", []string{"-gemm-workers", "2"}, "only to synthesis/generation"},
		{"oversubscription", []string{"-mode", "generate", "-threads", "1", "-gemm-workers", "4"}, "model directory required"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := run(tt.args)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("got %v, want %q", err, tt.want)
			}
		})
	}
}
