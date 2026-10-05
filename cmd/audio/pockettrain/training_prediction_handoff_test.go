//go:build linux

package main

import (
	"context"
	"strings"
	"testing"
)

func TestPreparedPredictionHandoffIdentity(t *testing.T) {
	pin := strings.Repeat("a", 64)
	for _, prefix := range []string{"ptx", "vulkan"} {
		name := prefix + "-resident-predictionhandoff-hybrid"
		scope := name + "-v1"
		if got, err := trainingBackendScope(name, true, pin, 1024); err != nil || got != scope {
			t.Fatal("scope", got, err)
		}
		if _, err := trainingBackendScope(name, false, pin, 1024); err == nil {
			t.Fatal("optin")
		}
		want := hashText("base;resident-predictionhandoff-backend=" + scope + ";device-runtime=" + pin)
		got := trainingBackendOptionsHash("base", scope, pin)
		if got != want || got == trainingBackendOptionsHash("base", prefix+"-resident-distilladmission-hybrid-v1", pin) {
			t.Fatal("identity")
		}
		if !strings.Contains(trainingBackendNotice(scope), "retained") {
			t.Fatal("notice")
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, _, err := openTrainingBackend(ctx, scope, 1024); err == nil {
			t.Fatal("cancel before native")
		}
	}
}
