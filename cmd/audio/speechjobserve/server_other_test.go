//go:build !linux || !amd64

package main

import (
	"context"
	"io"
	"strings"
	"testing"
)

func TestStartUnsupportedPlatform(t *testing.T) {
	if err := start(context.Background(), "/tmp/server.json", true, io.Discard); err == nil || !strings.Contains(err.Error(), "Linux/amd64") {
		t.Fatal(err)
	}
	if err := startWithRuntime(context.Background(), "/tmp/server.json", false, io.Discard, vulkanProfileRuntime{}); err == nil || !strings.Contains(err.Error(), "Linux/amd64") {
		t.Fatal(err)
	}
}
