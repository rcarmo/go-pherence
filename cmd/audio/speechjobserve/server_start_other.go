//go:build !linux || !amd64

package main

import (
	"context"
	"fmt"
	"io"
)

func start(context.Context, string, bool, io.Writer) error {
	return fmt.Errorf("speechjobserve model loading and checking require Linux/amd64")
}
func startWithRuntime(context.Context, string, bool, io.Writer, vulkanProfileRuntime) error {
	return fmt.Errorf("speechjobserve model loading and checking require Linux/amd64")
}
