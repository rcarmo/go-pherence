//go:build !linux || !amd64

package main

import (
	"context"
	"fmt"
	"github.com/rcarmo/go-pherence/runtime/speechjob/httpapi"
)

func buildProfile(context.Context, ServerConfig, bool) ([]httpapi.Profile, error) {
	return nil, fmt.Errorf("speechjobserve model profile requires Linux/amd64")
}
