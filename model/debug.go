package model

import (
	"fmt"
	"os"
)

func loaderDebugf(format string, args ...any) {
	if os.Getenv("GO_PHERENCE_LOAD_DEBUG") != "" {
		fmt.Printf(format, args...)
	}
}
