package debuglog

import (
	"fmt"
	"os"
)

func Printf(format string, args ...any) {
	if os.Getenv("GO_PHERENCE_GPU_DEBUG") != "" {
		fmt.Printf(format, args...)
	}
}

func Println(args ...any) {
	if os.Getenv("GO_PHERENCE_GPU_DEBUG") != "" {
		fmt.Println(args...)
	}
}
