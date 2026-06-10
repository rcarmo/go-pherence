package nv

import "github.com/rcarmo/go-pherence/backends/nvidia/internal/debuglog"

func debugf(format string, args ...any) {
	debuglog.Printf(format, args...)
}

func debugln(args ...any) {
	debuglog.Println(args...)
}
