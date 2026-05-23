package spacemitort

import (
	"os"
	"runtime"
)

// Capabilities describes the locally visible SpacemiT AI runtime surface.
type Capabilities struct {
	Arch                  string
	HasDevTCM             bool
	HasSpineTCMLibrary    bool
	HasSpacemitEPLibrary  bool
	HasONNXRuntimeLibrary bool
	HasPythonORTPackage   bool
	LikelyUsable          bool
}

// RuntimeCapabilities probes the files/packages that make the SpacemiT ONNX
// Runtime EP usable. It intentionally avoids dlopen/cgo so it is safe on all
// platforms and in cross-compiled tests.
func RuntimeCapabilities() Capabilities {
	c := Capabilities{Arch: runtime.GOARCH}
	c.HasDevTCM = fileExists("/dev/tcm")
	c.HasSpineTCMLibrary = anyPathExists([]string{
		"/usr/lib/libspine_tcm.so",
		"/usr/lib/libspine_tcm.so.0",
		"/usr/lib/riscv64-linux-gnu/libspine_tcm.so",
	})
	c.HasSpacemitEPLibrary = anyPathExists([]string{
		"/usr/lib/libspacemit_ep.so",
		"/usr/lib/libspacemit_ep.so.2",
		"/usr/lib/libspacemit_ep.so.2.0.2+rc5",
	})
	c.HasONNXRuntimeLibrary = anyPathExists([]string{
		"/usr/lib/libonnxruntime.so",
		"/usr/lib/libonnxruntime.so.1",
		"/usr/lib/libonnxruntime.so.1.24.2+spacemit.a1",
	})
	c.HasPythonORTPackage = anyPathExists([]string{
		"/usr/lib/python3.14/dist-packages/onnxruntime",
		"/usr/lib/python3.13/dist-packages/onnxruntime",
		"/usr/lib/python3/dist-packages/onnxruntime",
	})
	c.LikelyUsable = c.Arch == "riscv64" && c.HasDevTCM && c.HasSpineTCMLibrary && c.HasSpacemitEPLibrary && c.HasONNXRuntimeLibrary
	return c
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func anyPathExists(paths []string) bool {
	for _, path := range paths {
		if fileExists(path) {
			return true
		}
	}
	return false
}
