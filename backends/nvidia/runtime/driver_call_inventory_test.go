package nvidia

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
)

// All raw driver-call boundaries require review. Higher-level tensor kernels
// must use the checked driver wrappers, not EnsureContext followed by a raw call.
// This inventory is a guard against new unreviewed sites, not a lock proof.
func TestRawCUDADriverCallsStayInReviewedScopes(t *testing.T) {
	allowed := map[string]map[string]bool{
		"runtime.go":      {"Init": true, "ensureContextLocked": true, "Malloc": true, "Free": true, "Upload": true, "UploadBytes": true, "UploadUint32": true, "DownloadBytes": true, "Download": true, "syncCounted": true, "loadModuleDataWithLog": true, "loadPTXModuleLocked": true, "LaunchKernel": true, "MemInfo": true, "Shutdown": true},
		"driver_scope.go": {"unloadModule": true, "prewarmAllocatorLocked": true},
		"devbuf.go":       {"copyDtoDAsync": true, "CopyDtoD": true, "ZeroFloat32Buffer": true},
		"streams.go":      {"initStreamsLocked": true, "PrefetchWeights": true, "MarkComputeDone": true, "WaitPrefetch": true, "SyncAll": true, "BeginCapture": true, "EndCapture": true, "Launch": true, "Destroy": true, "launchKernelOnStreamLocked": true, "shutdownStreams": true},
		"mega_module.go":  {"loadMegaModule": true},
		"bf16_native.go":  {"InitNativeBF16": true},
	}
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	fs := token.NewFileSet()
	calls := 0
	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(fs, name, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				id, ok := call.Fun.(*ast.Ident)
				if !ok || len(id.Name) < 3 || !strings.HasPrefix(id.Name, "cu") || id.Name[2] < 'A' || id.Name[2] > 'Z' {
					return true
				}
				calls++
				if !allowed[name][fn.Name.Name] {
					t.Errorf("unreviewed CUDA driver call %s in %s:%s", id.Name, name, fn.Name.Name)
				}
				return true
			})
		}
	}
	if calls < 40 {
		t.Fatal("driver inventory unexpectedly incomplete", calls)
	}
}
