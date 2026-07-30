package nvidia

// Unified kernel loader: merges ALL PTX entries into one module.
// Solves the cuModuleLoadData error 201 (can't load multiple modules).

import (
	"github.com/rcarmo/go-pherence/backends/nvidia/internal/debuglog"
	"os"
	"strings"
	"sync"
	"unsafe"
)

var (
	megaModuleOnce sync.Once
	megaModule     CUmodule
	megaModuleOK   bool

	// Function handles extracted from the mega module
	sgemmReady bool

	// Set in sgemm.go, runtime helper files
)

// stripPTXHeader removes .version/.target/.address_size lines from PTX
func stripPTXHeader(ptx string) string {
	var lines []string
	for _, l := range strings.Split(ptx, "\n") {
		trimmed := strings.TrimSpace(l)
		if strings.HasPrefix(trimmed, ".version") ||
			strings.HasPrefix(trimmed, ".target") ||
			strings.HasPrefix(trimmed, ".address_size") {
			continue
		}
		lines = append(lines, l)
	}
	return strings.Join(lines, "\n")
}

// loadMegaModule compiles ALL PTX kernels into one CUDA module.
func loadMegaModule() {
	megaModuleOnce.Do(func() {
		if !Init() {
			return
		}

		// Pre-warm allocator
		var warmPtr CUdeviceptr
		if r := cuMemAlloc(&warmPtr, 64*1024*1024); r == CUDA_SUCCESS {
			cuMemFree(warmPtr)
		}

		// Combine all PTX entries into one module
		var combined strings.Builder
		combined.WriteString(".version 7.0\n.target sm_80\n.address_size 64\n\n")

		entries := megaModuleEntries()
		if err := validateModuleEntries(entries); err != nil {
			debuglog.Printf("[gpu] invalid mega module entries: %v\n", err)
			return
		}

		for _, e := range entries {
			combined.WriteString(stripPTXHeader(e.ptx))
			combined.WriteString("\n")
		}

		ptxStr := combined.String()
		if os.Getenv("GO_PHERENCE_GPU_DEBUG_PTX") != "" {
			_ = os.WriteFile("/tmp/go-pherence-mega.ptx", []byte(ptxStr), 0o600)
		}
		ptxBytes := append([]byte(ptxStr), 0)

		EnsureContext()
		if r := loadModuleDataWithLog(&megaModule, unsafe.Pointer(&ptxBytes[0])); r != CUDA_SUCCESS {
			debuglog.Printf("[gpu] mega module load failed: error %d\n", r)
			return
		}

		// Extract all function handles
		allOK := true
		functions := moduleFunctions{}
		extractFn := func(name string) CUfunction {
			nameBytes := append([]byte(name), 0)
			var fn CUfunction
			if r := cuModuleGetFunction(&fn, megaModule, unsafe.Pointer(&nameBytes[0])); r != CUDA_SUCCESS {
				debuglog.Printf("[gpu] get %s: error %d\n", name, r)
				allOK = false
				return 0
			}
			return fn
		}

		for _, e := range entries {
			functions[e.name] = extractFn(e.name)
		}
		bindMegaModuleFunctions(functions)

		if allOK {
			markMegaModuleReady(len(entries))
			// Initialize streams for prefetch overlap
			if err := initStreams(); err != nil {
				debuglog.Printf("[gpu] streams: %v\n", err)
			}
			// Try native BF16 kernels (Ampere+)
			InitNativeBF16()
		}
	})
}

// InitAllKernels loads all GPU ptx. Call from the CUDA-owning thread.
func InitAllKernels() {
	loadMegaModule()
}

// SgemmReady returns true if GPU SGEMM is available.
func SgemmReady() bool {
	loadMegaModule()
	return sgemmReady
}

// Q4Ready returns true if the INT4 GPU kernel is available.
func Q4Ready() bool {
	loadMegaModule()
	return q4Ready
}

func shutdownMegaModule() {
	freeNVFP4Scratch()
	FreeBF16LMHeadScratch()
	shutdownAttentionSplitKVCandidate()
	if megaModule != 0 && cuModuleUnload != nil {
		EnsureContext()
		cuModuleUnload(megaModule)
	}
	resetMegaModuleState()
}
