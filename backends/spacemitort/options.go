package spacemitort

import "strconv"

// Options maps to documented SpaceMITExecutionProvider options.
type Options struct {
	IntraThreadNum         int
	UseGlobalIntraThread   bool
	DumpSubgraphs          bool
	DebugProfilePrefix     string
	DumpTensorsDir         string
	DisableOpTypeFilter    string
	DisableOpNameFilter    string
	DisableFloat16Epilogue bool
	DenseAccuracyLevel     int
	EnableBlockLayout      bool
	EnableDMA              bool
}

// ProviderName is the ONNX Runtime provider name documented by SpacemiT.
const ProviderName = "SpaceMITExecutionProvider"

// SharedProviderEntryPoint is the factory symbol exported by libspacemit_ep.so
// and used by the Python spacemit_ort helper.
const SharedProviderEntryPoint = "GetSpaceMITSharedProviderFactory"

// ProviderOptions returns string options suitable for ORT provider option maps
// or environment-variable based runners.
func (o Options) ProviderOptions() map[string]string {
	m := map[string]string{}
	if o.IntraThreadNum > 0 {
		m["SPACEMIT_EP_INTRA_THREAD_NUM"] = strconv.Itoa(o.IntraThreadNum)
	}
	setBool(m, "SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD", o.UseGlobalIntraThread)
	setBool(m, "SPACEMIT_EP_DUMP_SUBGRAPHS", o.DumpSubgraphs)
	if o.DebugProfilePrefix != "" {
		m["SPACEMIT_EP_DEBUG_PROFILE"] = o.DebugProfilePrefix
	}
	if o.DumpTensorsDir != "" {
		m["SPACEMIT_EP_DUMP_TENSORS"] = o.DumpTensorsDir
	}
	if o.DisableOpTypeFilter != "" {
		m["SPACEMIT_EP_DISABLE_OP_TYPE_FILTER"] = o.DisableOpTypeFilter
	}
	if o.DisableOpNameFilter != "" {
		m["SPACEMIT_EP_DISABLE_OP_NAME_FILTER"] = o.DisableOpNameFilter
	}
	setBool(m, "SPACEMIT_EP_DISABLE_FLOAT16_EPILOGUE", o.DisableFloat16Epilogue)
	if o.DenseAccuracyLevel > 0 {
		m["SPACEMIT_EP_DENSE_ACCURACY_LEVEL"] = strconv.Itoa(o.DenseAccuracyLevel)
	}
	setBool(m, "SPACEMIT_EP_ENABLE_BLOCKLAYOUT", o.EnableBlockLayout)
	setBool(m, "SPACEMIT_EP_ENABLE_DMA", o.EnableDMA)
	return m
}

func setBool(m map[string]string, key string, value bool) {
	if value {
		m[key] = "1"
	}
}

// PythonProviderOptions mirrors the provider option shape used by the vendor
// python package. ONNX Runtime's Python API loads the EP as a shared library
// provider when these two fields are present.
func (o Options) PythonProviderOptions(sharedLibraryPath string) map[string]string {
	m := o.ProviderOptions()
	if sharedLibraryPath != "" {
		m["shared_lib_path"] = sharedLibraryPath
	}
	m["provider_factory_entry_point"] = SharedProviderEntryPoint
	return m
}
