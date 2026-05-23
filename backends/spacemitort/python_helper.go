package spacemitort

// PythonHelperSnippet is a minimal reference for loading the SpacemiT EP from
// Python. It mirrors /usr/lib/python3.14/dist-packages/spacemit_ort/__init__.py.
const PythonHelperSnippet = `
import onnxruntime as ort
import spacemit_ort

providers = [spacemit_ort.kSpaceMITExecutionProvider]
provider_options = [{
    "shared_lib_path": spacemit_ort.EPLibPath,
    "provider_factory_entry_point": spacemit_ort.kExecutionProviderSharedLibraryEntry,
}]
session = ort.InferenceSession(model_path, providers=providers, provider_options=provider_options)
`
