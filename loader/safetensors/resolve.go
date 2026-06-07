package safetensors

import "path/filepath"

// TensorInfosFrom resolves a safetensors source for a model directory (or an
// explicit file path) and returns its tensor infos. Resolution order: the
// explicit file if given, then a sharded index (model.safetensors.index.json),
// then a single model.safetensors.
//
// It centralizes the source-resolution previously duplicated across the model
// inspect commands.
func TensorInfosFrom(modelDir, explicit string) (map[string]TensorInfo, error) {
	if explicit != "" {
		f, err := Open(explicit)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		return f.TensorInfos(), nil
	}
	if sf, err := OpenSharded(filepath.Join(modelDir, "model.safetensors.index.json")); err == nil {
		defer sf.Close()
		return sf.TensorInfos(), nil
	}
	f, err := Open(filepath.Join(modelDir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return f.TensorInfos(), nil
}

// NamesFrom resolves a safetensors source (see TensorInfosFrom) and returns its
// tensor names.
func NamesFrom(modelDir, explicit string) ([]string, error) {
	if explicit != "" {
		f, err := Open(explicit)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		return f.Names(), nil
	}
	if sf, err := OpenSharded(filepath.Join(modelDir, "model.safetensors.index.json")); err == nil {
		defer sf.Close()
		return sf.Names(), nil
	}
	f, err := Open(filepath.Join(modelDir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return f.Names(), nil
}
