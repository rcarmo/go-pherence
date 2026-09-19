package safetensors

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
)

// Select by index existence, not OpenSharded's error: a missing shard must not
// silently select a different checkpoint. Broken index symlinks also fail closed.
func resolveMetadataPath(dir, explicit string) (string, bool, error) {
	if explicit != "" {
		return explicit, strings.HasSuffix(explicit, ".index.json"), nil
	}
	index := filepath.Join(dir, "model.safetensors.index.json")
	if _, err := os.Lstat(index); err == nil {
		return index, true, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", false, err
	}
	return filepath.Join(dir, "model.safetensors"), false, nil
}

// TensorInfosFrom resolves a safetensors source for a model directory (or an
// explicit file path) and returns its tensor infos. Resolution order: the
// explicit file if given, then a sharded index (model.safetensors.index.json),
// then a single model.safetensors.
//
// It centralizes the source-resolution previously duplicated across the model
// inspect commands.
func TensorInfosFrom(modelDir, explicit string) (map[string]TensorInfo, error) {
	path, sharded, err := resolveMetadataPath(modelDir, explicit)
	if err != nil {
		return nil, err
	}
	if sharded {
		sf, err := OpenSharded(path)
		if err != nil {
			return nil, err
		}
		defer sf.Close()
		return sf.TensorInfos(), nil
	}
	f, err := Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return f.TensorInfos(), nil
}

// NamesFrom resolves a safetensors source (see TensorInfosFrom) and returns its
// tensor names.
func NamesFrom(modelDir, explicit string) ([]string, error) {
	path, sharded, err := resolveMetadataPath(modelDir, explicit)
	if err != nil {
		return nil, err
	}
	if sharded {
		sf, err := OpenSharded(path)
		if err != nil {
			return nil, err
		}
		defer sf.Close()
		return sf.Names(), nil
	}
	f, err := Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return f.Names(), nil
}
