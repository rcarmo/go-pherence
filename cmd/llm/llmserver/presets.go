package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type ModelPreset struct {
	ID         string
	Path       string
	Threads    int
	BatchSize  int
	GPULayers  int
	CtxSize    int
	CacheTypeK string
	CacheTypeV string
}

func ParseModelPresets(path string) ([]ModelPreset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var presets []ModelPreset
	var cur *ModelPreset
	scanner := bufio.NewScanner(f)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, ";") {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			if cur != nil {
				presets = append(presets, *cur)
			}
			id := strings.TrimSpace(line[1 : len(line)-1])
			if id == "" {
				return nil, fmt.Errorf("%s:%d: empty preset name", path, lineNo)
			}
			cur = &ModelPreset{ID: id}
			continue
		}
		if cur == nil {
			return nil, fmt.Errorf("%s:%d: key outside preset section", path, lineNo)
		}
		key, val, ok := strings.Cut(line, "=")
		if !ok {
			return nil, fmt.Errorf("%s:%d: expected key = value", path, lineNo)
		}
		key = strings.TrimSpace(key)
		val = strings.TrimSpace(val)
		switch key {
		case "model", "path":
			cur.Path = val
		case "threads":
			cur.Threads, err = atoiField(path, lineNo, key, val)
		case "batch-size", "batch_size", "ubatch", "ubatch-size":
			cur.BatchSize, err = atoiField(path, lineNo, key, val)
		case "gpu-layers", "gpu_layers", "ngl":
			cur.GPULayers, err = atoiField(path, lineNo, key, val)
		case "ctx-size", "ctx_size", "context-size":
			cur.CtxSize, err = atoiField(path, lineNo, key, val)
		case "cache-type-k", "cache_type_k":
			cur.CacheTypeK = val
		case "cache-type-v", "cache_type_v":
			cur.CacheTypeV = val
		case "flash-attn", "flash_attn", "split-mode", "split_mode", "tensor-split", "tensor_split", "main-gpu", "main_gpu",
			"temperature", "temp", "top-k", "top_k", "top-p", "top_p", "min-p", "min_p", "typical-p", "typical_p",
			"repeat-penalty", "repeat_penalty", "presence-penalty", "presence_penalty", "frequency-penalty", "frequency_penalty",
			"mirostat", "seed":
			return nil, fmt.Errorf("%s:%d: unsupported llama.cpp tuning control %q", path, lineNo, key)
		default:
			// Ignore unrelated llama-server keys so shared preset files remain usable.
			// Known graph, placement, KV, and sampling controls are handled above or
			// explicitly rejected rather than silently changing semantics.
		}
		if err != nil {
			return nil, err
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if cur != nil {
		presets = append(presets, *cur)
	}
	for i := range presets {
		if presets[i].Path == "" {
			return nil, fmt.Errorf("%s: preset %q missing model path", path, presets[i].ID)
		}
	}
	return presets, nil
}

func atoiField(path string, lineNo int, key, val string) (int, error) {
	n, err := strconv.Atoi(val)
	if err != nil {
		return 0, fmt.Errorf("%s:%d: invalid integer for %s: %q", path, lineNo, key, val)
	}
	return n, nil
}
