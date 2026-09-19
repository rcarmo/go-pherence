package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	backbone "github.com/rcarmo/go-pherence/model"
	"github.com/rcarmo/go-pherence/model/jevlike"
)

type verifiedAsset struct {
	Path   string `json:"path"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}
type verifiedAssets struct {
	Version int `json:"version"`
	Sources []struct {
		Repository string `json:"repository"`
		Revision   string `json:"revision"`
	} `json:"sources"`
	Files []verifiedAsset `json:"files"`
}

// Full hashes at startup are deliberate. No checkpoint is admitted merely
// because its directory name or hidden width matches the experiment.
func verifyScoreAssets(dir, record string) (string, error) {
	data, err := os.ReadFile(record)
	if err != nil {
		return "", err
	}
	var manifest verifiedAssets
	if err = json.Unmarshal(data, &manifest); err != nil {
		return "", err
	}
	if manifest.Version != 1 || len(manifest.Sources) != 1 || len(manifest.Files) == 0 {
		return "", fmt.Errorf("invalid verified asset manifest")
	}
	if len(manifest.Sources[0].Revision) != 40 {
		return "", fmt.Errorf("unversioned encoder")
	}
	base, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	base, err = filepath.EvalSymlinks(base)
	if err != nil {
		return "", err
	}
	seen := map[string]bool{}
	for _, file := range manifest.Files {
		path := file.Path
		if !filepath.IsAbs(path) {
			path = filepath.Join(filepath.Dir(record), path)
		}
		abs, err := filepath.Abs(path)
		if err != nil {
			return "", err
		}
		abs, err = filepath.EvalSymlinks(abs)
		if err != nil {
			return "", err
		}
		if filepath.Dir(abs) != base {
			return "", fmt.Errorf("asset %s is outside model directory", path)
		}
		if seen[filepath.Base(path)] {
			return "", fmt.Errorf("duplicate asset")
		}
		seen[filepath.Base(path)] = true
		f, err := os.Open(path)
		if err != nil {
			return "", err
		}
		h := sha256.New()
		n, err := io.Copy(h, f)
		f.Close()
		if err != nil {
			return "", err
		}
		if n != file.Bytes || hex.EncodeToString(h.Sum(nil)) != file.SHA256 {
			return "", fmt.Errorf("asset identity mismatch: %s", path)
		}
	}
	for _, name := range []string{"config.json", "tokenizer.json", "tokenizer_config.json"} {
		if !seen[name] {
			return "", fmt.Errorf("unverified %s", name)
		}
	}
	// All safetensor files referenced by the loader must be verified.
	if data, err := os.ReadFile(filepath.Join(base, "model.safetensors.index.json")); err == nil {
		if !seen["model.safetensors.index.json"] {
			return "", fmt.Errorf("unverified tensor index")
		}
		var idx struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err = json.Unmarshal(data, &idx); err != nil {
			return "", err
		}
		if len(idx.WeightMap) == 0 {
			return "", fmt.Errorf("empty tensor index")
		}
		for _, name := range idx.WeightMap {
			if filepath.Base(name) != name || !seen[name] {
				return "", fmt.Errorf("unverified shard %s", name)
			}
		}
	} else if !os.IsNotExist(err) {
		return "", err
	} else if !seen["model.safetensors"] {
		return "", fmt.Errorf("unverified model weights")
	}
	return manifest.Sources[0].Repository + "@" + manifest.Sources[0].Revision, nil
}

func runScore(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("score", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("encoder-model", "", "Pinned local BF16 Qwen3 directory")
	identity := fs.String("verified-assets", "", "Downloader model-verified.json; complete startup hashes required")
	request := fs.String("request", "", "JSON request with evidence, question, stable candidate IDs/text and temperature")
	budget := fs.Uint64("gpu-budget-mib", 10*1024, "Maximum encoder-owned GPU MiB")
	reserve := fs.Uint64("gpu-reserve-mib", 1024, "Minimum free GPU MiB after allocation")
	maxTokens := fs.Int("max-tokens", 512, "Rendered prompt limit; overlength rejected")
	if err := parseSubcommandFlags(fs, args); err != nil {
		if err == errHelpRequested {
			return nil
		}
		return err
	}
	if *dir == "" || *identity == "" || *request == "" {
		return fmt.Errorf("-encoder-model, -verified-assets and -request required")
	}
	if *budget > 1<<20 || *reserve > 1<<20 || *budget == 0 || *reserve == 0 {
		return fmt.Errorf("invalid memory budget")
	}
	f, err := os.Open(*request)
	if err != nil {
		return err
	}
	defer f.Close()
	dec := json.NewDecoder(io.LimitReader(f, 1<<20))
	dec.DisallowUnknownFields()
	var input jevlike.DirectChoiceRequest
	if err = dec.Decode(&input); err != nil {
		return err
	}
	if dec.Decode(new(any)) != io.EOF {
		return fmt.Errorf("trailing request data")
	}
	prompt, err := jevlike.LoadQwen3ChoicePrompt(*dir, *maxTokens)
	if err != nil {
		return err
	}
	if _, _, _, err = prompt.Prepare(input); err != nil {
		return err
	} // no GPU for bad input
	started := time.Now()
	modelID, err := verifyScoreAssets(*dir, *identity)
	if err != nil {
		return err
	}
	hashSeconds := time.Since(started).Seconds()
	encoder, err := backbone.NewFrozenGPUEncoder(*dir, backbone.FrozenGPUOptions{MaxTokens: *maxTokens, BudgetBytes: *budget << 20, ReserveBytes: *reserve << 20})
	if err != nil {
		return err
	}
	defer nvidia.Shutdown()
	defer encoder.Close()
	started = time.Now()
	result, err := jevlike.ScoreChoices(encoder, prompt, input)
	if err != nil {
		return err
	}
	seconds := time.Since(started).Seconds()
	result.ProjectionBackend = "cpu-f64-selected-bf16-rows"
	return writeJSON(stdout, struct {
		ModelID         string                     `json:"model_id"`
		Backend         string                     `json:"backend"`
		HashSeconds     float64                    `json:"hash_seconds"`
		DecisionSeconds float64                    `json:"decision_seconds"`
		Stats           backbone.FrozenGPUStats    `json:"gpu"`
		Result          jevlike.DirectChoiceResult `json:"result"`
	}{modelID, "cuda-bf16-resident-compensated-f32", hashSeconds, seconds, encoder.Stats(), result})
}
