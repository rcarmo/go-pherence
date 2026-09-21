package main

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

func TestSafetensorNamesReadsShardedIndex(t *testing.T) {
	dir := t.TempDir()
	header := `{"mtp.fc.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(header)))
	shard := append(append([]byte{}, lenBuf[:]...), []byte(header)...)
	shard = append(shard, 0, 0, 0, 0)
	if err := os.WriteFile(filepath.Join(dir, "model-00001-of-00001.safetensors"), shard, 0644); err != nil {
		t.Fatalf("write shard: %v", err)
	}
	index := `{"weight_map":{"mtp.fc.weight":"model-00001-of-00001.safetensors"}}`
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(index), 0644); err != nil {
		t.Fatalf("write index: %v", err)
	}
	names, err := safetensorNames(dir)
	if err != nil {
		t.Fatalf("safetensorNames: %v", err)
	}
	if len(names) != 1 || names[0] != "mtp.fc.weight" {
		t.Fatalf("names=%v", names)
	}
}

func TestShouldFailStrict(t *testing.T) {
	meta := loaderconfig.QwenNativeMTPMetadata{HasNativeMTP: true}
	if !shouldFailStrict(true, meta, Report{}) {
		t.Fatal("strict incomplete native MTP did not fail")
	}
	if shouldFailStrict(false, meta, Report{}) {
		t.Fatal("non-strict failed")
	}
	if shouldFailStrict(true, meta, Report{MTPTensorComplete: true}) {
		t.Fatal("complete tensor set failed")
	}
	if shouldFailStrict(true, loaderconfig.QwenNativeMTPMetadata{}, Report{}) {
		t.Fatal("non-MTP metadata failed")
	}
}

func TestReportCanLoadSharedHeadJSON(t *testing.T) {
	report := Report{Config: loaderconfig.QwenNativeMTPMetadata{HiddenSize: 4, VocabSize: 2}, OptionalSharedHeadTensors: []string{"mtp.shared_head_head.weight"}, CanLoadSharedHead: true, MTPTensorCount: 3, OptionalSharedHeadCount: 1, MissingMTPTensorCount: 2, MTPTensorComplete: false}
	data, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var decoded Report
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if !decoded.CanLoadSharedHead || decoded.Config.VocabSize != 2 || len(decoded.OptionalSharedHeadTensors) != 1 || decoded.MTPTensorCount != 3 || decoded.OptionalSharedHeadCount != 1 || decoded.MissingMTPTensorCount != 2 || decoded.MTPTensorComplete {
		t.Fatalf("decoded=%+v", decoded)
	}
}

func TestSafetensorNamesFailsClosedOnPresentBrokenIndex(t *testing.T) {
	for _, index := range []string{"invalid", `{"weight_map":{"mtp.fc.weight":"missing.safetensors"}}`} {
		dir := t.TempDir()
		header := `{"mtp.fc.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`
		var size [8]byte
		binary.LittleEndian.PutUint64(size[:], uint64(len(header)))
		data := append(append(size[:], []byte(header)...), 0, 0, 0, 0)
		if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), data, 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(index), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := safetensorNames(dir); err == nil {
			t.Fatal("broken index silently replaced with monolithic file")
		}
	}
	if names, err := safetensorNames(t.TempDir()); err != nil || len(names) != 0 {
		t.Fatal("metadata-only directory rejected", names, err)
	}
}
func TestMetadataEnumerationLimits(t *testing.T) {
	for _, n := range []int{-1, 4097, int(^uint(0) >> 1)} {
		if validateMetadata(loaderconfig.QwenNativeMTPMetadata{NumHiddenLayers: n}) == nil {
			t.Fatal(n)
		}
		if validateMetadata(loaderconfig.QwenNativeMTPMetadata{MTPNumHiddenLayers: n}) == nil {
			t.Fatal(n)
		}
	}
	if err := validateMetadata(loaderconfig.QwenNativeMTPMetadata{NumHiddenLayers: 32, MTPNumHiddenLayers: 1}); err != nil {
		t.Fatal(err)
	}
}
