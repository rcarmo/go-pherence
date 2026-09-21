package jevlike

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ConvertFeatureCache rounds immutable F32 features into a NEW FP16 cache.
// Options were pooled in F32 before original storage. The converted contract has
// a different identity; never use a F32-bound checkpoint with it implicitly.
func ConvertFeatureCache(source, destination string, budget int64) (int, error) {
	c, err := LoadFeatureContract(source)
	if err != nil {
		return 0, err
	}
	if c.DType != "f32" {
		return 0, fmt.Errorf("source must be F32")
	}
	if _, err = os.Stat(destination); !os.IsNotExist(err) {
		return 0, fmt.Errorf("conversion destination must not exist")
	}
	if _, err = os.Stat(filepath.Join(source, ".writer.lock")); !os.IsNotExist(err) {
		return 0, fmt.Errorf("close source writer before conversion")
	}
	src, err := OpenFeatureCache(source, c, budget, nil, nil)
	if err != nil {
		return 0, err
	}
	defer src.Close()
	target := c
	target.DType = "f16"
	var current featureHeader
	// Conversion writes checked records directly; neither callback may run.
	tokenize := func(string) ([]int, error) { return nil, fmt.Errorf("conversion cannot tokenize") }
	encode := func([]int) ([][]float32, error) { return nil, fmt.Errorf("conversion cannot execute an encoder") }
	dst, err := OpenFeatureCache(destination, target, budget, tokenize, encode)
	if err != nil {
		return 0, err
	}
	defer dst.Close()
	entries, err := os.ReadDir(source)
	if err != nil {
		return 0, err
	}
	count := 0
	for _, entry := range entries {
		if !strings.HasSuffix(entry.Name(), ".jvf") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(source, entry.Name()))
		if err != nil {
			return count, err
		}
		if len(data) < 40 {
			return count, fmt.Errorf("truncated feature")
		}
		n := int(binary.LittleEndian.Uint32(data[4:]))
		if n < 2 || n > len(data)-40 {
			return count, fmt.Errorf("invalid header")
		}
		if err = json.Unmarshal(data[8:8+n], &current); err != nil {
			return count, err
		}
		if current.Key+".jvf" != entry.Name() {
			return count, fmt.Errorf("source filename/key mismatch")
		}
		rows, err := src.decodeFeature(data, current.Key, current.Text, current.MaxTokens, current.Pooled)
		if err != nil {
			return count, err
		}
		query := struct {
			Contract, Text string
			Limit          int
			Pooled         bool
		}{target.ID(), current.Text, current.MaxTokens, current.Pooled}
		q, _ := json.Marshal(query)
		key := featureHash(q)
		payload, err := featurePayload(rows, "f16")
		if err != nil {
			return count, err
		}
		current.Key = key
		current.ContractID = target.ID()
		current.DType = "f16"
		current.PayloadSHA256 = featureHash(payload)
		bytes, err := packFeature(current, payload)
		if err != nil {
			return count, err
		}
		if dst.bytes+int64(len(bytes)) > budget {
			return count, fmt.Errorf("converted cache budget exceeded")
		}
		if err = featurePublish(filepath.Join(destination, key+".jvf"), bytes); err != nil {
			return count, err
		}
		dst.bytes += int64(len(bytes))
		count++
	}
	return count, nil
}
