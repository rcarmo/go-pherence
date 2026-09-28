package pockettts

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/rcarmo/go-pherence/internal/checked"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

const trainingLatentMetadataLimit = 1 << 20

// TrainingLatentMetadata is the exact sidecar emitted by upstream
// training/scripts/precompute_latents.py at UpstreamCommit.
type TrainingLatentMetadata struct {
	StitchFrames int     `json:"stitch_frames"`
	NoiseFloor   float64 `json:"noise_floor"`
	FrameRate    float64 `json:"frame_rate"`
	WeightsPath  string  `json:"weights_path"`
	MimiHash     string  `json:"mimi_hash"`
}

// TrainingLatentCacheLimits makes admission of large, externally generated
// stores explicit. MaxTotalElements counts decoded latent float values.
type TrainingLatentCacheLimits struct {
	MaxEntries       int
	MaxFramesPerRow  int
	MaxTotalElements int
}

// TrainingLatentShard describes one admitted upstream per-utterance shard.
type TrainingLatentShard struct {
	Path     string
	Frames   int
	Channels int
	SHA256   [32]byte
}

// TrainingLatentCache is immutable after admission. Accessors return values or
// deep copies; LoadRow returns owned F32 data and verifies its content digest.
type TrainingLatentCache struct {
	manifestPath string
	metadata     TrainingLatentMetadata
	entries      []TrainingEntry
	shards       []TrainingLatentShard
}

func (c *TrainingLatentCache) ManifestPath() string {
	if c == nil {
		return ""
	}
	return c.manifestPath
}
func (c *TrainingLatentCache) Metadata() TrainingLatentMetadata {
	if c == nil {
		return TrainingLatentMetadata{}
	}
	return c.metadata
}
func (c *TrainingLatentCache) Len() int {
	if c == nil {
		return 0
	}
	return len(c.entries)
}
func (c *TrainingLatentCache) Entry(index int) (TrainingEntry, bool) {
	if c == nil || index < 0 || index >= len(c.entries) {
		return TrainingEntry{}, false
	}
	entry := c.entries[index]
	entry.Words = append([]TrainingWord(nil), entry.Words...)
	for i := range entry.Words {
		if entry.Words[i].Start != nil {
			v := *entry.Words[i].Start
			entry.Words[i].Start = &v
		}
		if entry.Words[i].End != nil {
			v := *entry.Words[i].End
			entry.Words[i].End = &v
		}
	}
	return entry, true
}
func (c *TrainingLatentCache) Shard(index int) (TrainingLatentShard, bool) {
	if c == nil || index < 0 || index >= len(c.shards) {
		return TrainingLatentShard{}, false
	}
	return c.shards[index], true
}

func (m TrainingLatentMetadata) validate(expectedHash string) error {
	if m.StitchFrames <= 0 || !finite64(m.NoiseFloor) || m.NoiseFloor < 0 || m.FrameRate != float64(FrameRateNumerator)/FrameRateDenominator || strings.TrimSpace(m.WeightsPath) == "" {
		return fmt.Errorf("invalid Pocket TTS latent metadata")
	}
	decoded, err := hex.DecodeString(m.MimiHash)
	if err != nil || len(decoded) != 32 || strings.ToLower(m.MimiHash) != m.MimiHash {
		return fmt.Errorf("invalid Pocket TTS Mimi encode hash")
	}
	if expectedHash != "" && m.MimiHash != expectedHash {
		return fmt.Errorf("Pocket TTS latent Mimi hash=%s want=%s", m.MimiHash, expectedHash)
	}
	return nil
}

func readTrainingLatentMetadata(path, expectedHash string) (TrainingLatentMetadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return TrainingLatentMetadata{}, fmt.Errorf("open Pocket TTS latent metadata: %w", err)
	}
	defer f.Close()
	if stat, statErr := f.Stat(); statErr != nil || stat.Size() > trainingLatentMetadataLimit {
		return TrainingLatentMetadata{}, fmt.Errorf("Pocket TTS latent metadata exceeds %d bytes", trainingLatentMetadataLimit)
	}
	decoder := json.NewDecoder(io.LimitReader(f, trainingLatentMetadataLimit+1))
	decoder.DisallowUnknownFields()
	var metadata TrainingLatentMetadata
	if err = decoder.Decode(&metadata); err != nil {
		return TrainingLatentMetadata{}, fmt.Errorf("decode Pocket TTS latent metadata: %w", err)
	}
	var trailing any
	if err = decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("trailing JSON value")
		}
		return TrainingLatentMetadata{}, fmt.Errorf("decode Pocket TTS latent metadata: %w", err)
	}
	if err = metadata.validate(expectedHash); err != nil {
		return TrainingLatentMetadata{}, err
	}
	return metadata, nil
}

// OpenTrainingLatentCache admits an upstream frozen-Mimi cache without
// materializing tensor bodies. The manifest must use upstream's
// <source>_latents.jsonl naming and exact index-addressed shard paths.
func OpenTrainingLatentCache(manifestPath, expectedMimiHash string, latentChannels int, limits TrainingLatentCacheLimits) (*TrainingLatentCache, error) {
	if latentChannels <= 0 || limits.MaxEntries <= 0 || limits.MaxFramesPerRow <= 0 || limits.MaxTotalElements <= 0 {
		return nil, fmt.Errorf("invalid Pocket TTS latent cache admission limits")
	}
	base := filepath.Base(manifestPath)
	if !strings.HasSuffix(base, "_latents.jsonl") {
		return nil, fmt.Errorf("Pocket TTS latent manifest must end in _latents.jsonl")
	}
	metadataPath := strings.TrimSuffix(manifestPath, ".jsonl") + ".meta.json"
	metadata, err := readTrainingLatentMetadata(metadataPath, expectedMimiHash)
	if err != nil {
		return nil, err
	}
	entries, err := loadLatentTrainingManifest(manifestPath, limits.MaxEntries)
	if err != nil {
		return nil, err
	}
	root := filepath.Dir(manifestPath)
	rootReal, err := filepath.EvalSymlinks(root)
	if err != nil {
		return nil, fmt.Errorf("resolve Pocket TTS latent root: %w", err)
	}
	sourceStem := strings.TrimSuffix(base, "_latents.jsonl")
	tag := metadata.MimiHash[:8]
	shards := make([]TrainingLatentShard, len(entries))
	totalElements := 0
	for i := range entries {
		expectedRelative := filepath.ToSlash(filepath.Join("latents", tag, fmt.Sprintf("%s_%08d.safetensors", sourceStem, i)))
		if entries[i].LatentsFile != expectedRelative {
			return nil, fmt.Errorf("Pocket TTS latent manifest row %d shard=%q want=%q", i+1, entries[i].LatentsFile, expectedRelative)
		}
		path := filepath.Join(root, filepath.FromSlash(expectedRelative))
		resolved, resolveErr := filepath.EvalSymlinks(path)
		if resolveErr != nil {
			return nil, fmt.Errorf("resolve Pocket TTS latent shard row %d: %w", i+1, resolveErr)
		}
		rel, relErr := filepath.Rel(rootReal, resolved)
		if relErr != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return nil, fmt.Errorf("Pocket TTS latent shard row %d escapes manifest root", i+1)
		}
		file, openErr := safetensors.Open(resolved)
		if openErr != nil {
			return nil, fmt.Errorf("open Pocket TTS latent shard row %d: %w", i+1, openErr)
		}
		infos := file.TensorInfos()
		info, ok := infos["latents"]
		if len(infos) != 1 || !ok || info.DType != "F32" || len(info.Shape) != 2 || info.Shape[0] <= 0 || info.Shape[0] > limits.MaxFramesPerRow || info.Shape[1] != latentChannels {
			file.Close()
			return nil, fmt.Errorf("invalid Pocket TTS latent shard row %d tensors=%v", i+1, infos)
		}
		elements, ok := checked.MulInt(info.Shape[0], latentChannels)
		if !ok {
			file.Close()
			return nil, fmt.Errorf("Pocket TTS latent shard row %d shape overflows", i+1)
		}
		totalElements, ok = checked.AddInt(totalElements, elements)
		if !ok || totalElements > limits.MaxTotalElements {
			file.Close()
			return nil, fmt.Errorf("Pocket TTS latent cache exceeds %d elements", limits.MaxTotalElements)
		}
		raw, dtype, shape, rawErr := file.GetRaw("latents")
		if rawErr != nil || dtype != "F32" || len(shape) != 2 {
			file.Close()
			return nil, fmt.Errorf("read Pocket TTS latent shard row %d: %w", i+1, rawErr)
		}
		digest := sha256.Sum256(raw)
		if closeErr := file.Close(); closeErr != nil {
			return nil, fmt.Errorf("close Pocket TTS latent shard row %d: %w", i+1, closeErr)
		}
		shards[i] = TrainingLatentShard{Path: resolved, Frames: info.Shape[0], Channels: latentChannels, SHA256: digest}
	}
	return &TrainingLatentCache{manifestPath: manifestPath, metadata: metadata, entries: entries, shards: shards}, nil
}

// LoadRow returns one cache row as owned F32 values. Admission deliberately
// rejects BF16 and other dtypes; upstream's pinned Mimi training config is F32.
func (c *TrainingLatentCache) LoadRow(index int) ([]float32, int, int, error) {
	if c == nil || index < 0 || index >= len(c.shards) {
		return nil, 0, 0, fmt.Errorf("invalid Pocket TTS latent cache row %d", index)
	}
	shard := c.shards[index]
	file, err := safetensors.Open(shard.Path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer file.Close()
	infos := file.TensorInfos()
	info, ok := infos["latents"]
	if len(infos) != 1 || !ok || info.DType != "F32" || len(info.Shape) != 2 || info.Shape[0] != shard.Frames || info.Shape[1] != shard.Channels {
		return nil, 0, 0, fmt.Errorf("Pocket TTS latent cache row %d changed after admission", index)
	}
	raw, dtype, rawShape, rawErr := file.GetRaw("latents")
	if rawErr != nil || dtype != "F32" || len(rawShape) != 2 || sha256.Sum256(raw) != shard.SHA256 {
		return nil, 0, 0, fmt.Errorf("Pocket TTS latent cache row %d changed after admission", index)
	}
	values, shape, err := file.GetFloat32("latents")
	if err != nil || len(shape) != 2 || shape[0] != shard.Frames || shape[1] != shard.Channels || len(values) != shard.Frames*shard.Channels {
		return nil, 0, 0, fmt.Errorf("Pocket TTS latent cache row %d changed after admission", index)
	}
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, 0, 0, fmt.Errorf("Pocket TTS latent cache row %d is non-finite", index)
		}
	}
	return values, shard.Frames, shard.Channels, nil
}

// StitchTrainingLatentTarget reproduces upstream latent mode: freshly encode
// the fixed padded overlap at the selected cut, discard the corresponding
// cached frames, and append the untouched cached tail. targetFrames is the
// valid-mask length; output can be longer when targetFrames < stitchFrames.
func StitchTrainingLatentTarget(stored []float32, storedFrames, channels, cutFrames, targetFrames, stitchFrames int, freshPrefix []float32) ([]float32, error) {
	if storedFrames <= 0 || channels <= 0 || cutFrames < 0 || targetFrames <= 0 || stitchFrames <= 0 || cutFrames > storedFrames || targetFrames > storedFrames-cutFrames {
		return nil, fmt.Errorf("invalid Pocket TTS latent stitch geometry")
	}
	storedElements, ok := checked.MulInt(storedFrames, channels)
	if !ok || len(stored) != storedElements {
		return nil, fmt.Errorf("invalid Pocket TTS stored latent shape")
	}
	validStitchFrames := min(stitchFrames, targetFrames)
	freshElements, ok := checked.MulInt(stitchFrames, channels)
	if !ok || len(freshPrefix) != freshElements {
		return nil, fmt.Errorf("invalid Pocket TTS fresh latent prefix shape")
	}
	outFrames, ok := checked.AddInt(stitchFrames, targetFrames-validStitchFrames)
	if !ok {
		return nil, fmt.Errorf("Pocket TTS stitched latent shape overflows")
	}
	outElements, ok := checked.MulInt(outFrames, channels)
	if !ok {
		return nil, fmt.Errorf("Pocket TTS stitched latent shape overflows")
	}
	out := make([]float32, outElements)
	copy(out, freshPrefix)
	tailStart := (cutFrames + validStitchFrames) * channels
	tailEnd := (cutFrames + targetFrames) * channels
	copy(out[freshElements:], stored[tailStart:tailEnd])
	return out, nil
}

// SaveTrainingLatentShard writes the deterministic native interchange form:
// one F32 tensor named "latents" with row-major [frames,channels] layout. The
// temporary file and containing directory are synced before publication.
func SaveTrainingLatentShard(path string, values []float32, frames, channels int) error {
	elements, ok := checked.MulInt(frames, channels)
	if !ok || frames <= 0 || channels <= 0 || len(values) != elements {
		return fmt.Errorf("invalid Pocket TTS latent shard shape")
	}
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("Pocket TTS latent shard is non-finite")
		}
	}
	byteLength, ok := checked.MulInt(elements, 4)
	if !ok {
		return fmt.Errorf("Pocket TTS latent shard byte size overflows")
	}
	header := []byte(fmt.Sprintf(`{"latents":{"dtype":"F32","shape":[%d,%d],"data_offsets":[0,%d]}}`, frames, channels, byteLength))
	if padding := (8 - len(header)%8) % 8; padding != 0 {
		header = append(header, bytes.Repeat([]byte{' '}, padding)...)
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	file, err := os.CreateTemp(dir, ".pockettts-latents-*.tmp")
	if err != nil {
		return err
	}
	tmp := file.Name()
	defer os.Remove(tmp)
	var size [8]byte
	binary.LittleEndian.PutUint64(size[:], uint64(len(header)))
	if _, err = file.Write(size[:]); err == nil {
		_, err = file.Write(header)
	}
	var raw [4]byte
	for _, value := range values {
		if err != nil {
			break
		}
		binary.LittleEndian.PutUint32(raw[:], math.Float32bits(value))
		_, err = file.Write(raw[:])
	}
	if err == nil {
		err = file.Sync()
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	if err = os.Rename(tmp, path); err != nil {
		return err
	}
	directory, err := os.Open(dir)
	if err != nil {
		return err
	}
	if err = directory.Sync(); err != nil {
		directory.Close()
		return err
	}
	return directory.Close()
}
