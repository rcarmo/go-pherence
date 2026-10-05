//go:build linux

package main

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
	"sort"
	"strconv"

	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/model/pockettts"
)

// Metadata-only safetensors admission: bounded header, no mmap or tensor decode.
func inspectModel(path string, limit int64, cfg pockettts.Config) (asset, error) {
	infos, provenance, err := inspectModelStructure(path, limit)
	if err != nil {
		return asset{}, err
	}
	if inv := pockettts.InspectTensorInfos(cfg, infos); !inv.CheckpointReady() {
		return asset{}, fmt.Errorf("model inventory invalid: missing=%v issues=%v", inv.Missing, inv.Issues)
	}
	return provenance, nil
}

// Kept separate for tiny real-file structural tests. Production always adds
// Pocket inventory checking above; no seam bypasses this reader in production.
func inspectModelStructure(path string, limit int64) (map[string]safetensors.TensorInfo, asset, error) {
	f, size, err := boundedFile(path, limit)
	if err != nil {
		return nil, asset{}, err
	}
	defer f.Close()
	var prefix [8]byte
	if _, err = io.ReadFull(f, prefix[:]); err != nil {
		return nil, asset{}, err
	}
	n := binary.LittleEndian.Uint64(prefix[:])
	if n == 0 || n > 1<<20 || size < 8 || n > uint64(size-8) {
		return nil, asset{}, fmt.Errorf("model header exceeds 1MiB/file bound")
	}
	header := make([]byte, int(n))
	if _, err = io.ReadFull(f, header); err != nil {
		return nil, asset{}, err
	}
	infos, err := parseModelHeader(header, size-8-int64(n))
	if err != nil {
		return nil, asset{}, err
	}
	if _, err = f.Seek(0, io.SeekStart); err != nil {
		return nil, asset{}, err
	}
	h := sha256.New()
	read, err := io.Copy(h, io.LimitReader(f, limit+1))
	if err != nil {
		return nil, asset{}, err
	}
	if read != size {
		return nil, asset{}, fmt.Errorf("model size changed while hashing")
	}
	return infos, asset{hex.EncodeToString(h.Sum(nil)), size}, nil
}

// strictObject rejects duplicate keys instead of silently keeping the last.
func strictObject(data []byte) (map[string]json.RawMessage, error) {
	d := json.NewDecoder(bytes.NewReader(data))
	first, err := d.Token()
	if err != nil || first != json.Delim('{') {
		return nil, fmt.Errorf("expected JSON object")
	}
	out := map[string]json.RawMessage{}
	for d.More() {
		key, err := d.Token()
		if err != nil {
			return nil, err
		}
		name, ok := key.(string)
		if !ok {
			return nil, fmt.Errorf("invalid object key")
		}
		if _, ok := out[name]; ok {
			return nil, fmt.Errorf("duplicate header key %q", name)
		}
		var value json.RawMessage
		if err := d.Decode(&value); err != nil {
			return nil, err
		}
		out[name] = value
		if len(out) > 512 {
			return nil, fmt.Errorf("header object entry limit exceeded")
		}
	}
	if _, err = d.Token(); err != nil {
		return nil, err
	}
	var extra any
	if err = d.Decode(&extra); err != io.EOF {
		return nil, fmt.Errorf("trailing header JSON")
	}
	return out, nil
}

func parseModelHeader(header []byte, dataBytes int64) (map[string]safetensors.TensorInfo, error) {
	raw, err := strictObject(header)
	if err != nil {
		return nil, err
	}
	infos := make(map[string]safetensors.TensorInfo, len(raw))
	type interval struct{ start, end int64 }
	spans := make([]interval, 0, len(raw))
	for name, value := range raw {
		if name == "__metadata__" {
			meta, err := strictObject(value)
			if err != nil {
				return nil, fmt.Errorf("metadata: %w", err)
			}
			for _, v := range meta {
				if _, err := strictString(v); err != nil {
					return nil, fmt.Errorf("metadata: %w", err)
				}
			}
			continue
		}
		fields, err := strictObject(value)
		if err != nil {
			return nil, err
		}
		if len(fields) != 3 || fields["dtype"] == nil || fields["shape"] == nil || fields["data_offsets"] == nil {
			return nil, fmt.Errorf("invalid tensor fields")
		}
		dtype, err := strictString(fields["dtype"])
		if err != nil {
			return nil, fmt.Errorf("dtype: %w", err)
		}
		shape, err := strictInts(fields["shape"])
		if err != nil {
			return nil, fmt.Errorf("shape: %w", err)
		}
		offsets, err := strictInts(fields["data_offsets"])
		if err != nil {
			return nil, fmt.Errorf("offsets: %w", err)
		}
		if name == "" || len(name) > 256 || len(shape) == 0 || len(shape) > 8 || len(offsets) != 2 {
			return nil, fmt.Errorf("invalid tensor name/rank/offset count")
		}
		width := int64(0)
		switch dtype {
		case "BF16":
			width = 2
		case "F32":
			width = 4
		default:
			return nil, fmt.Errorf("unsupported tensor dtype %q", dtype)
		}
		length := width
		for _, dim := range shape {
			if dim <= 0 || int64(dim) > math.MaxInt64/length {
				return nil, fmt.Errorf("tensor shape overflow")
			}
			length *= int64(dim)
		}
		start, end := int64(offsets[0]), int64(offsets[1])
		if start < 0 || end < start || end > dataBytes || end-start != length {
			return nil, fmt.Errorf("tensor %q offset/shape mismatch", name)
		}
		infos[name] = safetensors.TensorInfo{DType: dtype, Shape: shape, DataOffsets: [2]int{offsets[0], offsets[1]}}
		spans = append(spans, interval{start, end})
	}
	sort.Slice(spans, func(i, j int) bool { return spans[i].start < spans[j].start })
	end := int64(0)
	for _, span := range spans {
		if span.start != end {
			return nil, fmt.Errorf("tensor payload gap/overlap")
		}
		end = span.end
	}
	if len(spans) == 0 || end != dataBytes {
		return nil, fmt.Errorf("tensor payload not fully covered")
	}
	return infos, nil
}

func strictString(raw []byte) (string, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || raw[0] != '"' {
		return "", fmt.Errorf("expected non-null JSON string")
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", err
	}
	return value, nil
}
func strictInts(raw []byte) ([]int, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || raw[0] != '[' {
		return nil, fmt.Errorf("expected non-null integer array")
	}
	var elements []json.RawMessage
	if err := json.Unmarshal(raw, &elements); err != nil {
		return nil, err
	}
	if len(elements) > 8 {
		return nil, fmt.Errorf("integer array exceeds rank/offset bound")
	}
	values := make([]int, len(elements))
	for i, element := range elements {
		s := string(bytes.TrimSpace(element))
		if len(s) == 0 {
			return nil, fmt.Errorf("expected integer syntax")
		}
		for j, c := range s {
			if (c < '0' || c > '9') && !(j == 0 && c == '-') {
				return nil, fmt.Errorf("expected integer syntax")
			}
		}
		n, err := strconv.ParseInt(s, 10, strconv.IntSize)
		if err != nil {
			return nil, fmt.Errorf("integer range: %w", err)
		}
		if n < 0 {
			return nil, fmt.Errorf("negative integer geometry")
		}
		values[i] = int(n)
	}
	return values, nil
}

// Deliberately narrow PCM16/F32 WAV contract: fixed RIFF/fmt/data header,
// mono 24kHz, no implicit resampling/downmixing. Stream validation in bounded
// chunks avoids trusting forged data lengths for decoded-buffer allocation.
func scanWAV(f *os.File, maxSamples int) (int, asset, error) {
	st, err := f.Stat()
	if err != nil {
		return 0, asset{}, fmt.Errorf("stat WAV: %w", err)
	}
	maxBytes := int64(maxSamples)*4 + 44
	if !st.Mode().IsRegular() || st.Size() < 44 || st.Size() > maxBytes {
		return 0, asset{}, fmt.Errorf("WAV must be a regular file within sample/byte ceiling")
	}
	var head [44]byte
	if _, err = io.ReadFull(f, head[:]); err != nil {
		return 0, asset{}, err
	}
	if string(head[:4]) != "RIFF" || string(head[8:12]) != "WAVE" || string(head[12:16]) != "fmt " || binary.LittleEndian.Uint32(head[16:20]) != 16 || string(head[36:40]) != "data" {
		return 0, asset{}, fmt.Errorf("require canonical 44-byte mono 24kHz PCM16/F32 WAV header")
	}
	format := binary.LittleEndian.Uint16(head[20:22])
	bits := binary.LittleEndian.Uint16(head[34:36])
	width := int64(bits / 8)
	if binary.LittleEndian.Uint16(head[22:24]) != 1 || binary.LittleEndian.Uint32(head[24:28]) != 24000 || !(format == 1 && bits == 16 || format == 3 && bits == 32) || binary.LittleEndian.Uint16(head[32:34]) != uint16(width) || binary.LittleEndian.Uint32(head[28:32]) != 24000*uint32(width) {
		return 0, asset{}, fmt.Errorf("unsupported WAV format/rate/channels/alignment")
	}
	payload := int64(binary.LittleEndian.Uint32(head[40:44]))
	samples := payload / width
	if payload == 0 || payload%width != 0 || samples > int64(maxSamples) || samples > int64(int(^uint(0)>>1)) || payload+44 != st.Size() || int64(binary.LittleEndian.Uint32(head[4:8])) != st.Size()-8 {
		return 0, asset{}, fmt.Errorf("WAV declared sample count/size exceeds ceiling or file")
	}
	h := sha256.New()
	h.Write(head[:])
	var buf [32 << 10]byte
	left := payload
	for left > 0 {
		n := int(min(int64(len(buf)), left))
		if _, err = io.ReadFull(f, buf[:n]); err != nil {
			return 0, asset{}, err
		}
		h.Write(buf[:n])
		for off := 0; off < n; off += int(width) {
			v := float64(0)
			if format == 1 {
				raw := int16(binary.LittleEndian.Uint16(buf[off:]))
				if raw == math.MinInt16 || raw == math.MaxInt16 {
					return 0, asset{}, fmt.Errorf("WAV has full-scale/clipping sample")
				}
				v = float64(raw) / 32768
			} else {
				v = float64(math.Float32frombits(binary.LittleEndian.Uint32(buf[off:])))
			}
			if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) >= 1 {
				return 0, asset{}, fmt.Errorf("WAV nonfinite/out-of-range/full-scale sample")
			}
		}
		left -= int64(n)
	}
	return int(samples), asset{hex.EncodeToString(h.Sum(nil)), st.Size()}, nil
}

func planRow(cfg pockettts.Config, tok *pockettts.Tokenizer, e pockettts.TrainingEntry, samples, maxTokens int, maxResident int64) (rowReport, error) {
	var r rowReport
	if !isFinite64(e.Start) || !isFinite64(e.Duration) || e.Start < 0 || e.Duration <= 0 {
		return r, fmt.Errorf("invalid segment duration/start")
	}
	start, err := timestampSample(e.Start, samples)
	if err != nil {
		return r, err
	}
	end, err := timestampSample(e.Start+e.Duration, samples)
	if err != nil || end <= start {
		return r, fmt.Errorf("manifest segment exceeds recording duration or is empty")
	}
	cuts := e.EligibleTrainingCuts(1)
	if len(cuts) == 0 {
		return r, fmt.Errorf("no eligible training cut")
	}
	cut := cuts[0]
	if len(cut.TargetTranscript) > 4096 {
		return r, fmt.Errorf("target transcript exceeds 4096 bytes")
	}
	ids, err := tok.Encode(cut.TargetTranscript)
	if err != nil {
		return r, err
	}
	if len(ids) == 0 || len(ids) > maxTokens {
		return r, fmt.Errorf("target tokenizer count exceeds limit")
	}
	cutSample, err := timestampSample(e.Start+cut.Seconds, samples)
	if err != nil || cutSample <= start || cutSample >= end {
		return r, fmt.Errorf("cut is outside segment")
	}
	// Match Mimi's zero-padding geometry exactly, never floor fractional frames.
	voice := (cutSample - start + pockettts.SamplesPerFrame - 1) / pockettts.SamplesPerFrame
	target := (end - cutSample + pockettts.SamplesPerFrame - 1) / pockettts.SamplesPerFrame
	if voice > pockettts.ProductionTrainingVoiceFrames || target > pockettts.ProductionTrainingTargetFrames {
		return r, fmt.Errorf("sample-exact prompt/target exceeds 62/375-frame envelope; no implicit trim")
	}
	shape := pockettts.TrainingShape{MicroBatchRows: 1, TargetFrames: target, VoiceFrames: voice, TextTokens: len(ids), GradientAccumulation: 1, FlowBatchMultiplier: 1}
	limits := pockettts.ProductionTrainingShapeLimits(maxResident)
	limits.MaxTextTokens = maxTokens
	plan, err := pockettts.PlanTrainingShape(cfg, shape, limits)
	if err != nil {
		return r, err
	}
	return rowReport{Samples: samples, CutSeconds: cut.Seconds, SegmentStartSample: start, CutSample: cutSample, SegmentEndSample: end, TextTokens: len(ids), TargetFrames: target, VoiceFrames: voice, SequenceRows: plan.SequenceRows, ResidentUpperBytes: plan.ResidentUpperBytes}, nil
}

func isFinite64(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
func timestampSample(seconds float64, maxSamples int) (int, error) {
	if !isFinite64(seconds) || seconds < 0 || seconds > float64(maxSamples)/24000 {
		return 0, fmt.Errorf("timestamp exceeds recording sample bound")
	}
	return int(math.Round(seconds * 24000)), nil
}
