package whisper

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/audio/media"
)

func q8DequantRows(src []float32, rows, cols int) ([]float32, uint64) {
	out := make([]float32, len(src))
	for row := 0; row < rows; row++ {
		base := row * cols
		maximum := float32(0)
		for _, value := range src[base : base+cols] {
			maximum = float32(math.Max(float64(maximum), math.Abs(float64(value))))
		}
		scale := maximum / 127
		inverse := float32(0)
		if scale != 0 {
			inverse = 1 / scale
		}
		for col, value := range src[base : base+cols] {
			q := int(math.RoundToEven(float64(value * inverse)))
			q = min(127, max(-127, q))
			out[base+col] = float32(q) * scale
		}
	}
	return out, uint64((len(src)+3)/4*4 + rows*4)
}

func q8ProjectionEncoder(src *Encoder) (*Encoder, uint64, uint64) {
	out := *src
	out.Layers = append([]EncoderLayer(nil), src.Layers...)
	var q8Bytes, f32Bytes uint64
	for i := range out.Layers {
		s, d := &src.Layers[i], src.cfg.EncoderDModel
		o := &out.Layers[i]
		for _, spec := range []struct {
			dst        *[]float32
			in         []float32
			rows, cols int
		}{
			{&o.QWeight, s.QWeight, d, d}, {&o.KWeight, s.KWeight, d, d}, {&o.VWeight, s.VWeight, d, d}, {&o.OWeight, s.OWeight, d, d},
			{&o.FC1Weight, s.FC1Weight, src.cfg.EncoderFFNDim, d}, {&o.FC2Weight, s.FC2Weight, d, src.cfg.EncoderFFNDim},
		} {
			var bytes uint64
			*spec.dst, bytes = q8DequantRows(spec.in, spec.rows, spec.cols)
			q8Bytes += bytes
			f32Bytes += uint64(len(spec.in) * 4)
		}
	}
	return &out, q8Bytes, f32Bytes
}

func TestVulkanWhisperQ8Weight(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_WHISPER_Q8_WEIGHT") != "1" {
		t.Skip("explicit trained Whisper Q8-weight qualification required")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 2*time.Minute {
		t.Fatal("Whisper Q8-weight qualification requires timeout<=2m")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 110*time.Second)
	defer cancel()
	model, tok, policy := pinnedTinySpeechModel(t, ctx)
	dequantEncoder, qbytes, fbytes := q8ProjectionEncoder(model.Encoder)
	if !vk.VulkanInit() {
		t.Fatal("Vulkan unavailable")
	}
	device := vk.VulkanDeviceName()
	wantDevice := os.Getenv("GO_PHERENCE_VULKAN_DEVICE")
	lower := strings.ToLower(device)
	if wantDevice == "" || !strings.Contains(device, wantDevice) || strings.Contains(lower, "llvmpipe") || strings.Contains(lower, "lavapipe") {
		t.Fatal("unexpected physical device", device)
	}
	before := vk.VulkanMemoryStats()
	if before.Allocations != 0 || before.Bytes != 0 {
		t.Fatal("isolated process required", before)
	}
	defer func() {
		after := vk.VulkanMemoryStats()
		if after.Allocations != before.Allocations || after.Bytes != before.Bytes {
			t.Error("Q8 encoder leak", before, after)
		}
	}()
	const frames = 34
	mel := make([]float32, model.Config.NumMelBins*frames)
	for i := range mel {
		mel[i] = float32(math.Sin(float64(i)*.023) * .5)
	}
	base, e := NewVulkanEncoder(ctx, model.Encoder, frames)
	if e != nil {
		t.Fatal(e)
	}
	defer base.Close()
	dequant, e := NewVulkanEncoder(ctx, dequantEncoder, frames)
	if e != nil {
		t.Fatal(e)
	}
	defer dequant.Close()
	quant, e := NewVulkanEncoderQ8Weight(ctx, model.Encoder, frames)
	if e != nil {
		t.Fatal(e)
	}
	defer quant.Close()
	bo, e := base.Forward(ctx, mel)
	if e != nil {
		t.Fatal(e)
	}
	do, e := dequant.Forward(ctx, mel)
	if e != nil {
		t.Fatal(e)
	}
	qo, e := quant.Forward(ctx, mel)
	if e != nil {
		t.Fatal(e)
	}
	kernelMax := 0.0
	for i, v := range qo {
		kernelMax = math.Max(kernelMax, math.Abs(float64(v-do[i])))
	}
	mx, rms, ref := 0., 0., 0.
	for i, v := range qo {
		d := float64(v - bo[i])
		mx = math.Max(mx, math.Abs(d))
		rms += d * d
		ref += float64(bo[i]) * float64(bo[i])
	}
	rms = math.Sqrt(rms / float64(len(qo)))
	ref = math.Sqrt(ref / float64(len(qo)))
	bs, e := NewDecoderStateContext(ctx, model.Config, bo, frames/2, model.Decoder)
	if e != nil {
		t.Fatal(e)
	}
	qs, e := NewDecoderStateContext(ctx, model.Config, qo, frames/2, model.Decoder)
	if e != nil {
		t.Fatal(e)
	}
	top := func(a []float32) int {
		j := 0
		for i, v := range a {
			if v > a[j] {
				j = i
			}
		}
		return j
	}
	steps := []map[string]any{}
	for i, token := range []int{50258, 50259, 50359} {
		b := append([]float32(nil), model.Decoder.ForwardToken(token, bs)...)
		q := model.Decoder.ForwardToken(token, qs)
		bm, qm := top(b), top(q)
		md := 0.
		for j, v := range q {
			md = math.Max(md, math.Abs(float64(v-b[j])))
		}
		steps = append(steps, map[string]any{"step": i, "baseline_top": bm, "q8_top": qm, "max_logit_abs": md})
		if bm != qm {
			t.Fatal("top1", i, bm, qm)
		}
	}
	record, _ := json.Marshal(map[string]any{"values": len(qo), "max_abs": mx, "rms": rms, "reference_rms": ref, "rms_ratio": rms / ref, "packed_vs_dequant_graph_max_abs": kernelMax, "q8_projection_bytes": qbytes, "f32_projection_bytes": fbytes, "steps": steps})
	t.Log("WHISPER_Q8_QUALITY " + string(record))
	if err := base.Close(); err != nil {
		t.Fatal(err)
	}
	if err := dequant.Close(); err != nil {
		t.Fatal(err)
	}
	if err := quant.Close(); err != nil {
		t.Fatal(err)
	}
	base, dequant, quant = nil, nil, nil

	input := filepath.Join("..", "..", "testdata", "jfk.wav")
	pinnedSpeechFile(t, input, "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e")
	reader, err := media.OpenCanonicalPCM(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	base, err = NewVulkanEncoder(ctx, model.Encoder, model.Config.MaxLength)
	if err != nil {
		t.Fatal(err)
	}
	defer base.Close()
	quant, err = NewVulkanEncoderQ8Weight(ctx, model.Encoder, model.Config.MaxLength)
	if err != nil {
		t.Fatal(err)
	}
	defer quant.Close()
	run := func(name string, enc *VulkanEncoder) ([]WindowTranscript, int64) {
		var windows []WindowTranscript
		start := time.Now()
		err := model.TranscribePCMWindows(ctx, reader, int64(reader.Timeline().Samples), tok, PCMTranscribeOptions{Language: "en", Generation: policy, MaxNewTokens: 96, VulkanEncoder: enc}, func(w WindowTranscript) error { windows = append(windows, w); return nil })
		ns := time.Since(start).Nanoseconds()
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("WHISPER_Q8_SAMPLE path=%s wall_ns=%d", name, ns)
		return windows, ns
	}
	bw, _ := run("f32-warm", base)
	qw, _ := run("q8-warm", quant)
	if !reflect.DeepEqual(bw, qw) {
		t.Fatalf("Q8 transcript changed\nbaseline=%+v\nq8=%+v", bw, qw)
	}
	text := ""
	for _, w := range qw {
		for _, s := range w.Segments {
			text += " " + s.Text
		}
	}
	const reference = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country"
	edits, words := speechFixtureWER(reference, strings.TrimSpace(text))
	if edits != 0 || words != 22 {
		t.Fatal("Q8 JFK WER", edits, words, text)
	}
	times := map[string][]int64{"f32": {}, "q8": {}}
	for block := 0; block < 2; block++ {
		order := []string{"f32", "q8", "q8", "f32"}
		if block == 1 {
			order = []string{"q8", "f32", "f32", "q8"}
		}
		for _, name := range order {
			enc := base
			if name == "q8" {
				enc = quant
			}
			windows, ns := run(name, enc)
			if !reflect.DeepEqual(windows, bw) {
				t.Fatal("timed transcript changed", name, block)
			}
			times[name] = append(times[name], ns)
		}
	}
	medians := map[string]int64{}
	for name, values := range times {
		sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
		medians[name] = (values[1] + values[2]) / 2
	}
	result, _ := json.Marshal(map[string]any{"device": device, "windows": len(qw), "exact_tokens_timestamps": true, "word_edits": edits, "reference_words": words, "text": strings.TrimSpace(text), "baseline_stats": base.Stats(), "q8_stats": quant.Stats(), "median_ns": medians, "speedup": float64(medians["f32"]) / float64(medians["q8"]), "samples_per_path": 4})
	t.Log("WHISPER_Q8_JFK " + string(result))
}
