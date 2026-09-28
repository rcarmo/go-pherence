package pockettts

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

const pocketExportHeaderLimit = 16 << 20

type pocketExportTensor struct {
	name   string
	dtype  string
	shape  []int
	values []float32
	raw    []byte
}

type pocketExportHeaderEntry struct {
	DType       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// PocketExportPublishedError means the atomic rename succeeded but the final
// directory fsync could not confirm crash durability. Path contains the new
// complete export; this is deliberately distinct from pre-publication errors.
type PocketExportPublishedError struct {
	Path string
	Err  error
}

func (e *PocketExportPublishedError) Error() string {
	return fmt.Sprintf("Pocket TTS export %s was published but directory sync failed: %v", e.Path, e.Err)
}
func (e *PocketExportPublishedError) Unwrap() error { return e.Err }

// ExportPocketSafetensors writes the exact single-file format emitted by
// upstream export_pocket_safetensors: FlowLM parameters/buffers under flow_lm.*
// plus frozen Mimi tensors under mimi.*. The training-only flow.w_s_t network
// is excluded. When useEMA is true, EMA shadows replace only tracked FlowLM
// parameters; buffers and Mimi remain live/frozen values.
func ExportPocketSafetensors(path string, cfg Config, trainer *FullTrainer, frozen *safetensors.File, useEMA bool) error {
	if err := cfg.Validate(); err != nil {
		return err
	}
	if trainer == nil || frozen == nil {
		return fmt.Errorf("Pocket TTS export requires trainer and frozen Mimi source")
	}
	if err := trainer.refreshParameterBindings(); err != nil {
		return err
	}
	flow, err := pocketFlowExportTensors(cfg, trainer, useEMA)
	if err != nil {
		return err
	}
	mimi, err := pocketMimiExportTensors(cfg, frozen)
	if err != nil {
		return err
	}
	tensors := append(flow, mimi...)
	sort.Slice(tensors, func(i, j int) bool { return tensors[i].name < tensors[j].name })
	for i := 1; i < len(tensors); i++ {
		if tensors[i-1].name == tensors[i].name {
			return fmt.Errorf("duplicate Pocket TTS export tensor %q", tensors[i].name)
		}
	}
	return writePocketExport(path, tensors)
}

func pocketFlowExportTensors(cfg Config, trainer *FullTrainer, useEMA bool) ([]pocketExportTensor, error) {
	if trainer.FlowLM == nil || trainer.Flow == nil {
		return nil, fmt.Errorf("Pocket TTS export has no FlowLM")
	}
	shapes := expectedFlowExportShapes(cfg)
	params, err := fullParameterMap(trainer.FlowLM, trainer.Flow, trainer.Weighting)
	if err != nil {
		return nil, err
	}
	buffers := fullBufferMap(trainer.FlowLM, trainer.Flow)
	values := make(map[string][]float32, len(shapes))
	if useEMA && (trainer.EMADecay <= 0 || trainer.EMA == nil) {
		return nil, fmt.Errorf("Pocket TTS export requested missing EMA")
	}
	for name, value := range params {
		if !strings.HasPrefix(name, "flow_lm.") {
			continue
		}
		selected := value
		if useEMA {
			if shadow, ok := trainer.EMA[name]; ok {
				if len(shadow) != len(value) {
					return nil, fmt.Errorf("Pocket TTS export EMA shape mismatch for %q", name)
				}
				selected = shadow
			}
		}
		values[name] = selected
	}
	for name, value := range buffers {
		values[name] = value
	}
	if len(values) != len(shapes) {
		return nil, fmt.Errorf("Pocket TTS FlowLM export tensors=%d want=%d", len(values), len(shapes))
	}
	out := make([]pocketExportTensor, 0, len(shapes))
	for name, shape := range shapes {
		value, ok := values[name]
		elements, shapeOK := exportShapeElements(shape)
		if !ok || !shapeOK || len(value) != elements || !finiteF32(value) {
			return nil, fmt.Errorf("invalid Pocket TTS FlowLM export tensor %q", name)
		}
		out = append(out, pocketExportTensor{name: name, dtype: "F32", shape: append([]int(nil), shape...), values: value})
	}
	return out, nil
}

func pocketMimiExportTensors(cfg Config, frozen *safetensors.File) ([]pocketExportTensor, error) {
	infos, expected := frozen.TensorInfos(), expectedMimiStateShapes(cfg)
	mimiCount := 0
	for name := range infos {
		if strings.HasPrefix(name, "mimi.") {
			mimiCount++
		}
	}
	if mimiCount != len(expected) {
		return nil, fmt.Errorf("Pocket TTS frozen Mimi tensors=%d want=%d", mimiCount, len(expected))
	}
	out := make([]pocketExportTensor, 0, len(expected))
	for name, shape := range expected {
		info, ok := infos[name]
		if !ok || !equalShape(info.Shape, shape) || (info.DType != "F32" && info.DType != "BF16") {
			return nil, fmt.Errorf("Pocket TTS frozen Mimi source missing or mismatching %q", name)
		}
		values, got, err := frozen.GetFloat32(name)
		if err != nil || !equalShape(got, shape) || !finiteF32(values) {
			return nil, fmt.Errorf("invalid Pocket TTS frozen Mimi tensor %q", name)
		}
		out = append(out, pocketExportTensor{name: name, dtype: "F32", shape: append([]int(nil), shape...), values: values})
	}
	return out, nil
}

func expectedFlowExportShapes(cfg Config) map[string][]int {
	h, c, f := cfg.FlowLM.Transformer.DModel, cfg.Mimi.InnerDim, cfg.FlowLM.Flow.Dim
	ff := h * cfg.FlowLM.Transformer.HiddenScale
	out := map[string][]int{
		"flow_lm.conditioner.embed.weight": {cfg.FlowLM.LookupTable.NBins + 1, h},
		"flow_lm.bos_emb":                  {c}, "flow_lm.bos_before_voice": {1, 1, h},
		"flow_lm.emb_mean": {c}, "flow_lm.emb_std": {c},
		"flow_lm.speaker_proj_weight": {h, c}, "flow_lm.input_linear.weight": {h, c},
		"flow_lm.out_eos.weight": {1, h}, "flow_lm.out_eos.bias": {1},
		"flow_lm.out_norm.weight": {h}, "flow_lm.out_norm.bias": {h},
		"flow_lm.flow_net.input_proj.weight": {f, c}, "flow_lm.flow_net.input_proj.bias": {f},
		"flow_lm.flow_net.cond_embed.weight": {f, h}, "flow_lm.flow_net.cond_embed.bias": {f},
		"flow_lm.flow_net.final_layer.linear.weight": {c, f}, "flow_lm.flow_net.final_layer.linear.bias": {c},
		"flow_lm.flow_net.final_layer.adaLN_modulation.1.weight": {2 * f, f}, "flow_lm.flow_net.final_layer.adaLN_modulation.1.bias": {2 * f},
	}
	for i := 0; i < cfg.FlowLM.Transformer.NumLayers; i++ {
		p := fmt.Sprintf("flow_lm.transformer.layers.%d.", i)
		out[p+"norm1.weight"], out[p+"norm1.bias"] = []int{h}, []int{h}
		out[p+"norm2.weight"], out[p+"norm2.bias"] = []int{h}, []int{h}
		out[p+"self_attn.in_proj.weight"], out[p+"self_attn.out_proj.weight"] = []int{3 * h, h}, []int{h, h}
		out[p+"linear1.weight"], out[p+"linear2.weight"] = []int{ff, h}, []int{h, ff}

	}
	timeConditions := 2
	if cfg.FlowLM.Flow.Type == "flow_matching" {
		timeConditions = 1
	}
	for i := 0; i < timeConditions; i++ {
		p := fmt.Sprintf("flow_lm.flow_net.time_embed.%d.", i)
		out[p+"freqs"] = []int{128}
		out[p+"mlp.0.weight"], out[p+"mlp.0.bias"] = []int{f, 256}, []int{f}
		out[p+"mlp.2.weight"], out[p+"mlp.2.bias"] = []int{f, f}, []int{f}
		out[p+"mlp.3.alpha"] = []int{f}
	}
	for i := 0; i < cfg.FlowLM.Flow.Depth; i++ {
		p := fmt.Sprintf("flow_lm.flow_net.res_blocks.%d.", i)
		out[p+"in_ln.weight"], out[p+"in_ln.bias"] = []int{f}, []int{f}
		out[p+"mlp.0.weight"], out[p+"mlp.0.bias"] = []int{f, f}, []int{f}
		out[p+"mlp.2.weight"], out[p+"mlp.2.bias"] = []int{f, f}, []int{f}
		out[p+"adaLN_modulation.1.weight"], out[p+"adaLN_modulation.1.bias"] = []int{3 * f, f}, []int{3 * f}
	}
	return out
}

func expectedMimiStateShapes(cfg Config) map[string][]int {
	m, h, ff := cfg.Mimi.Transformer.DModel, cfg.Mimi.InnerDim, cfg.Mimi.Transformer.DimFeedforward
	out := map[string][]int{}
	conv := func(name string, outChannels, inChannels, kernel int, bias bool) {
		out["mimi."+name+".weight"] = []int{outChannels, inChannels, kernel}
		if bias {
			out["mimi."+name+".bias"] = []int{outChannels}
		}
	}
	// Encoder module list: first conv, residual+downsample per reversed ratio, final conv.
	channels := cfg.Mimi.SEANet.NFilters
	conv("encoder.model.0.conv", channels, cfg.Mimi.Channels, cfg.Mimi.SEANet.KernelSize, true)
	index := 1
	for i := len(cfg.Mimi.SEANet.Ratios) - 1; i >= 0; i-- {
		for r := 0; r < cfg.Mimi.SEANet.NResidualLayers; r++ {
			hidden := channels / cfg.Mimi.SEANet.Compress
			conv(fmt.Sprintf("encoder.model.%d.block.1.conv", index), hidden, channels, cfg.Mimi.SEANet.ResidualKernelSize, true)
			conv(fmt.Sprintf("encoder.model.%d.block.3.conv", index), channels, hidden, 1, true)
			index++
		}
		index++
		ratio := cfg.Mimi.SEANet.Ratios[i]
		conv(fmt.Sprintf("encoder.model.%d.conv", index), 2*channels, channels, 2*ratio, true)
		channels *= 2
		index++
	}
	index++
	conv(fmt.Sprintf("encoder.model.%d.conv", index), cfg.Mimi.SEANet.Dimension, channels, cfg.Mimi.SEANet.LastKernelSize, true)
	// Decoder module list mirrors configured ratio order.
	channels = cfg.Mimi.SEANet.NFilters * (1 << len(cfg.Mimi.SEANet.Ratios))
	conv("decoder.model.0.conv", channels, cfg.Mimi.SEANet.Dimension, cfg.Mimi.SEANet.KernelSize, true)
	index = 2
	for _, ratio := range cfg.Mimi.SEANet.Ratios {
		next := channels / 2
		name := fmt.Sprintf("decoder.model.%d.convtr", index)
		conv(name, channels, next, 2*ratio, true)
		out["mimi."+name+".bias"] = []int{next}
		index++
		for r := 0; r < cfg.Mimi.SEANet.NResidualLayers; r++ {
			hidden := next / cfg.Mimi.SEANet.Compress
			conv(fmt.Sprintf("decoder.model.%d.block.1.conv", index), hidden, next, cfg.Mimi.SEANet.ResidualKernelSize, true)
			conv(fmt.Sprintf("decoder.model.%d.block.3.conv", index), next, hidden, 1, true)
			index++
		}
		index++
		channels = next
	}
	conv(fmt.Sprintf("decoder.model.%d.conv", index), cfg.Mimi.Channels, channels, cfg.Mimi.SEANet.LastKernelSize, true)
	for _, side := range []string{"encoder", "decoder"} {
		for i := 0; i < cfg.Mimi.Transformer.NumLayers; i++ {
			p := fmt.Sprintf("mimi.%s_transformer.transformer.layers.%d.", side, i)
			out[p+"self_attn.in_proj.weight"] = []int{3 * m, m}
			out[p+"self_attn.out_proj.weight"] = []int{m, m}
			out[p+"norm1.weight"], out[p+"norm1.bias"] = []int{m}, []int{m}
			out[p+"norm2.weight"], out[p+"norm2.bias"] = []int{m}, []int{m}
			out[p+"linear1.weight"], out[p+"linear2.weight"] = []int{ff, m}, []int{m, ff}
			out[p+"layer_scale_1.scale"], out[p+"layer_scale_2.scale"] = []int{m}, []int{m}
		}
	}
	out["mimi.quantizer.output_proj.weight"] = []int{cfg.Mimi.OuterDim, h, 1}
	out["mimi.downsample.conv.conv.weight"] = []int{h, cfg.Mimi.SEANet.Dimension, 32}
	out["mimi.upsample.convtr.convtr.weight"] = []int{cfg.Mimi.SEANet.Dimension, 1, 32}
	return out
}

func writePocketExport(path string, tensors []pocketExportTensor) error {
	return writePocketExportWithSync(path, tensors, func(directory *os.File) error { return directory.Sync() })
}

func writePocketExportWithSync(path string, tensors []pocketExportTensor, syncDirectory func(*os.File) error) error {
	if syncDirectory == nil {
		return fmt.Errorf("Pocket TTS export directory sync callback is nil")
	}
	entries := make(map[string]pocketExportHeaderEntry, len(tensors))
	var offset int64
	for _, tensor := range tensors {
		if tensor.name == "" || tensor.name == "__metadata__" {
			return fmt.Errorf("invalid Pocket TTS export tensor name %q", tensor.name)
		}
		elements, ok := exportShapeElements(tensor.shape)
		if !ok || elements <= 0 {
			return fmt.Errorf("invalid Pocket TTS export shape for %q", tensor.name)
		}
		bytesPer := 4
		if tensor.dtype == "BF16" {
			bytesPer = 2
		} else if tensor.dtype != "F32" {
			return fmt.Errorf("unsupported Pocket TTS export dtype %s", tensor.dtype)
		}
		byteLen, ok := shapeInt64Mul(int64(elements), int64(bytesPer))
		if !ok {
			return fmt.Errorf("Pocket TTS export tensor %q overflows", tensor.name)
		}
		end, ok := shapeInt64Add(offset, byteLen)
		if !ok {
			return fmt.Errorf("Pocket TTS export offsets overflow")
		}
		if (tensor.values != nil && int64(len(tensor.values))*4 != byteLen) || (tensor.raw != nil && int64(len(tensor.raw)) != byteLen) || ((tensor.values == nil) == (tensor.raw == nil)) {
			return fmt.Errorf("invalid Pocket TTS export payload %q", tensor.name)
		}
		entries[tensor.name] = pocketExportHeaderEntry{DType: tensor.dtype, Shape: tensor.shape, DataOffsets: [2]int64{offset, end}}
		offset = end
	}
	var header bytes.Buffer
	header.WriteByte('{')
	for i, tensor := range tensors {
		if i > 0 {
			header.WriteByte(',')
		}
		key, _ := json.Marshal(tensor.name)
		value, err := json.Marshal(entries[tensor.name])
		if err != nil {
			return err
		}
		header.Write(key)
		header.WriteByte(':')
		header.Write(value)
	}
	header.WriteByte('}')
	for header.Len()%8 != 0 {
		header.WriteByte(' ')
	}
	if header.Len() > pocketExportHeaderLimit {
		return fmt.Errorf("Pocket TTS export header exceeds %d bytes", pocketExportHeaderLimit)
	}
	dir := filepath.Dir(path)
	directory, err := os.Open(dir)
	if err != nil {
		return fmt.Errorf("open existing Pocket TTS export directory: %w", err)
	}
	defer directory.Close()
	stat, err := directory.Stat()
	if err != nil || !stat.IsDir() {
		return fmt.Errorf("Pocket TTS export parent is not a directory")
	}
	if err = syncDirectory(directory); err != nil {
		return fmt.Errorf("sync Pocket TTS export directory before publication: %w", err)
	}
	file, err := os.CreateTemp(dir, ".pockettts-export-*.tmp")
	if err != nil {
		return err
	}
	tmp := file.Name()
	defer os.Remove(tmp)
	writer := bufio.NewWriterSize(file, 1<<20)
	var length [8]byte
	binary.LittleEndian.PutUint64(length[:], uint64(header.Len()))
	if _, err = writer.Write(length[:]); err == nil {
		_, err = writer.Write(header.Bytes())
	}
	chunk := make([]byte, 4096*4)
	for _, tensor := range tensors {
		if err != nil {
			break
		}
		if tensor.raw != nil {
			_, err = writer.Write(tensor.raw)
			continue
		}
		for off := 0; off < len(tensor.values) && err == nil; {
			n := min(4096, len(tensor.values)-off)
			for i := 0; i < n; i++ {
				binary.LittleEndian.PutUint32(chunk[i*4:], math.Float32bits(tensor.values[off+i]))
			}
			_, err = writer.Write(chunk[:n*4])
			off += n
		}
	}
	if err == nil {
		err = writer.Flush()
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
	if err = syncDirectory(directory); err != nil {
		return &PocketExportPublishedError{Path: path, Err: err}
	}
	return nil
}

func exportShapeElements(shape []int) (int, bool) {
	if len(shape) == 0 {
		return 0, false
	}
	elements := 1
	for _, dim := range shape {
		var ok bool
		elements, ok = checkedMulPositive(elements, dim)
		if !ok {
			return 0, false
		}
	}
	return elements, true
}
func checkedMulPositive(a, b int) (int, bool) {
	if a <= 0 || b <= 0 || a > math.MaxInt/b {
		return 0, false
	}
	return a * b, true
}
func finiteRawTensor(dtype string, raw []byte) bool {
	switch dtype {
	case "F32":
		if len(raw)%4 != 0 {
			return false
		}
		for i := 0; i < len(raw); i += 4 {
			bits := binary.LittleEndian.Uint32(raw[i:])
			if bits&0x7f800000 == 0x7f800000 {
				return false
			}
		}
	case "BF16":
		if len(raw)%2 != 0 {
			return false
		}
		for i := 0; i < len(raw); i += 2 {
			bits := binary.LittleEndian.Uint16(raw[i:])
			if bits&0x7f80 == 0x7f80 {
				return false
			}
		}
	default:
		return false
	}
	return true
}
