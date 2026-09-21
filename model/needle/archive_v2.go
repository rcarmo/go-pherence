package needle

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
)

// The v2 archive has no architecture metadata. Require explicit fields even
// where the source-checkpoint parser normally offers defaults.
func parseArchiveConfigV2(raw json.RawMessage) (Config, error) {
	var c Config
	if len(raw) == 0 || len(raw) > 64<<10 {
		return c, fmt.Errorf("needle: archive sidecar must be 1..65536 bytes")
	}
	node, err := decodeJSONValue(raw)
	if err != nil {
		return c, err
	}
	fields, ok := node.(map[string]any)
	if !ok {
		return c, fmt.Errorf("needle: archive sidecar must be a JSON object")
	}
	for _, key := range []string{"generation", "vocab_size", "d_model", "attn_dim", "num_heads", "num_kv_heads", "num_layers", "max_seq_len", "mhc_lanes", "engram_orders", "engram_heads", "engram_slots", "engram_layers", "rope_theta", "pad_token_id", "contrastive_dim"} {
		if value, present := fields[key]; !present || value == nil {
			return c, fmt.Errorf("needle: archive sidecar requires %s", key)
		}
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err = dec.Decode(&c); err != nil {
		return c, err
	}
	if err = dec.Decode(new(any)); err != io.EOF {
		return c, fmt.Errorf("needle: trailing sidecar data")
	}
	if c.VocabSize < 4 || c.DModel < 2 || c.Heads < 1 || c.KVHeads < 1 || c.Layers < 1 || c.MaxSeq < 1 || c.Lanes < 1 || c.EngramSlots < 1 {
		return c, fmt.Errorf("needle: sidecar geometry must be explicit and positive")
	}
	if c.Generation != 2 || c.ArchiveDecoded || c.ArchiveKVWindow != 0 || c.ConvTaps != 0 || len(c.GlobalLayers) != 0 || c.Window != 0 || len(c.LadderWidths) != 0 || len(c.LadderOrder) != 0 || c.EngramSeedHeads != 0 || c.QKDim != 0 || c.VDim != 0 || c.AttnDim < 1 || c.EngramHeads < 1 || len(c.EngramOrders) == 0 || c.ContrastiveDim < 1 || c.ContrastiveDim > 4096 || c.RopeTheta <= 0 || c.OutVocab != 0 && c.OutVocab != c.VocabSize {
		return c, fmt.Errorf("needle: incompatible Needle2 archive sidecar")
	}
	if err = c.validate(); err != nil {
		return c, err
	}
	return c, nil
}
