package promptcache

import (
	"encoding/binary"
	"fmt"
	"unicode/utf8"
)

// Identity describes all model- and runtime-specific properties that must
// match for prompt-cache reuse to be safe.
type Identity struct {
	ModelFingerprint      string
	CheckpointFingerprint string
	Backend               string
	WeightLayout          string
	WeightDType           string
	KVPolicy              string
	KVPrecision           string
	ConfigFingerprint     string
	RoPEFingerprint       string
	AdapterID             string
	MultimodalHash        string
	CacheSalt             string
}

func (id Identity) Validate() error {
	for _, field := range []struct {
		name     string
		value    string
		required bool
	}{
		{name: "model_fingerprint", value: id.ModelFingerprint, required: true},
		{name: "checkpoint_fingerprint", value: id.CheckpointFingerprint, required: true},
		{name: "backend", value: id.Backend, required: true},
		{name: "weight_layout", value: id.WeightLayout, required: true},
		{name: "weight_dtype", value: id.WeightDType, required: true},
		{name: "kv_policy", value: id.KVPolicy, required: true},
		{name: "kv_precision", value: id.KVPrecision, required: true},
		{name: "config_fingerprint", value: id.ConfigFingerprint, required: true},
		{name: "rope_fingerprint", value: id.RoPEFingerprint, required: true},
		{name: "adapter_id", value: id.AdapterID},
		{name: "multimodal_hash", value: id.MultimodalHash},
		{name: "cache_salt", value: id.CacheSalt},
	} {
		if field.required && field.value == "" {
			return fmt.Errorf("%w: %s is empty", ErrInvalidIdentity, field.name)
		}
		if !utf8.ValidString(field.value) {
			return fmt.Errorf("%w: %s is not valid UTF-8", ErrInvalidIdentity, field.name)
		}
	}
	return nil
}

func (id Identity) CanonicalBytes() ([]byte, error) {
	if err := id.Validate(); err != nil {
		return nil, err
	}
	fields := [...]string{
		id.ModelFingerprint,
		id.CheckpointFingerprint,
		id.Backend,
		id.WeightLayout,
		id.WeightDType,
		id.KVPolicy,
		id.KVPrecision,
		id.ConfigFingerprint,
		id.RoPEFingerprint,
		id.AdapterID,
		id.MultimodalHash,
		id.CacheSalt,
	}
	buf := make([]byte, 0, 32)
	buf = append(buf, "promptcache.identity.v1"...)
	for _, field := range fields {
		if len(field) > int(^uint32(0)) {
			return nil, fmt.Errorf("%w: field too large", ErrInvalidIdentity)
		}
		var raw [4]byte
		binary.BigEndian.PutUint32(raw[:], uint32(len(field)))
		buf = append(buf, raw[:]...)
		buf = append(buf, field...)
	}
	return buf, nil
}
