package kv

import (
	"fmt"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// LayerF32KVConfig describes one per-layer float32 K/V cache shape.
type LayerF32KVConfig struct {
	Dim           int
	Sliding       bool
	SlidingWindow int
}

// LayeredF32KVCheckpoint captures a restorable multi-layer K/V state.
type LayeredF32KVCheckpoint struct {
	valid           bool
	maxPrefillChunk int
	configs         []LayerF32KVConfig
	layers          []F32KVCheckpoint
}

// LayeredF32KV manages one float32 K/V store per layer, mixing full-history
// linear layers with sliding-window ring layers.
type LayeredF32KV struct {
	configs         []LayerF32KVConfig
	stores          []F32KVStore
	maxPrefillChunk int
}

// NewLayeredF32KV builds one K/V store per layer configuration.
func NewLayeredF32KV(configs []LayerF32KVConfig, maxPrefillChunk int) (*LayeredF32KV, error) {
	if err := validateLayerF32KVConfigs(configs, maxPrefillChunk); err != nil {
		return nil, err
	}
	m := &LayeredF32KV{
		configs:         append([]LayerF32KVConfig(nil), configs...),
		stores:          make([]F32KVStore, len(configs)),
		maxPrefillChunk: maxPrefillChunk,
	}
	for i, cfg := range configs {
		if cfg.Sliding {
			capacity, _ := layerF32KVCapacity(cfg, maxPrefillChunk)
			m.stores[i] = NewRingF32KV(cfg.Dim, capacity)
			continue
		}
		m.stores[i] = NewLinearF32KV(cfg.Dim)
	}
	if err := m.Validate(); err != nil {
		return nil, err
	}
	return m, nil
}

// Append appends one K/V row to a specific layer.
func (m *LayeredF32KV) Append(layer int, k, v []float32) error {
	store, err := m.storeForLayer(layer)
	if err != nil {
		return err
	}
	return store.Append(k, v)
}

// View returns a logical oldest-to-newest view for a specific layer.
func (m *LayeredF32KV) View(layer int) (F32KVView, error) {
	store, err := m.storeForLayer(layer)
	if err != nil {
		return F32KVView{}, err
	}
	if err := m.validateLayerStore(layer, store); err != nil {
		return F32KVView{}, err
	}
	return store.View(), nil
}

// MaterializeLayer returns a contiguous K/V snapshot for a specific layer.
func (m *LayeredF32KV) MaterializeLayer(layer int) (k, v []float32, startToken int, err error) {
	store, err := m.storeForLayer(layer)
	if err != nil {
		return nil, nil, 0, err
	}
	if err := m.validateLayerStore(layer, store); err != nil {
		return nil, nil, 0, err
	}
	k, v, startToken = store.Materialize()
	return k, v, startToken, nil
}

// Reset clears every layer.
func (m *LayeredF32KV) Reset() {
	if m == nil {
		return
	}
	for _, store := range m.stores {
		if store != nil {
			store.Reset()
		}
	}
}

// Bytes returns the current logical K/V bytes across all layers.
func (m *LayeredF32KV) Bytes() int64 {
	if m == nil {
		return 0
	}
	var total int64
	for _, store := range m.stores {
		if store == nil {
			continue
		}
		total = checked.SaturatingAddInt64(total, store.Bytes())
	}
	return total
}

// Checkpoint captures all layers in restorable logical order.
func (m *LayeredF32KV) Checkpoint() LayeredF32KVCheckpoint {
	if m == nil || m.Validate() != nil {
		return LayeredF32KVCheckpoint{}
	}
	cp := LayeredF32KVCheckpoint{
		valid:           true,
		maxPrefillChunk: m.maxPrefillChunk,
		configs:         append([]LayerF32KVConfig(nil), m.configs...),
		layers:          make([]F32KVCheckpoint, len(m.stores)),
	}
	for i, store := range m.stores {
		cp.layers[i] = store.Checkpoint()
		if !cp.layers[i].valid {
			return LayeredF32KVCheckpoint{}
		}
	}
	return cp
}

// Restore restores every layer from a checkpoint.
func (m *LayeredF32KV) Restore(cp LayeredF32KVCheckpoint) error {
	if err := m.validateRestoreCheckpoint(cp); err != nil {
		return err
	}
	return m.restoreLayers(cp)
}

// KeepAppended restores the checkpoint and replays the first keepTokens rows
// appended after it for every layer.
func (m *LayeredF32KV) KeepAppended(cp LayeredF32KVCheckpoint, keepTokens int) error {
	if keepTokens < 0 {
		return fmt.Errorf("keepTokens=%d must be >= 0", keepTokens)
	}
	if err := m.validateRestoreCheckpoint(cp); err != nil {
		return err
	}
	rollback := m.Checkpoint()
	if !rollback.valid {
		return fmt.Errorf("failed to checkpoint current layered KV state")
	}
	rollbackOnError := func(err error) error {
		if restoreErr := m.restoreLayers(rollback); restoreErr != nil {
			return fmt.Errorf("keep appended failed: %v (rollback failed: %w)", err, restoreErr)
		}
		return err
	}

	type keptRows struct {
		k []float32
		v []float32
	}
	kept := make([]keptRows, len(m.stores))
	stagedTokens := -1
	for i, store := range m.stores {
		layerCP := cp.layers[i]
		currentK, currentV, currentStart := store.Materialize()
		currentTokens := store.Tokens()

		checkpointEnd, ok := checked.AddInt(layerCP.startToken, layerCP.tokens)
		if !ok {
			return fmt.Errorf("layer %d checkpoint end token overflow", i)
		}
		currentEnd, ok := checked.AddInt(currentStart, currentTokens)
		if !ok {
			return fmt.Errorf("layer %d current end token overflow", i)
		}
		if currentEnd < checkpointEnd {
			return fmt.Errorf("layer %d current end token=%d shorter than checkpoint end=%d", i, currentEnd, checkpointEnd)
		}
		layerStagedTokens := currentEnd - checkpointEnd
		if stagedTokens < 0 {
			stagedTokens = layerStagedTokens
		} else if layerStagedTokens != stagedTokens {
			return fmt.Errorf("layer %d staged tokens=%d want %d", i, layerStagedTokens, stagedTokens)
		}
		if keepTokens > layerStagedTokens {
			return fmt.Errorf("layer %d staged tokens=%d shorter than keepTokens=%d", i, layerStagedTokens, keepTokens)
		}
		if keepTokens == 0 {
			continue
		}
		if currentStart > checkpointEnd {
			return fmt.Errorf("layer %d checkpoint prefix no longer recoverable", i)
		}

		overlapRows := checkpointEnd - currentStart
		if overlapRows > 0 {
			if overlapRows > currentTokens {
				return fmt.Errorf("layer %d checkpoint overlap rows=%d exceed current tokens=%d", i, overlapRows, currentTokens)
			}
			cpOffsetRows := currentStart - layerCP.startToken
			if cpOffsetRows < 0 {
				return fmt.Errorf("layer %d current start token=%d precedes checkpoint start=%d", i, currentStart, layerCP.startToken)
			}
			wantK, ok := rowSlice(layerCP.k, layerCP.dim, cpOffsetRows, overlapRows)
			if !ok {
				return fmt.Errorf("layer %d checkpoint K overlap out of range", i)
			}
			wantV, ok := rowSlice(layerCP.v, layerCP.dim, cpOffsetRows, overlapRows)
			if !ok {
				return fmt.Errorf("layer %d checkpoint V overlap out of range", i)
			}
			gotK, ok := rowSlice(currentK, layerCP.dim, 0, overlapRows)
			if !ok {
				return fmt.Errorf("layer %d current K overlap out of range", i)
			}
			gotV, ok := rowSlice(currentV, layerCP.dim, 0, overlapRows)
			if !ok {
				return fmt.Errorf("layer %d current V overlap out of range", i)
			}
			if !equalFloat32Slices(gotK, wantK) || !equalFloat32Slices(gotV, wantV) {
				return fmt.Errorf("layer %d checkpoint prefix mismatch", i)
			}
		}

		keepStartRows := checkpointEnd - currentStart
		keepK, ok := rowSlice(currentK, layerCP.dim, keepStartRows, keepTokens)
		if !ok {
			return fmt.Errorf("layer %d kept K rows out of range", i)
		}
		keepV, ok := rowSlice(currentV, layerCP.dim, keepStartRows, keepTokens)
		if !ok {
			return fmt.Errorf("layer %d kept V rows out of range", i)
		}
		kept[i].k = append([]float32(nil), keepK...)
		kept[i].v = append([]float32(nil), keepV...)
	}

	if err := m.restoreLayers(cp); err != nil {
		return rollbackOnError(err)
	}
	for i := range kept {
		dim := m.configs[i].Dim
		for row := 0; row < keepTokens; row++ {
			start := row * dim
			end := start + dim
			if err := m.stores[i].Append(kept[i].k[start:end], kept[i].v[start:end]); err != nil {
				return rollbackOnError(fmt.Errorf("layer %d: %w", i, err))
			}
		}
	}
	return nil
}

// Validate verifies the manager, per-layer configs, and underlying stores.
func (m *LayeredF32KV) Validate() error {
	if m == nil {
		return fmt.Errorf("nil LayeredF32KV")
	}
	if err := validateLayerF32KVConfigs(m.configs, m.maxPrefillChunk); err != nil {
		return err
	}
	if len(m.stores) != len(m.configs) {
		return fmt.Errorf("layered KV stores=%d want %d", len(m.stores), len(m.configs))
	}
	for i, store := range m.stores {
		if err := m.validateLayerStore(i, store); err != nil {
			return err
		}
	}
	return nil
}

// EstimateLayeredF32KVBytes estimates full-history linear bytes and sliding-ring
// bytes for a mixed per-layer cache shape.
func EstimateLayeredF32KVBytes(configs []LayerF32KVConfig, maxContext, maxChunk int) (linearBytes, ringBytes int64, err error) {
	if maxContext < 0 {
		return 0, 0, fmt.Errorf("maxContext=%d must be >= 0", maxContext)
	}
	if err := validateLayerF32KVConfigs(configs, maxChunk); err != nil {
		return 0, 0, err
	}
	for i, cfg := range configs {
		if cfg.Sliding {
			capacity, _ := layerF32KVCapacity(cfg, maxChunk)
			bytes, err := exactF32KVBytes(capacity, cfg.Dim)
			if err != nil {
				return 0, 0, fmt.Errorf("layer %d ring estimate: %w", i, err)
			}
			ringBytes = checked.SaturatingAddInt64(ringBytes, bytes)
			continue
		}
		bytes, err := exactF32KVBytes(maxContext, cfg.Dim)
		if err != nil {
			return 0, 0, fmt.Errorf("layer %d linear estimate: %w", i, err)
		}
		linearBytes = checked.SaturatingAddInt64(linearBytes, bytes)
	}
	return linearBytes, ringBytes, nil
}

func (m *LayeredF32KV) storeForLayer(layer int) (F32KVStore, error) {
	if m == nil {
		return nil, fmt.Errorf("nil LayeredF32KV")
	}
	if layer < 0 || layer >= len(m.stores) {
		return nil, fmt.Errorf("layer %d out of range [0,%d)", layer, len(m.stores))
	}
	store := m.stores[layer]
	if store == nil {
		return nil, fmt.Errorf("layer %d has nil store", layer)
	}
	return store, nil
}

func (m *LayeredF32KV) validateLayerStore(layer int, store F32KVStore) error {
	cfg := m.configs[layer]
	if store == nil {
		return fmt.Errorf("layer %d has nil store", layer)
	}
	if store.Dim() != cfg.Dim {
		return fmt.Errorf("layer %d dim=%d want %d", layer, store.Dim(), cfg.Dim)
	}
	if cfg.Sliding {
		expectedCapacity, _ := layerF32KVCapacity(cfg, m.maxPrefillChunk)
		ring, ok := store.(*RingF32KV)
		if !ok {
			return fmt.Errorf("layer %d expected RingF32KV", layer)
		}
		if ring.Capacity() != expectedCapacity {
			return fmt.Errorf("layer %d capacity=%d want %d", layer, ring.Capacity(), expectedCapacity)
		}
		if err := ring.validateState(); err != nil {
			return fmt.Errorf("layer %d: %w", layer, err)
		}
		return nil
	}
	linear, ok := store.(*LinearF32KV)
	if !ok {
		return fmt.Errorf("layer %d expected LinearF32KV", layer)
	}
	if linear.Capacity() != 0 {
		return fmt.Errorf("layer %d linear capacity=%d want 0", layer, linear.Capacity())
	}
	if err := linear.validateState(); err != nil {
		return fmt.Errorf("layer %d: %w", layer, err)
	}
	return nil
}

func (m *LayeredF32KV) validateLayerCheckpoint(layer int, cp F32KVCheckpoint) error {
	cfg := m.configs[layer]
	if _, err := validateF32KVCheckpoint(cp); err != nil {
		return fmt.Errorf("layer %d: %w", layer, err)
	}
	expectedCapacity, _ := layerF32KVCapacity(cfg, m.maxPrefillChunk)
	if cp.dim != cfg.Dim {
		return fmt.Errorf("layer %d checkpoint dim=%d want %d", layer, cp.dim, cfg.Dim)
	}
	if cp.capacity != expectedCapacity {
		return fmt.Errorf("layer %d checkpoint capacity=%d want %d", layer, cp.capacity, expectedCapacity)
	}
	return nil
}

func (m *LayeredF32KV) validateRestoreCheckpoint(cp LayeredF32KVCheckpoint) error {
	if m == nil {
		return fmt.Errorf("nil LayeredF32KV")
	}
	if err := m.Validate(); err != nil {
		return err
	}
	if err := validateLayeredF32KVCheckpoint(cp); err != nil {
		return err
	}
	if cp.maxPrefillChunk != m.maxPrefillChunk {
		return fmt.Errorf("layered KV checkpoint maxPrefillChunk=%d want %d", cp.maxPrefillChunk, m.maxPrefillChunk)
	}
	if len(cp.configs) != len(m.configs) {
		return fmt.Errorf("layered KV checkpoint layers=%d want %d", len(cp.configs), len(m.configs))
	}
	for i := range m.configs {
		if m.configs[i] != cp.configs[i] {
			return fmt.Errorf("layered KV checkpoint config mismatch at layer %d", i)
		}
		if err := m.validateLayerCheckpoint(i, cp.layers[i]); err != nil {
			return err
		}
	}
	return nil
}

func (m *LayeredF32KV) restoreLayers(cp LayeredF32KVCheckpoint) error {
	for i, store := range m.stores {
		if err := store.Restore(cp.layers[i]); err != nil {
			return fmt.Errorf("layer %d: %w", i, err)
		}
	}
	return nil
}

func equalFloat32Slices(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func validateLayeredF32KVCheckpoint(cp LayeredF32KVCheckpoint) error {
	if !cp.valid {
		return fmt.Errorf("invalid layered checkpoint")
	}
	if err := validateLayerF32KVConfigs(cp.configs, cp.maxPrefillChunk); err != nil {
		return err
	}
	if len(cp.layers) != len(cp.configs) {
		return fmt.Errorf("layered checkpoint layers=%d want %d", len(cp.layers), len(cp.configs))
	}
	for i, layer := range cp.layers {
		if _, err := validateF32KVCheckpoint(layer); err != nil {
			return fmt.Errorf("layer %d: %w", i, err)
		}
		expectedCapacity, _ := layerF32KVCapacity(cp.configs[i], cp.maxPrefillChunk)
		if layer.dim != cp.configs[i].Dim {
			return fmt.Errorf("layer %d checkpoint dim=%d want %d", i, layer.dim, cp.configs[i].Dim)
		}
		if layer.capacity != expectedCapacity {
			return fmt.Errorf("layer %d checkpoint capacity=%d want %d", i, layer.capacity, expectedCapacity)
		}
	}
	return nil
}

func validateLayerF32KVConfigs(configs []LayerF32KVConfig, maxPrefillChunk int) error {
	if maxPrefillChunk < 0 {
		return fmt.Errorf("maxPrefillChunk=%d must be >= 0", maxPrefillChunk)
	}
	for i, cfg := range configs {
		if err := validateLayerF32KVConfig(cfg, maxPrefillChunk); err != nil {
			return fmt.Errorf("layer %d: %w", i, err)
		}
	}
	return nil
}

func validateLayerF32KVConfig(cfg LayerF32KVConfig, maxPrefillChunk int) error {
	if cfg.Dim <= 0 {
		return fmt.Errorf("dim=%d must be > 0", cfg.Dim)
	}
	if !cfg.Sliding {
		if cfg.SlidingWindow != 0 {
			return fmt.Errorf("non-sliding layer has slidingWindow=%d", cfg.SlidingWindow)
		}
		return nil
	}
	if cfg.SlidingWindow <= 0 {
		return fmt.Errorf("slidingWindow=%d must be > 0", cfg.SlidingWindow)
	}
	if _, ok := checked.AddInt(cfg.SlidingWindow, maxPrefillChunk); !ok {
		return fmt.Errorf("sliding window=%d chunk=%d overflow", cfg.SlidingWindow, maxPrefillChunk)
	}
	return nil
}

func layerF32KVCapacity(cfg LayerF32KVConfig, maxPrefillChunk int) (int, error) {
	if err := validateLayerF32KVConfig(cfg, maxPrefillChunk); err != nil {
		return 0, err
	}
	if !cfg.Sliding {
		return 0, nil
	}
	capacity, ok := checked.AddInt(cfg.SlidingWindow, maxPrefillChunk)
	if !ok {
		return 0, fmt.Errorf("sliding window=%d chunk=%d overflow", cfg.SlidingWindow, maxPrefillChunk)
	}
	return capacity, nil
}

func exactF32KVBytes(tokens, dim int) (int64, error) {
	if tokens < 0 || dim < 0 {
		return 0, fmt.Errorf("tokens=%d dim=%d must be >= 0", tokens, dim)
	}
	elems, ok := checked.MulInt(tokens, dim)
	if !ok {
		return 0, fmt.Errorf("tokens=%d dim=%d overflow", tokens, dim)
	}
	max := checked.MaxInt64()
	if int64(elems) > max/8 {
		return 0, fmt.Errorf("bytes overflow for tokens=%d dim=%d", tokens, dim)
	}
	return int64(elems) * 8, nil
}
