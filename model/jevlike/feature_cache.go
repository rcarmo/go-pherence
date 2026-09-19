package jevlike

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
	"sync"

	"github.com/rcarmo/go-pherence/half"
)

// FeatureContract identifies unprojected frozen features, never trainable keys or
// values. ModelID must include verified weights/config/tokenizer file identity.
type FeatureContract struct {
	Version        int    `json:"version"`
	ModelID        string `json:"model_id"`
	DatasetSHA256  string `json:"dataset_sha256"`
	Backend        string `json:"backend"`
	TokenPolicy    string `json:"token_policy"`
	Representation string `json:"representation"`
	Pooling        string `json:"pooling"`
	DType          string `json:"dtype"`
	Width          int    `json:"width"`
	ContextTokens  int    `json:"context_tokens"`
	OptionTokens   int    `json:"option_tokens"`
}

func (c FeatureContract) Validate() error {
	if c.Version != 1 || (!strings.Contains(c.ModelID, "#sha256=") || !featureHashValid(strings.SplitN(c.ModelID, "#sha256=", 2)[1])) || !featureHashValid(c.DatasetSHA256) || c.Backend == "" || c.TokenPolicy != "plain/no-bos/no-eos/reject-overlength" || c.Representation != "causal/final-rmsnorm/all-token-rows" || c.Pooling != "option-mean-f32-before-storage" || (c.DType != "f32" && c.DType != "f16") || c.Width < 1 || c.Width > 8192 || c.ContextTokens < 1 || c.ContextTokens > 512 || c.OptionTokens < 1 || c.OptionTokens > 512 {
		return fmt.Errorf("invalid feature contract")
	}
	return nil
}
func (c FeatureContract) ID() string { b, _ := json.Marshal(c); return featureHash(b) }
func featureHash(b []byte) string    { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func featureHashValid(s string) bool {
	b, e := hex.DecodeString(s)
	return e == nil && len(b) == 32 && hex.EncodeToString(b) == s
}

type FeatureCache struct {
	mu            sync.Mutex
	dir           string
	Contract      FeatureContract
	tokenize      func(string) ([]int, error)
	encode        func([]int) ([][]float32, error)
	writable      bool
	closed        bool
	budget, bytes int64
	Hits, Misses  int
}

// OpenFeatureCache checks the exact expected contract. A nil tokenize/encode
// pair opens cache-only mode: a missing entry is an error, never a GPU call.
// One exclusive writer owns a root; interrupted extraction retains immutable
// completed entries. A stale .writer.lock requires operator inspection/removal.
func OpenFeatureCache(dir string, expected FeatureContract, budget int64, tokenize func(string) ([]int, error), encode func([]int) ([][]float32, error)) (*FeatureCache, error) {
	if err := expected.Validate(); err != nil {
		return nil, err
	}
	if budget < 1 || budget > 12<<30 || (tokenize == nil) != (encode == nil) {
		return nil, fmt.Errorf("invalid cache budget/encoder")
	}
	writable := encode != nil
	if writable {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, err
		}
	}
	info, err := os.Lstat(dir)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return nil, fmt.Errorf("cache root must be a real directory")
	}
	c := &FeatureCache{dir: dir, Contract: expected, writable: writable, budget: budget, tokenize: tokenize, encode: encode}
	if writable {
		if err = os.Mkdir(filepath.Join(dir, ".writer.lock"), 0o700); err != nil {
			return nil, fmt.Errorf("cache writer lock: %w", err)
		}
	}
	fail := func(err error) (*FeatureCache, error) { c.Close(); return nil, err }
	path := filepath.Join(dir, "contract.json")
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) && writable {
		data, _ = json.Marshal(expected)
		if err = featurePublish(path, data); err != nil {
			return fail(err)
		}
	} else if err != nil {
		return fail(err)
	}
	var actual FeatureContract
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err = dec.Decode(&actual); err != nil {
		return fail(err)
	}
	if dec.Decode(new(any)) != io.EOF || actual != expected {
		return fail(fmt.Errorf("stale/incompatible feature contract"))
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return fail(err)
	}
	for _, entry := range entries {
		if entry.Name() == ".writer.lock" {
			continue
		}
		st, err := entry.Info()
		if err != nil {
			return fail(err)
		}
		if !st.Mode().IsRegular() {
			return fail(fmt.Errorf("unsupported cache entry %s", entry.Name()))
		}
		c.bytes += st.Size()
	}
	if c.bytes > budget {
		return fail(fmt.Errorf("existing cache exceeds budget"))
	}
	return c, nil
}
func LoadFeatureContract(dir string) (FeatureContract, error) {
	var c FeatureContract
	b, e := os.ReadFile(filepath.Join(dir, "contract.json"))
	if e != nil {
		return c, e
	}
	e = json.Unmarshal(b, &c)
	if e != nil {
		return c, e
	}
	return c, c.Validate()
}
func (c *FeatureCache) Close() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.closed && c.writable {
		_ = os.Remove(filepath.Join(c.dir, ".writer.lock"))
	}
	c.closed = true
}
func (c *FeatureCache) FeatureReference() string { return "feature:" + c.Contract.ID() }
func (c *FeatureCache) Bytes() int64             { c.mu.Lock(); defer c.mu.Unlock(); return c.bytes }
func (c *FeatureCache) Encode(text string, maxTokens int) ([][]float32, error) {
	return c.feature(text, maxTokens, false)
}
func (c *FeatureCache) EncodeOption(text string, maxTokens int) ([]float32, error) {
	rows, err := c.feature(text, maxTokens, true)
	if err != nil {
		return nil, err
	}
	return rows[0], nil
}

type featureHeader struct {
	Version       int    `json:"version"`
	Key           string `json:"key"`
	ContractID    string `json:"contract_id"`
	Text          string `json:"text"`
	MaxTokens     int    `json:"max_tokens"`
	Pooled        bool   `json:"pooled"`
	Tokens        []int  `json:"tokens"`
	Rows          int    `json:"rows"`
	Width         int    `json:"width"`
	DType         string `json:"dtype"`
	PayloadSHA256 string `json:"payload_sha256"`
}

func (c *FeatureCache) feature(text string, limit int, pooled bool) ([][]float32, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil, fmt.Errorf("cache closed")
	}
	want := c.Contract.ContextTokens
	if pooled {
		want = c.Contract.OptionTokens
	}
	if limit != want || text == "" {
		return nil, fmt.Errorf("feature limit/text differs from contract")
	}
	query := struct {
		Contract, Text string
		Limit          int
		Pooled         bool
	}{c.Contract.ID(), text, limit, pooled}
	q, _ := json.Marshal(query)
	key := featureHash(q)
	path := filepath.Join(c.dir, key+".jvf")
	if st, err := os.Lstat(path); err == nil {
		if !st.Mode().IsRegular() || st.Size() > (17<<20) {
			return nil, fmt.Errorf("invalid feature file")
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		rows, err := c.decodeFeature(data, key, text, limit, pooled)
		if err != nil {
			return nil, err
		}
		c.Hits++
		return rows, nil
	} else if !os.IsNotExist(err) {
		return nil, err
	}
	c.Misses++
	if !c.writable {
		return nil, fmt.Errorf("cache miss %s (offline; no fallback)", key)
	}
	ids, err := c.tokenize(text)
	if err != nil {
		return nil, err
	}
	if len(ids) == 0 || len(ids) > limit {
		return nil, fmt.Errorf("overlength/empty feature tokens=%d limit=%d; no truncation", len(ids), limit)
	}
	for _, id := range ids {
		if id < 0 {
			return nil, fmt.Errorf("negative token ID")
		}
	}
	size := len(ids) * c.Contract.Width * 4
	if pooled {
		size = c.Contract.Width * 4
	}
	if c.bytes+int64(size)+int64(len(q))+4096 > c.budget {
		return nil, fmt.Errorf("feature cache budget exceeded")
	}
	rows, err := c.encode(ids)
	if err != nil {
		return nil, err
	}
	if len(rows) != len(ids) {
		return nil, fmt.Errorf("encoder token row mismatch")
	}
	if err = validateFeatureRows(rows, c.Contract.Width); err != nil {
		return nil, err
	}
	if pooled {
		vector := make([]float32, c.Contract.Width)
		for _, row := range rows {
			for d, v := range row {
				vector[d] += v / float32(len(rows))
			}
		}
		rows = [][]float32{vector}
	}
	payload, err := featurePayload(rows, c.Contract.DType)
	if err != nil {
		return nil, err
	}
	header := featureHeader{1, key, c.Contract.ID(), text, limit, pooled, ids, len(rows), c.Contract.Width, c.Contract.DType, featureHash(payload)}
	data, err := packFeature(header, payload)
	if err != nil {
		return nil, err
	}
	if c.bytes+int64(len(data)) > c.budget {
		return nil, fmt.Errorf("feature cache budget exceeded")
	}
	if err = featurePublish(path, data); err != nil {
		return nil, err
	}
	c.bytes += int64(len(data))
	// Use exactly the representation future offline epochs will read.
	return c.decodeFeature(data, key, text, limit, pooled)
}
func validateFeatureRows(rows [][]float32, width int) error {
	if len(rows) < 1 || len(rows) > 512 {
		return fmt.Errorf("invalid feature row count")
	}
	for _, row := range rows {
		if len(row) != width {
			return fmt.Errorf("feature width mismatch")
		}
		for _, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("nonfinite features")
			}
		}
	}
	return nil
}
func featurePayload(rows [][]float32, dtype string) ([]byte, error) {
	size := 4
	if dtype == "f16" {
		size = 2
	}
	var out bytes.Buffer
	for _, row := range rows {
		for _, v := range row {
			var b [4]byte
			if size == 4 {
				binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
			} else {
				bits := half.F32ToF16(v)
				if math.IsInf(float64(half.F16ToF32(bits)), 0) {
					return nil, fmt.Errorf("FP16 feature overflow")
				}
				binary.LittleEndian.PutUint16(b[:], bits)
			}
			out.Write(b[:size])
		}
	}
	return out.Bytes(), nil
}
func (c *FeatureCache) decodeFeature(data []byte, key, text string, limit int, pooled bool) ([][]float32, error) {
	if len(data) < 40 || string(data[:4]) != "JVF1" {
		return nil, fmt.Errorf("invalid feature magic/size")
	}
	sum := sha256.Sum256(data[:len(data)-32])
	if !bytes.Equal(sum[:], data[len(data)-32:]) {
		return nil, fmt.Errorf("feature checksum mismatch")
	}
	n := int(binary.LittleEndian.Uint32(data[4:]))
	if n < 2 || n > 1<<20 || n > len(data)-40 {
		return nil, fmt.Errorf("invalid feature header size")
	}
	var h featureHeader
	d := json.NewDecoder(bytes.NewReader(data[8 : 8+n]))
	d.DisallowUnknownFields()
	if err := d.Decode(&h); err != nil {
		return nil, err
	}
	if d.Decode(new(any)) != io.EOF {
		return nil, fmt.Errorf("trailing feature metadata")
	}
	if h.Version != 1 || h.Key != key || h.ContractID != c.Contract.ID() || h.Text != text || h.MaxTokens != limit || h.Pooled != pooled || h.Width != c.Contract.Width || h.DType != c.Contract.DType || len(h.Tokens) < 1 || len(h.Tokens) > limit || h.Rows < 1 || h.Rows > limit || (pooled && h.Rows != 1) || (!pooled && h.Rows != len(h.Tokens)) {
		return nil, fmt.Errorf("feature identity/shape mismatch")
	}
	for _, id := range h.Tokens {
		if id < 0 {
			return nil, fmt.Errorf("invalid stored token")
		}
	}
	if c.tokenize != nil {
		ids, err := c.tokenize(text)
		if err != nil {
			return nil, err
		}
		b, _ := json.Marshal(ids)
		a, _ := json.Marshal(h.Tokens)
		if !bytes.Equal(a, b) {
			return nil, fmt.Errorf("feature tokenizer IDs changed")
		}
	}
	payload := data[8+n : len(data)-32]
	size := 4
	if h.DType == "f16" {
		size = 2
	}
	if len(payload) != h.Rows*h.Width*size || featureHash(payload) != h.PayloadSHA256 {
		return nil, fmt.Errorf("feature payload identity/length mismatch")
	}
	rows := make([][]float32, h.Rows)
	for i := range rows {
		rows[i] = make([]float32, h.Width)
		for j := range rows[i] {
			p := (i*h.Width + j) * size
			if size == 4 {
				rows[i][j] = math.Float32frombits(binary.LittleEndian.Uint32(payload[p:]))
			} else {
				rows[i][j] = half.F16ToF32(binary.LittleEndian.Uint16(payload[p:]))
			}
		}
	}
	return rows, validateFeatureRows(rows, h.Width)
}
func packFeature(header featureHeader, payload []byte) ([]byte, error) {
	meta, err := json.Marshal(header)
	if err != nil {
		return nil, err
	}
	if len(meta) > 1<<20 {
		return nil, fmt.Errorf("feature metadata exceeds bound")
	}
	data := make([]byte, 8, 8+len(meta)+len(payload)+32)
	copy(data, "JVF1")
	binary.LittleEndian.PutUint32(data[4:], uint32(len(meta)))
	data = append(data, meta...)
	data = append(data, payload...)
	sum := sha256.Sum256(data)
	return append(data, sum[:]...), nil
}

func featurePublish(path string, data []byte) error {
	f, err := os.CreateTemp(filepath.Dir(path), ".feature-*.tmp")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	if _, err = f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err = f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	// Hard-link publication is atomic and never overwrites a completed feature.
	if err = os.Link(f.Name(), path); err != nil {
		return err
	}
	dir, err := os.Open(filepath.Dir(path))
	if err != nil {
		return err
	}
	defer dir.Close()
	return dir.Sync()
}
