package promptcache

import (
	"bytes"
	"container/list"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"reflect"
	"sync"
)

var (
	ErrInvalidIdentity  = errors.New("promptcache: invalid identity")
	ErrInvalidBlockSize = errors.New("promptcache: invalid block size")
	ErrIncompleteBlock  = errors.New("promptcache: incomplete token block")
	ErrNilSnapshot      = errors.New("promptcache: nil snapshot")
	ErrPositionMismatch = errors.New("promptcache: snapshot position mismatch")
	ErrSizeOverflow     = errors.New("promptcache: size overflow")
	ErrOverBudget       = errors.New("promptcache: over budget")
)

type Snapshot interface {
	Position() int
	SizeBytes() (int64, error)
	Clone() Snapshot
}

type HashFunc func([]byte) [32]byte

type Option func(*Cache)

func WithHashFunc(fn HashFunc) Option {
	return func(c *Cache) {
		if fn != nil {
			c.hash = fn
		}
	}
}

type Stats struct {
	MaxBytes          int64  `json:"max_bytes"`
	UsedBytes         int64  `json:"used_bytes"`
	Entries           int    `json:"entries"`
	Puts              uint64 `json:"puts"`
	Lookups           uint64 `json:"lookups"`
	Hits              uint64 `json:"hits"`
	Misses            uint64 `json:"misses"`
	Replacements      uint64 `json:"replacements"`
	Evictions         uint64 `json:"evictions"`
	Collisions        uint64 `json:"collisions"`
	BudgetRejections  uint64 `json:"budget_rejections"`
	InvalidRejections uint64 `json:"invalid_rejections"`
}

type Cache struct {
	mu        sync.Mutex
	maxBytes  int64
	usedBytes int64
	hash      HashFunc
	ll        *list.List
	buckets   map[blockKey][]*list.Element
	stats     Stats
}

type blockKey struct {
	Digest    [32]byte
	BlockSize int
	EndPos    int
}

type entry struct {
	identityCanonical []byte
	blockSize         int
	endPos            int
	digest            [32]byte
	tokens            []int
	snapshot          Snapshot
	sizeBytes         int64
}

func New(maxBytes int64, opts ...Option) *Cache {
	if maxBytes < 0 {
		maxBytes = 0
	}
	c := &Cache{
		maxBytes: maxBytes,
		hash:     sha256.Sum256,
		ll:       list.New(),
		buckets:  map[blockKey][]*list.Element{},
	}
	c.stats.MaxBytes = maxBytes
	for _, opt := range opts {
		if opt != nil {
			opt(c)
		}
	}
	if c.hash == nil {
		c.hash = sha256.Sum256
	}
	return c
}

func (c *Cache) Put(identity Identity, blockSize int, tokens []int, snap Snapshot) error {
	identityCanonical, validatedSnap, sizeBytes, key, err := c.prepare(identity, blockSize, tokens, snap)
	if err != nil {
		if c != nil {
			c.mu.Lock()
			if errors.Is(err, ErrOverBudget) {
				c.stats.BudgetRejections++
			} else {
				c.stats.InvalidRejections++
			}
			c.mu.Unlock()
		}
		return err
	}
	if c == nil {
		return fmt.Errorf("promptcache: nil cache")
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	c.stats.Puts++

	var existing *list.Element
	for _, el := range c.buckets[key] {
		ent := el.Value.(*entry)
		if ent.matches(identityCanonical, tokens) {
			existing = el
			break
		}
		c.stats.Collisions++
	}

	if existing != nil {
		ent := existing.Value.(*entry)
		threshold := c.maxBytes - sizeBytes
		for c.usedBytes-ent.sizeBytes > threshold {
			if !c.evictOne(existing) {
				break
			}
		}
		baseUsed := c.usedBytes - ent.sizeBytes
		if baseUsed < 0 {
			baseUsed = 0
		}
		ent.identityCanonical = append(ent.identityCanonical[:0], identityCanonical...)
		ent.tokens = append(ent.tokens[:0], tokens...)
		ent.snapshot = validatedSnap
		ent.sizeBytes = sizeBytes
		ent.blockSize = blockSize
		ent.endPos = len(tokens)
		ent.digest = key.Digest
		c.usedBytes = baseUsed + sizeBytes
		c.ll.MoveToFront(existing)
		c.stats.UsedBytes = c.usedBytes
		c.stats.Entries = c.ll.Len()
		c.stats.Replacements++
		return nil
	}

	threshold := c.maxBytes - sizeBytes
	for c.usedBytes > threshold {
		if !c.evictOne(nil) {
			break
		}
	}
	ent := &entry{
		identityCanonical: append([]byte(nil), identityCanonical...),
		blockSize:         blockSize,
		endPos:            len(tokens),
		digest:            key.Digest,
		tokens:            append([]int(nil), tokens...),
		snapshot:          validatedSnap,
		sizeBytes:         sizeBytes,
	}
	el := c.ll.PushFront(ent)
	c.buckets[key] = append(c.buckets[key], el)
	c.usedBytes += sizeBytes
	c.stats.UsedBytes = c.usedBytes
	c.stats.Entries = c.ll.Len()
	return nil
}

func (c *Cache) FindLongest(identity Identity, blockSize int, tokens []int) (Snapshot, bool, error) {
	if c == nil {
		return nil, false, fmt.Errorf("promptcache: nil cache")
	}
	if blockSize <= 0 {
		c.mu.Lock()
		c.stats.InvalidRejections++
		c.mu.Unlock()
		return nil, false, ErrInvalidBlockSize
	}
	identityCanonical, err := identity.CanonicalBytes()
	if err != nil {
		c.mu.Lock()
		c.stats.InvalidRejections++
		c.mu.Unlock()
		return nil, false, err
	}
	keys := prefixKeys(identityCanonical, blockSize, tokens, c.hash)

	c.mu.Lock()
	defer c.mu.Unlock()
	c.stats.Lookups++

	for i := len(keys) - 1; i >= 0; i-- {
		key := keys[i]
		queryPrefix := tokens[:key.EndPos]
		for _, el := range c.buckets[key] {
			ent := el.Value.(*entry)
			if !ent.matches(identityCanonical, queryPrefix) {
				c.stats.Collisions++
				continue
			}
			clone, ok := cloneSnapshotForRead(ent.snapshot, ent.endPos)
			if !ok {
				continue
			}
			c.ll.MoveToFront(el)
			c.stats.Hits++
			return clone, true, nil
		}
	}
	c.stats.Misses++
	return nil, false, nil
}

func (c *Cache) Stats() Stats {
	if c == nil {
		return Stats{}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	st := c.stats
	st.MaxBytes = c.maxBytes
	st.UsedBytes = c.usedBytes
	st.Entries = c.ll.Len()
	return st
}

func (c *Cache) prepare(identity Identity, blockSize int, tokens []int, snap Snapshot) ([]byte, Snapshot, int64, blockKey, error) {
	if c == nil {
		return nil, nil, 0, blockKey{}, fmt.Errorf("promptcache: nil cache")
	}
	if blockSize <= 0 {
		return nil, nil, 0, blockKey{}, ErrInvalidBlockSize
	}
	if len(tokens) == 0 || len(tokens)%blockSize != 0 {
		return nil, nil, 0, blockKey{}, ErrIncompleteBlock
	}
	identityCanonical, err := identity.CanonicalBytes()
	if err != nil {
		return nil, nil, 0, blockKey{}, err
	}
	validatedSnap, sizeBytes, err := cloneSnapshotForStore(snap, len(tokens))
	if err != nil {
		return nil, nil, 0, blockKey{}, err
	}
	if c.maxBytes <= 0 || sizeBytes > c.maxBytes {
		return nil, nil, 0, blockKey{}, ErrOverBudget
	}
	keys := prefixKeys(identityCanonical, blockSize, tokens, c.hash)
	if len(keys) == 0 {
		return nil, nil, 0, blockKey{}, ErrIncompleteBlock
	}
	return identityCanonical, validatedSnap, sizeBytes, keys[len(keys)-1], nil
}

func (c *Cache) evictOne(skip *list.Element) bool {
	for el := c.ll.Back(); el != nil; el = el.Prev() {
		if el == skip {
			continue
		}
		ent := el.Value.(*entry)
		key := ent.key()
		c.removeBucketElement(key, el)
		c.ll.Remove(el)
		c.usedBytes -= ent.sizeBytes
		if c.usedBytes < 0 {
			c.usedBytes = 0
		}
		c.stats.Evictions++
		c.stats.UsedBytes = c.usedBytes
		c.stats.Entries = c.ll.Len()
		return true
	}
	return false
}

func (c *Cache) removeBucketElement(key blockKey, target *list.Element) {
	bucket := c.buckets[key]
	for i, el := range bucket {
		if el != target {
			continue
		}
		bucket = append(bucket[:i], bucket[i+1:]...)
		if len(bucket) == 0 {
			delete(c.buckets, key)
		} else {
			c.buckets[key] = bucket
		}
		return
	}
}

func (e *entry) key() blockKey {
	return blockKey{Digest: e.digest, BlockSize: e.blockSize, EndPos: e.endPos}
}

func (e *entry) matches(identityCanonical []byte, tokens []int) bool {
	return bytes.Equal(e.identityCanonical, identityCanonical) && equalTokens(e.tokens, tokens)
}

func cloneSnapshotForStore(snap Snapshot, expectedPos int) (Snapshot, int64, error) {
	if isNilSnapshot(snap) {
		return nil, 0, ErrNilSnapshot
	}
	if snap.Position() != expectedPos {
		return nil, 0, ErrPositionMismatch
	}
	clone := snap.Clone()
	if isNilSnapshot(clone) {
		return nil, 0, ErrNilSnapshot
	}
	if clone.Position() != expectedPos {
		return nil, 0, ErrPositionMismatch
	}
	sizeBytes, err := clone.SizeBytes()
	if err != nil {
		return nil, 0, err
	}
	if sizeBytes < 0 {
		return nil, 0, ErrSizeOverflow
	}
	return clone, sizeBytes, nil
}

func cloneSnapshotForRead(snap Snapshot, expectedPos int) (Snapshot, bool) {
	if isNilSnapshot(snap) || snap.Position() != expectedPos {
		return nil, false
	}
	clone := snap.Clone()
	if isNilSnapshot(clone) || clone.Position() != expectedPos {
		return nil, false
	}
	sizeBytes, err := clone.SizeBytes()
	if err != nil || sizeBytes < 0 {
		return nil, false
	}
	return clone, true
}

func isNilSnapshot(snap Snapshot) bool {
	if snap == nil {
		return true
	}
	v := reflect.ValueOf(snap)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	default:
		return false
	}
}

func prefixKeys(identityCanonical []byte, blockSize int, tokens []int, hash HashFunc) []blockKey {
	if blockSize <= 0 {
		return nil
	}
	full := len(tokens) / blockSize * blockSize
	if full == 0 {
		return nil
	}
	out := make([]blockKey, 0, full/blockSize)
	var prev [32]byte
	for end := blockSize; end <= full; end += blockSize {
		prev = hashBlock(hash, identityCanonical, prev, tokens[end-blockSize:end])
		out = append(out, blockKey{Digest: prev, BlockSize: blockSize, EndPos: end})
	}
	return out
}

func hashBlock(hash HashFunc, identityCanonical []byte, prev [32]byte, tokens []int) [32]byte {
	buf := make([]byte, 0, len(identityCanonical)+len(prev)+8*len(tokens))
	buf = append(buf, identityCanonical...)
	buf = append(buf, prev[:]...)
	var raw [8]byte
	for _, token := range tokens {
		binary.BigEndian.PutUint64(raw[:], uint64(int64(token)))
		buf = append(buf, raw[:]...)
	}
	return hash(buf)
}

func equalTokens(a, b []int) bool {
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
