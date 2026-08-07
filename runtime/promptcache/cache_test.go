package promptcache

import (
	"errors"
	"fmt"
	"sync"
	"testing"
)

type testSnapshot struct {
	pos     int
	bytes   int64
	data    []int
	sizeErr error
}

func (s *testSnapshot) Position() int { return s.pos }

func (s *testSnapshot) SizeBytes() (int64, error) {
	if s == nil {
		return 0, nil
	}
	if s.sizeErr != nil {
		return 0, s.sizeErr
	}
	return s.bytes, nil
}

func (s *testSnapshot) Clone() Snapshot {
	if s == nil {
		return (*testSnapshot)(nil)
	}
	out := *s
	out.data = append([]int(nil), s.data...)
	return &out
}

func validIdentity(salt string) Identity {
	return Identity{
		ModelFingerprint:      "model-fp",
		CheckpointFingerprint: "ckpt-fp",
		Backend:               "cpu",
		WeightLayout:          "dense",
		WeightDType:           "f16",
		KVPolicy:              "full",
		KVPrecision:           "f16",
		ConfigFingerprint:     "cfg-fp",
		RoPEFingerprint:       "rope-fp",
		AdapterID:             "adapter-a",
		MultimodalHash:        "mm-none",
		CacheSalt:             salt,
	}
}

func mustPut(t *testing.T, c *Cache, id Identity, blockSize int, tokens []int, snap Snapshot) {
	t.Helper()
	if err := c.Put(id, blockSize, tokens, snap); err != nil {
		t.Fatalf("Put(%v): %v", tokens, err)
	}
}

func mustFind(t *testing.T, c *Cache, id Identity, blockSize int, tokens []int) Snapshot {
	t.Helper()
	snap, ok, err := c.FindLongest(id, blockSize, tokens)
	if err != nil {
		t.Fatalf("FindLongest(%v): %v", tokens, err)
	}
	if !ok {
		t.Fatalf("FindLongest(%v): miss", tokens)
	}
	return snap
}

func TestIdentityIsolationAndSalt(t *testing.T) {
	c := New(64)
	idA := validIdentity("salt-a")
	idB := validIdentity("salt-b")
	mustPut(t, c, idA, 2, []int{1, 2}, &testSnapshot{pos: 2, bytes: 8, data: []int{11}})

	got := mustFind(t, c, idA, 2, []int{1, 2, 3})
	if got.Position() != 2 {
		t.Fatalf("position=%d want 2", got.Position())
	}
	if _, ok, err := c.FindLongest(idB, 2, []int{1, 2, 3}); err != nil || ok {
		t.Fatalf("different salt should miss, ok=%v err=%v", ok, err)
	}

	idBackend := idA
	idBackend.Backend = "gpu"
	if _, ok, err := c.FindLongest(idBackend, 2, []int{1, 2}); err != nil || ok {
		t.Fatalf("different backend should miss, ok=%v err=%v", ok, err)
	}
}

func TestCollisionVerification(t *testing.T) {
	constantHash := func([]byte) [32]byte { return [32]byte{1} }
	c := New(128, WithHashFunc(constantHash))
	id := validIdentity("salt")
	mustPut(t, c, id, 2, []int{1, 2}, &testSnapshot{pos: 2, bytes: 8, data: []int{10}})
	mustPut(t, c, id, 2, []int{3, 4}, &testSnapshot{pos: 2, bytes: 8, data: []int{20}})

	got := mustFind(t, c, id, 2, []int{3, 4, 5, 6}).(*testSnapshot)
	if got.data[0] != 20 {
		t.Fatalf("collision returned wrong entry: %+v", got)
	}
	if _, ok, err := c.FindLongest(id, 2, []int{7, 8}); err != nil || ok {
		t.Fatalf("collision-only candidate should miss, ok=%v err=%v", ok, err)
	}
	st := c.Stats()
	if st.Collisions == 0 {
		t.Fatalf("expected collision accounting, got %+v", st)
	}
}

func TestFindLongestAndIncompleteQueryTail(t *testing.T) {
	c := New(128)
	id := validIdentity("salt")
	mustPut(t, c, id, 2, []int{1, 2}, &testSnapshot{pos: 2, bytes: 8, data: []int{2}})
	mustPut(t, c, id, 2, []int{1, 2, 3, 4}, &testSnapshot{pos: 4, bytes: 12, data: []int{4}})

	got := mustFind(t, c, id, 2, []int{1, 2, 3, 4, 5}).(*testSnapshot)
	if got.pos != 4 || got.data[0] != 4 {
		t.Fatalf("tail lookup got %+v", got)
	}
	got = mustFind(t, c, id, 2, []int{1, 2, 9}).(*testSnapshot)
	if got.pos != 2 || got.data[0] != 2 {
		t.Fatalf("shorter prefix lookup got %+v", got)
	}
}

func TestOwnershipAndCloning(t *testing.T) {
	c := New(128)
	id := validIdentity("salt")
	tokens := []int{1, 2}
	orig := &testSnapshot{pos: 2, bytes: 8, data: []int{7, 8}}
	mustPut(t, c, id, 2, tokens, orig)
	tokens[0] = 99
	orig.data[0] = 99

	got := mustFind(t, c, id, 2, []int{1, 2}).(*testSnapshot)
	if got.data[0] != 7 {
		t.Fatalf("stored snapshot was not cloned: %+v", got)
	}
	got.data[0] = 55
	got2 := mustFind(t, c, id, 2, []int{1, 2}).(*testSnapshot)
	if got2.data[0] != 7 {
		t.Fatalf("returned snapshot alias leaked: %+v", got2)
	}
}

func TestBudgetReplacementAndEviction(t *testing.T) {
	c := New(10)
	id := validIdentity("salt")
	mustPut(t, c, id, 2, []int{1, 2}, &testSnapshot{pos: 2, bytes: 4, data: []int{1}})
	mustPut(t, c, id, 2, []int{3, 4}, &testSnapshot{pos: 2, bytes: 4, data: []int{2}})
	mustPut(t, c, id, 2, []int{1, 2}, &testSnapshot{pos: 2, bytes: 6, data: []int{3}})

	if st := c.Stats(); st.UsedBytes != 10 || st.Replacements != 1 || st.Entries != 2 {
		t.Fatalf("after replacement stats=%+v", st)
	}
	mustFind(t, c, id, 2, []int{1, 2})
	mustPut(t, c, id, 2, []int{5, 6}, &testSnapshot{pos: 2, bytes: 7, data: []int{5}})

	if _, ok, err := c.FindLongest(id, 2, []int{3, 4}); err != nil || ok {
		t.Fatalf("expected [3 4] eviction, ok=%v err=%v", ok, err)
	}
	if _, ok, err := c.FindLongest(id, 2, []int{1, 2}); err != nil || ok {
		t.Fatalf("expected [1 2] eviction after budget pressure, ok=%v err=%v", ok, err)
	}
	got := mustFind(t, c, id, 2, []int{5, 6}).(*testSnapshot)
	if got.data[0] != 5 {
		t.Fatalf("missing retained entry: %+v", got)
	}
	if err := c.Put(id, 2, []int{7, 8}, &testSnapshot{pos: 2, bytes: 11}); !errors.Is(err, ErrOverBudget) {
		t.Fatalf("oversized put err=%v", err)
	}
	st := c.Stats()
	if st.BudgetRejections == 0 || st.Evictions != 2 || st.UsedBytes != 7 || st.Entries != 1 {
		t.Fatalf("final stats=%+v", st)
	}
}

func TestRejectMalformedInputs(t *testing.T) {
	c := New(32)
	id := validIdentity("salt")
	if err := c.Put(Identity{}, 2, []int{1, 2}, &testSnapshot{pos: 2, bytes: 8}); !errors.Is(err, ErrInvalidIdentity) {
		t.Fatalf("invalid identity err=%v", err)
	}
	if err := c.Put(id, 2, []int{1}, &testSnapshot{pos: 1, bytes: 8}); !errors.Is(err, ErrIncompleteBlock) {
		t.Fatalf("incomplete block err=%v", err)
	}
	if err := c.Put(id, 2, []int{1, 2}, &testSnapshot{pos: 1, bytes: 8}); !errors.Is(err, ErrPositionMismatch) {
		t.Fatalf("position mismatch err=%v", err)
	}
	if err := c.Put(id, 2, []int{1, 2}, nil); !errors.Is(err, ErrNilSnapshot) {
		t.Fatalf("nil interface err=%v", err)
	}
	var typedNil *testSnapshot
	if err := c.Put(id, 2, []int{1, 2}, typedNil); !errors.Is(err, ErrNilSnapshot) {
		t.Fatalf("typed nil err=%v", err)
	}
	if err := c.Put(id, 2, []int{1, 2}, &testSnapshot{pos: 2, bytes: -1}); !errors.Is(err, ErrSizeOverflow) {
		t.Fatalf("negative size err=%v", err)
	}
	sizeErr := fmt.Errorf("%w: synthetic", ErrSizeOverflow)
	if err := c.Put(id, 2, []int{1, 2}, &testSnapshot{pos: 2, sizeErr: sizeErr}); !errors.Is(err, ErrSizeOverflow) {
		t.Fatalf("size error err=%v", err)
	}
	if _, _, err := c.FindLongest(Identity{}, 2, []int{1, 2}); !errors.Is(err, ErrInvalidIdentity) {
		t.Fatalf("invalid lookup identity err=%v", err)
	}
}

func TestCacheConcurrentAccess(t *testing.T) {
	c := New(1 << 20)
	id := validIdentity("salt")

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		g := g
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				base := (g*200 + i) * 2
				tokens := []int{base, base + 1}
				snap := &testSnapshot{pos: 2, bytes: 16, data: []int{base}}
				if err := c.Put(id, 2, tokens, snap); err != nil {
					t.Errorf("Put(%v): %v", tokens, err)
					return
				}
				got, ok, err := c.FindLongest(id, 2, append([]int(nil), tokens...))
				if err != nil {
					t.Errorf("FindLongest(%v): %v", tokens, err)
					return
				}
				if !ok {
					t.Errorf("FindLongest(%v): miss", tokens)
					return
				}
				if got.Position() != 2 {
					t.Errorf("FindLongest(%v) position=%d", tokens, got.Position())
					return
				}
				_ = c.Stats()
			}
		}()
	}
	wg.Wait()
	st := c.Stats()
	if st.Hits == 0 || st.Puts == 0 || st.Lookups == 0 {
		t.Fatalf("unexpected stats: %+v", st)
	}
}
