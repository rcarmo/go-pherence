package whisperggml

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func writeRawFixture(t *testing.T, raw []byte) (string, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.bin")
	if err := os.WriteFile(path, raw, 0600); err != nil {
		t.Fatal(err)
	}
	h := sha256.Sum256(raw)
	return path, hex.EncodeToString(h[:])
}

type readCancelContext struct {
	context.Context
	calls, at int
}

func (c *readCancelContext) Err() error {
	c.calls++
	if c.calls >= c.at {
		return context.Canceled
	}
	return nil
}

func TestQ5BlocksOwnedAdmission(t *testing.T) {
	path, pin := writeRawFixture(t, fixture(6))
	file, err := Open(context.Background(), path, pin)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	raw, shape, err := file.Q5Blocks(context.Background(), "encoder.test")
	if err != nil || len(raw) != 22 || !reflect.DeepEqual(shape, []int{32, 1}) {
		t.Fatal(raw, shape, err)
	}
	raw[0] = 255
	shape[0] = 1
	again, dims, err := file.Q5Blocks(context.Background(), "encoder.test")
	if err != nil || again[0] == 255 || dims[0] != 32 {
		t.Fatal("borrowed")
	}
	if _, _, err := file.Q5Blocks(nil, "encoder.test"); err == nil {
		t.Fatal("nilctx")
	}
	if _, _, err := (*File)(nil).Q5Blocks(context.Background(), "x"); err == nil {
		t.Fatal("nilfile")
	}
	if _, _, err := file.Q5Blocks(context.Background(), "missing"); err == nil {
		t.Fatal("name")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := file.Q5Blocks(ctx, "encoder.test"); !errors.Is(err, context.Canceled) {
		t.Fatal("cancel", err)
	}
	last := &readCancelContext{Context: context.Background(), at: 3}
	if data, _, err := file.Q5Blocks(last, "encoder.test"); data != nil || !errors.Is(err, context.Canceled) {
		t.Fatal("final read cancellation", err)
	}
	if err := os.Truncate(path, 50); err != nil {
		t.Fatal(err)
	}
	if data, _, err := file.Q5Blocks(context.Background(), "encoder.test"); data != nil || err == nil {
		t.Fatal("short read")
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := file.Q5Blocks(context.Background(), "encoder.test"); err == nil {
		t.Fatal("closed")
	}
	for _, kind := range []int{0, 1} {
		p, h := writeRawFixture(t, fixture(kind))
		f, e := Open(context.Background(), p, h)
		if e != nil {
			t.Fatal(e)
		}
		if _, _, e = f.Q5Blocks(context.Background(), "encoder.test"); e == nil {
			t.Fatal("wrong type")
		}
		f.Close()
	}
}
