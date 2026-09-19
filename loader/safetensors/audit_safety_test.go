package safetensors

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestShardedRejectsTraversalAndSymlinkEscape(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "model")
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(root, "outside.safetensors")
	if err := os.WriteFile(outside, make([]byte, 8), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, filename := range []string{"../outside.safetensors", outside, "a/../b.safetensors", "a\\b.safetensors", "", "."} {
		b, _ := json.Marshal(map[string]any{"weight_map": map[string]string{"x": filename}})
		index := filepath.Join(dir, "index.json")
		os.WriteFile(index, b, 0o600)
		if _, err := OpenSharded(index); err == nil {
			t.Fatal("unsafe shard accepted", filename)
		}
	}
	if err := os.Symlink(outside, filepath.Join(dir, "escape.safetensors")); err != nil {
		t.Skip(err)
	}
	b, _ := json.Marshal(map[string]any{"weight_map": map[string]string{"x": "escape.safetensors"}})
	os.WriteFile(filepath.Join(dir, "index.json"), b, 0o600)
	if _, err := OpenSharded(filepath.Join(dir, "index.json")); err == nil {
		t.Fatal("symlink escape accepted")
	}
}
func TestEagerLoadIndependentFilesConcurrent(t *testing.T) {
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			f := &File{mmapData: make([]byte, 8193)}
			for range 100 {
				if _, err := f.EagerLoad(); err != nil {
					t.Error(err)
				}
			}
		}()
	}
	wg.Wait()
}

func TestShardedRelativePathThroughSymlinkedWorkingDirectory(t *testing.T) {
	root := t.TempDir()
	physical := filepath.Join(root, "real")
	if err := os.Mkdir(physical, 0o755); err != nil {
		t.Fatal(err)
	}
	alias := filepath.Join(root, "alias")
	if err := os.Symlink(physical, alias); err != nil {
		t.Skip(err)
	}
	shard := writeTestSafetensors(t, `{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, []byte{0, 0, 0, 0})
	data, err := os.ReadFile(shard)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(physical, "shard.safetensors"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	index, _ := json.Marshal(map[string]any{"weight_map": map[string]string{"x": "shard.safetensors"}})
	os.WriteFile(filepath.Join(physical, "index.json"), index, 0o600)
	t.Chdir(root)
	f, err := OpenSharded("alias/index.json")
	if err != nil {
		t.Fatal(err)
	}
	f.Close()
}

func TestMetadataResolutionFailsClosedAndAcceptsExplicitIndex(t *testing.T) {
	dir := t.TempDir()
	source := writeTestSafetensors(t, `{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, []byte{0, 0, 0, 0})
	data, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "model.safetensors"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	index := filepath.Join(dir, "model.safetensors.index.json")
	check := func(explicit string, wantErr bool) {
		t.Helper()
		names, err := NamesFrom(dir, explicit)
		if (err != nil) != wantErr {
			t.Fatalf("NamesFrom(%q): %v", explicit, err)
		}
		infos, err := TensorInfosFrom(dir, explicit)
		if (err != nil) != wantErr {
			t.Fatalf("TensorInfosFrom(%q): %v", explicit, err)
		}
		if !wantErr && (len(names) != 1 || names[0] != "x" || len(infos) != 1) {
			t.Fatal(names, infos)
		}
	}
	check("", false) // absent index permits single-file fallback
	for _, body := range []string{`{`, `{"weight_map":{"x":"missing.safetensors"}}`, `{"weight_map":{"x":"../escape.safetensors"}}`} {
		if err = os.WriteFile(index, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		check("", true)
		check(index, true)
	}
	if err = os.WriteFile(index, []byte(`{"weight_map":{"x":"model.safetensors"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	check("", false)
	check(index, false)
	check(filepath.Join(dir, "model.safetensors"), false)
}

func TestCopyingGettersOwnDataAndShapeAfterClose(t *testing.T) {
	path := writeTestSafetensors(t, `{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, []byte{0, 0, 128, 63})
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	vals, shape, err := f.GetFloat32("x")
	if err != nil {
		t.Fatal(err)
	}
	bf, _, err := f.GetBF16("x")
	if err != nil {
		t.Fatal(err)
	}
	// GetRaw deliberately borrows both data and metadata and must remain
	// allocation-free for streaming loaders. Do not mutate that view.
	shape[0] = 99
	if allocs := testing.AllocsPerRun(50, func() { _, _, _, _ = f.GetRaw("x") }); allocs != 0 {
		t.Fatalf("raw getter allocs=%g", allocs)
	}
	if f.Tensors["x"].Shape[0] != 1 {
		t.Fatal("returned shape aliases metadata")
	}
	if err = f.Close(); err != nil {
		t.Fatal(err)
	}
	if len(vals) != 1 || vals[0] != 1 || len(bf) != 1 || bf[0] != 0x3f80 {
		t.Fatal("converted data did not survive Close")
	}
	if _, _, err = f.GetFloat32("x"); err == nil {
		t.Fatal("closed getter accepted")
	}
	if _, _, _, err = f.GetRaw("x"); err == nil {
		t.Fatal("closed raw getter accepted")
	}
	if _, err = f.EagerLoad(); err == nil {
		t.Fatal("closed prefetch accepted")
	}
	if err = f.Close(); err != nil {
		t.Fatal(err)
	}
	var nilFile *File
	if err = nilFile.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCloseWaitsForOwnedConversionsAndPrefetch(t *testing.T) {
	// Real mmap, never dereference a borrowed GetRaw slice after Close.
	data := make([]byte, 4*4096)
	path := writeTestSafetensors(t, `{"x":{"dtype":"F32","shape":[4096],"data_offsets":[0,16384]}}`, data)
	for round := 0; round < 16; round++ {
		f, err := Open(path)
		if err != nil {
			t.Fatal(err)
		}
		start := make(chan struct{})
		var wg sync.WaitGroup
		for i := 0; i < 8; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start
				for j := 0; j < 16; j++ {
					if i%3 == 0 {
						_, _ = f.EagerLoad()
					} else if i%3 == 1 {
						_, _, _ = f.GetFloat32("x")
					} else {
						_, _, _ = f.GetBF16("x")
					}
				}
			}()
		}
		wg.Add(1)
		go func() { defer wg.Done(); <-start; _ = f.Close() }()
		close(start)
		wg.Wait()
		if err = f.Close(); err != nil {
			t.Fatal(err)
		}
	}
}
