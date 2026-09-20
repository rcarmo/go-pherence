package expertstream

import (
	"errors"
	"testing"
	"time"
)

func TestSlotBudgetIncludesAlignmentAndInventory(t *testing.T) {
	r, path, _ := mustOpenFixture(t, Options{Slots: 2})
	r.Close()
	// largest span=192 and alignment=64; each mapping reserves256 bytes.
	if _, err := Open(path, Options{Slots: 2, MaxSlotBytes: 511}); !errors.Is(err, ErrMemoryBudget) {
		t.Fatal(err)
	}
	r, err := Open(path, Options{Slots: 2, MaxSlotBytes: 512})
	if err != nil {
		t.Fatal(err)
	}
	r.Close()
	for _, opts := range []Options{{Slots: 4}, {Slots: 1, MaxSlotBytes: -1}, {Slots: 65537}} {
		if r, err := Open(path, opts); err == nil {
			r.Close()
			t.Fatal("invalid options accepted", opts)
		}
	}
}
func TestWithExpertsPinsSlotsUntilCallbackReturns(t *testing.T) {
	for _, closeReader := range []bool{false, true} {
		t.Run(map[bool]string{false: "reuse", true: "close"}[closeReader], func(t *testing.T) {
			r, _, _ := mustOpenFixture(t, Options{Slots: 1})
			defer r.Close()
			entered, release, done := make(chan struct{}), make(chan struct{}), make(chan error, 1)
			go func() {
				done <- r.WithExperts([]uint64{1}, func(experts []LoadedExpert) error {
					before := append([]byte(nil), experts[0].Slot.Bytes...)
					close(entered)
					<-release
					for i, b := range before {
						if experts[0].Slot.Bytes[i] != b {
							return errors.New("borrowed bytes changed during callback")
						}
					}
					return nil
				})
			}()
			<-entered
			writer := make(chan error, 1)
			go func() {
				if closeReader {
					writer <- r.Close()
				} else {
					_, err := r.Load([]uint64{2})
					writer <- err
				}
			}()
			select {
			case err := <-writer:
				t.Errorf("writer passed active owner: %v", err)
			case <-time.After(10 * time.Millisecond):
			}
			close(release)
			if err := <-done; err != nil {
				t.Fatal(err)
			}
			select {
			case err := <-writer:
				if err != nil {
					t.Fatal(err)
				}
			case <-time.After(time.Second):
				t.Fatal("writer did not resume")
			}
		})
	}
}
func TestWithExpertsErrorAndPanicReleaseOwner(t *testing.T) {
	r, _, _ := mustOpenFixture(t, Options{Slots: 1})
	defer r.Close()
	want := errors.New("callback")
	if err := r.WithExperts([]uint64{1}, func([]LoadedExpert) error { return want }); err != want {
		t.Fatal(err)
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Error("missing callback panic")
			}
		}()
		_ = r.WithExperts([]uint64{1}, func([]LoadedExpert) error { panic("callback") })
	}()
	if _, err := r.Load([]uint64{2}); err != nil {
		t.Fatal(err)
	}
	if err := r.WithExperts(nil, nil); err == nil {
		t.Fatal("nil callback")
	}
	r.Close()
	if _, err := r.Load(nil); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	var nilReader *Reader
	if _, err := nilReader.Load(nil); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
}
