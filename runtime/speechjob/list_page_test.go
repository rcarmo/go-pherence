package speechjob

import (
	"context"
	"errors"
	"os"
	"sort"
	"testing"
)

func TestListPageCursorAndBounds(t *testing.T) {
	s, _ := openTest(t)
	var ids []string
	for i := 0; i < 7; i++ {
		ids = append(ids, createTest(t, s).ID)
	}
	sort.Strings(ids)
	// Incomplete upload directories are inventory entries, never list entries.
	if e := s.root.Mkdir("ffffffffffffffffffffffffffffffff", 0700); e != nil {
		t.Fatal(e)
	}
	var seen []string
	cursor := ""
	for page := 0; page < 10; page++ {
		rows, next, e := s.ListPage(context.Background(), cursor, 2)
		if e != nil || len(rows) > 2 {
			t.Fatal(rows, e)
		}
		for _, m := range rows {
			seen = append(seen, m.ID)
		}
		if next == "" {
			break
		}
		if next <= cursor {
			t.Fatal("cursor not progressing")
		}
		cursor = next
	}
	if len(seen) != len(ids) {
		t.Fatal(seen, ids)
	}
	for i := range ids {
		if seen[i] != ids[i] {
			t.Fatal(seen, ids)
		}
	}
	if rows, next, e := s.ListPage(context.Background(), ids[len(ids)-1], 100); e != nil || len(rows) != 0 || next != "" {
		t.Fatal(rows, next, e)
	}
	for _, limit := range []int{0, 101} {
		if _, _, e := s.ListPage(context.Background(), "", limit); e == nil {
			t.Fatal(limit)
		}
	}
	if _, _, e := s.ListPage(context.Background(), "bad", 1); e == nil {
		t.Fatal("bad cursor")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, e := s.ListPage(ctx, "", 2); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	// Acknowledged manifest corruption fails the page rather than hiding its ID.
	if e := os.WriteFile(s.root.Name()+"/"+ids[0]+"/manifest.json", []byte("bad"), 0600); e != nil {
		t.Fatal(e)
	}
	if _, _, e := s.ListPage(context.Background(), "", 2); !errors.Is(e, ErrCorrupt) {
		t.Fatal(e)
	}
}
