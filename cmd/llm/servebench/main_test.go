package main

import (
	"os"
	"reflect"
	"testing"
)

func TestParsePositiveInts(t *testing.T) {
	got, err := parsePositiveInts("1, 2,2,8")
	if err != nil || !reflect.DeepEqual(got, []int{1, 2, 8}) {
		t.Fatalf("got=%v err=%v", got, err)
	}
	if _, err := parsePositiveInts("0"); err == nil {
		t.Fatal("accepted zero")
	}
}

func TestLoadPromptsInlineAndFile(t *testing.T) {
	got, err := loadPrompts(" a | b ", "")
	if err != nil || !reflect.DeepEqual(got, []string{"a", "b"}) {
		t.Fatalf("inline=%v err=%v", got, err)
	}
	path := t.TempDir() + "/prompts.txt"
	if err := os.WriteFile(path, []byte("one\r\ntwo\n\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err = loadPrompts("ignored", path)
	if err != nil || !reflect.DeepEqual(got, []string{"one", "two"}) {
		t.Fatalf("file=%v err=%v", got, err)
	}
}
