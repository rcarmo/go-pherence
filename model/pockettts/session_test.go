package pockettts

import "testing"

func TestNilSessionRejectsGeneration(t *testing.T) {
	var session *Session
	if _, err := session.GenerateInto(nil, nil, 1, 0, 1, 0, func(int, []float32) error { return nil }); err == nil {
		t.Fatal("accepted nil session")
	}
}
