package fft

import (
	"math"
	"reflect"
	"testing"
)

func TestMelTablesContract(t *testing.T) {
	window := PrecomputeHannWindow(400)
	if len(window) != 400 || window[0] != 0 || math.IsNaN(float64(window[200])) || math.IsInf(float64(window[200]), 0) {
		t.Fatal("invalid Hann table")
	}
	if again := PrecomputeHannWindow(400); !reflect.DeepEqual(window, again) {
		t.Fatal("Hann table is not deterministic")
	}
	filters := PrecomputeMelFilters(80, 257, 16000, 512)
	if len(filters) != 80*257 {
		t.Fatal("invalid filter extent", len(filters))
	}
	positive := 0
	for i, value := range filters {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || value < 0 || value > 1 {
			t.Fatalf("invalid filter[%d]=%v", i, value)
		}
		if value > 0 {
			positive++
		}
	}
	if positive == 0 {
		t.Fatal("empty filterbank")
	}
	if again := PrecomputeMelFilters(80, 257, 16000, 512); !reflect.DeepEqual(filters, again) {
		t.Fatal("filterbank is not deterministic")
	}
}
