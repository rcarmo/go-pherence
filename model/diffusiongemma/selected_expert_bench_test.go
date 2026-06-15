package diffusiongemma

import "testing"

func BenchmarkSelectedExpertGroupedMetadataReusePositions8TopK4(b *testing.B) {
	positions, topK, experts := 8, 4, 128
	ids := make([]int, positions*topK)
	vals := make([]float32, positions*topK)
	for pos := 0; pos < positions; pos++ {
		for slot := 0; slot < topK; slot++ {
			ids[pos*topK+slot] = (pos*17 + slot*19) % experts
			vals[pos*topK+slot] = float32(slot+1) / 10
		}
	}
	var items []SelectedExpertWorkItem
	var arr SelectedExpertWorkArrays
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var err error
		items, err = FlattenSelectedExpertsInto(items, ids, vals, positions, topK, experts)
		if err != nil {
			b.Fatal(err)
		}
		BuildSelectedExpertWorkArraysInto(&arr, items)
		grouped, err := BuildSelectedExpertGroupedWork(arr, experts)
		if err != nil {
			b.Fatal(err)
		}
		ga, err := BuildSelectedExpertGroupedArrays(arr, grouped)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := SummarizeSelectedExpertGroupedWork(grouped, arr.Len()); err != nil {
			b.Fatal(err)
		}
		if err := ga.Validate(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSelectedExpertGroupedMetadataPositions8TopK4(b *testing.B) {
	positions, topK, experts := 8, 4, 128
	ids := make([]int, positions*topK)
	vals := make([]float32, positions*topK)
	for pos := 0; pos < positions; pos++ {
		for slot := 0; slot < topK; slot++ {
			ids[pos*topK+slot] = (pos*17 + slot*19) % experts
			vals[pos*topK+slot] = float32(slot+1) / 10
		}
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		items, err := FlattenSelectedExperts(ids, vals, positions, topK, experts)
		if err != nil {
			b.Fatal(err)
		}
		arr := BuildSelectedExpertWorkArrays(items)
		grouped, err := BuildSelectedExpertGroupedWork(arr, experts)
		if err != nil {
			b.Fatal(err)
		}
		ga, err := BuildSelectedExpertGroupedArrays(arr, grouped)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := SummarizeSelectedExpertGroupedWork(grouped, arr.Len()); err != nil {
			b.Fatal(err)
		}
		if err := ga.Validate(); err != nil {
			b.Fatal(err)
		}
	}
}
