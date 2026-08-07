package servingbench

import (
	"reflect"
	"testing"
	"time"
)

func TestArrivalConfigValidate(t *testing.T) {
	cases := []struct {
		name string
		cfg  ArrivalConfig
		ok   bool
	}{
		{name: "fixed", cfg: ArrivalConfig{Mode: ArrivalFixed, Rate: 2}, ok: true},
		{name: "poisson", cfg: ArrivalConfig{Mode: ArrivalPoisson, Rate: 3}, ok: true},
		{name: "gamma default shape", cfg: ArrivalConfig{Mode: ArrivalGamma, Rate: 4}, ok: true},
		{name: "missing rate", cfg: ArrivalConfig{Mode: ArrivalFixed}, ok: false},
		{name: "bad shape", cfg: ArrivalConfig{Mode: ArrivalGamma, Rate: 1, GammaShape: -1}, ok: false},
		{name: "bad mode", cfg: ArrivalConfig{Mode: "burst", Rate: 1}, ok: false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.Validate()
			if tc.ok && err != nil {
				t.Fatalf("Validate() error = %v", err)
			}
			if !tc.ok && err == nil {
				t.Fatal("Validate() succeeded unexpectedly")
			}
		})
	}
}

func TestGenerateArrivalOffsetsDeterministic(t *testing.T) {
	cases := []struct {
		name string
		cfg  ArrivalConfig
		seed int64
		want []time.Duration
	}{
		{
			name: "fixed",
			cfg:  ArrivalConfig{Mode: ArrivalFixed, Rate: 2},
			seed: 7,
			want: []time.Duration{0, 500 * time.Millisecond, 1 * time.Second, 1500 * time.Millisecond, 2 * time.Second},
		},
		{
			name: "poisson",
			cfg:  ArrivalConfig{Mode: ArrivalPoisson, Rate: 2},
			seed: 7,
			want: []time.Duration{0, 416762645 * time.Nanosecond, 821071133 * time.Nanosecond, 969463917 * time.Nanosecond, 1725086472 * time.Nanosecond},
		},
		{
			name: "gamma shape lt 1",
			cfg:  ArrivalConfig{Mode: ArrivalGamma, Rate: 2, GammaShape: 0.5},
			seed: 7,
			want: []time.Duration{0, 2073187398 * time.Nanosecond, 2729982097 * time.Nanosecond, 3097982275 * time.Nanosecond, 5987175804 * time.Nanosecond},
		},
		{
			name: "gamma default shape",
			cfg:  ArrivalConfig{Mode: ArrivalGamma, Rate: 2},
			seed: 7,
			want: []time.Duration{0, 343932255 * time.Nanosecond, 1136658441 * time.Nanosecond, 1438557418 * time.Nanosecond, 3411351817 * time.Nanosecond},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := GenerateArrivalOffsets(len(tc.want), tc.cfg, tc.seed)
			if err != nil {
				t.Fatalf("GenerateArrivalOffsets() error = %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("GenerateArrivalOffsets() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestGenerateArrivalOffsetsErrors(t *testing.T) {
	if _, err := GenerateArrivalOffsets(-1, ArrivalConfig{Mode: ArrivalFixed, Rate: 1}, 1); err == nil {
		t.Fatal("expected error for negative count")
	}
	got, err := GenerateArrivalOffsets(0, ArrivalConfig{Mode: ArrivalFixed, Rate: 1}, 1)
	if err != nil {
		t.Fatalf("zero-count GenerateArrivalOffsets() error = %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("zero-count len = %d, want 0", len(got))
	}
}
