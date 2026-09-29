package nemotronasr

import "testing"

func TestEncoderMaskRowsPinnedLengths(t *testing.T) {
	for _, tc := range []struct {
		frames, valid, rows, visible int
		first                        bool
	}{
		{26, 25, 4, 4, true}, {32, 32, 4, 5, false}, {32, 24, 4, 4, false},
		{26, 1, 4, 1, true}, {32, 0, 4, 1, false}, {8, 0, 1, 1, false},
		{26, 26, 4, 5, true}, // the pinned mask formula is not clamped to target rows
	} {
		rows, visible, err := EncoderMaskRows(tc.frames, tc.valid, tc.first)
		if err != nil || rows != tc.rows || visible != tc.visible {
			t.Fatalf("frames=%d valid=%d rows=%d visible=%d err=%v", tc.frames, tc.valid, rows, visible, err)
		}
	}
	for _, tc := range []struct{ frames, valid int }{{0, 0}, {129, 0}, {26, -1}, {26, 27}} {
		if _, _, err := EncoderMaskRows(tc.frames, tc.valid, true); err == nil {
			t.Fatalf("accepted invalid mask frames=%d valid=%d", tc.frames, tc.valid)
		}
	}
}
