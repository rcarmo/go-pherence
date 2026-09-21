package httpinput

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDecodeSingleBoundedDocument(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		limit      int64
		strict     bool
		status     int
	}{
		{"exact", `{"x":1}`, 7, true, 0},
		{"whitespace", `{"x":1} `, 8, true, 0},
		{"oversize", `{"x":123}`, 7, true, 413},
		{"oversize tail", `{"x":1}   `, 8, true, 413},
		{"second value", `{"x":1} {}`, 30, true, 400},
		{"garbage tail", `{"x":1} x`, 30, true, 400},
		{"empty", "", 30, true, 400},
		{"unknown strict", `{"y":1}`, 30, true, 400},
		{"unknown compatible", `{"y":1}`, 30, false, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(tc.body))
			r.ContentLength = -1
			w := httptest.NewRecorder()
			var v struct {
				X int `json:"x"`
			}
			err := DecodeJSON(w, r, &v, tc.limit, tc.strict)
			if tc.status == 0 {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || ErrorStatus(err) != tc.status {
				t.Fatalf("err=%v status=%d want=%d", err, ErrorStatus(err), tc.status)
			}
		})
	}
}
