package board

import (
	"testing"
	"time"
)

func TestParseBenchOutputSchemas(t *testing.T) {
	for _, body := range []string{`[{"n_prompt":128,"n_gen":0,"avg_ts":12.5},{"n_prompt":0,"n_gen":32,"avg_ts":4.25}]`, "startup log\n{\"pp\":12.5}\n{\"tg\":4.25}"} {
		p, g, e := parseBenchOutput([]byte(body), true, true)
		if e != nil || p != 12.5 || g != 4.25 {
			t.Fatal(p, g, e)
		}
	}
	for _, body := range []string{"", "[]", `[{"avg_ts":4}]`, `{"pp":-1}`, `{"pp":"12"}`, `[{"n_prompt":128,"n_gen":32,"avg_ts":5}]`, "{bad"} {
		if _, _, err := parseBenchOutput([]byte(body), true, true); err == nil {
			t.Fatal("invalid output accepted", body)
		}
	}
	if _, err := RunGGUF("", 1, 1, 1, time.Second); err == nil {
		t.Fatal("invalid arguments accepted")
	}
}
