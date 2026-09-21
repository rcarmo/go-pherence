// Package httpinput provides bounded single-document HTTP JSON decoding.
package httpinput

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
)

// DecodeJSON reads exactly one JSON value, including the trailing whitespace,
// under a byte limit. LimitReader alone cannot distinguish a truncated body from
// a complete document followed by more data. The handler still owns Body.Close.
func DecodeJSON(w http.ResponseWriter, r *http.Request, dst any, limit int64, strict bool) error {
	if limit <= 0 {
		return fmt.Errorf("invalid JSON body limit")
	}
	r.Body = http.MaxBytesReader(w, r.Body, limit)
	dec := json.NewDecoder(r.Body)
	if strict {
		dec.DisallowUnknownFields()
	}
	if err := dec.Decode(dst); err != nil {
		return err
	}
	var extra json.RawMessage
	if err := dec.Decode(&extra); err != io.EOF {
		if err != nil {
			return err
		}
		return fmt.Errorf("expected a single JSON value")
	}
	return nil
}

func ErrorStatus(err error) int {
	var tooLarge *http.MaxBytesError
	if errors.As(err, &tooLarge) {
		return http.StatusRequestEntityTooLarge
	}
	return http.StatusBadRequest
}
