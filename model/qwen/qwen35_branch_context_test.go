package qwen

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestSIMDTreeContextWaitAndOutput(t *testing.T) {
	s := &Qwen35SIMDBranch{model: &Qwen35BaseModel{}, maxTokens: 3}
	rows := [][]float32{make([]float32, 1024), make([]float32, 1024), make([]float32, 1024)}
	dst, rope := make([]float32, 3*1024), make([]float32, 3*64)
	for i := range dst {
		dst[i] = 17
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := s.ForwardTreeIntoContext(canceled, dst, rows, 1, 1, []int{3}, rope, 1e-6); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := s.ForwardTreeIntoContext(nil, dst, rows, 1, 1, []int{3}, rope, 1e-6); err == nil {
		t.Fatal("nil context")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	ctx, stop := context.WithCancel(context.Background())
	defer stop()
	done := make(chan error, 1)
	go func() { done <- s.ForwardTreeIntoContext(ctx, dst, rows, 1, 1, []int{3}, rope, 1e-6) }()
	stop()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("waiter stuck")
	}
	for _, v := range dst {
		if v != 17 {
			t.Fatal("cancellation published output")
		}
	}
}
