package httpapi

import (
	"context"
	"errors"
	"io"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

func TestHTTPQueueAdmissionPanicBeforeMutationFailsClosed(t *testing.T) {
	for _, mode := range []string{"admission", "partial-release", "cancel-release"} {
		t.Run(mode, func(t *testing.T) {
			var calls atomic.Int32
			admitted, unblock := make(chan struct{}), make(chan struct{})
			defer func() {
				select {
				case <-unblock:
				default:
					close(unblock)
				}
			}()
			a := func(ctx context.Context) (func(), error) {
				close(admitted)
				switch mode {
				case "admission":
					panic("private")
				case "partial-release":
					return func() { panic("private") }, errors.New("private")
				default:
					<-ctx.Done()
					<-unblock
					return func() { panic("private") }, nil
				}
			}
			h := budgetHTTP(t, a, true, testStage("transcript", func(context.Context, *speechjob.Input, io.Writer) error { calls.Add(1); return nil }))
			j := upload(t, h)
			request(h, "POST", "/v1/jobs/"+j.ID+"/enqueue", nil)
			h.StartQueue(context.Background())
			<-admitted
			if mode == "cancel-release" {
				if w := request(h, "POST", "/v1/jobs/"+j.ID+"/cancel", nil); w.Code != 202 {
					t.Fatal(w)
				}
				close(unblock)
			}
			deadline := time.Now().Add(time.Second)
			for {
				_, e := h.queue.List()
				if e != nil {
					break
				}
				if time.Now().After(deadline) {
					t.Fatal("queue not poisoned")
				}
				time.Sleep(time.Millisecond)
			}
			if calls.Load() != 0 {
				t.Fatal("inference after admission panic")
			}
			if w := request(h, "DELETE", "/v1/jobs/"+j.ID, nil); w.Code != 503 {
				t.Fatal("uncertain HTTP mutation enabled", w)
			}
		})
	}
}
