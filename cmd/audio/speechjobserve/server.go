package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"sync"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/runtime/resourcebudget"
	"github.com/rcarmo/go-pherence/runtime/speechjob"
	"github.com/rcarmo/go-pherence/runtime/speechjob/httpapi"
)

type stageOwner interface {
	Stage() speechjob.Stage
	Close(context.Context) error
}
type builtProfiles struct {
	Profiles []httpapi.Profile
	owners   []stageOwner
}

func (b *builtProfiles) Close(ctx context.Context) error {
	if b == nil {
		return nil
	}
	var failures []error
	for i := len(b.owners) - 1; i >= 0; i-- {
		if b.owners[i] != nil {
			if e := b.owners[i].Close(ctx); e != nil {
				failures = append(failures, e)
			} else {
				b.owners[i] = nil
			}
		}
	}
	return errors.Join(failures...)
}

// Cleanup after owned native construction must finish before startup returns.
// A failed close is retried because the owner retains unresolved resources.
func closeBuiltProfiles(b *builtProfiles) {
	for {
		if e := b.Close(context.Background()); e == nil {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
}
func callVulkanCleanup(fn func() error) (err error, returned bool) {
	defer func() { _ = recover() }()
	err = fn()
	returned = true
	return err, returned
}

func closeVulkanResource(closeResource func() error, poll time.Duration, drain func(context.Context, time.Duration) error, quarantine func()) {
	for {
		closeErr, returned := callVulkanCleanup(closeResource)
		if !returned {
			quarantine()
			return
		}
		if closeErr == nil {
			return
		}
		// A partially constructed owner can retain an accepted submission. Prove
		// it idle with a fresh context before retrying destruction. Only a bounded
		// poll timeout/cancellation is retryable. Fatal, uncertain, unexpected or
		// panicking drain state holds startup until process teardown.
		drainErr, returned := callVulkanCleanup(func() error { return drain(context.Background(), poll) })
		if !returned || errors.Is(drainErr, vk.ErrVulkanDeviceLost) || errors.Is(drainErr, vk.ErrVulkanUncertain) || drainErr != nil && !errors.Is(drainErr, context.DeadlineExceeded) && !errors.Is(drainErr, context.Canceled) {
			quarantine()
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func closeVulkanEncoder(e interface{ Close() error }, poll time.Duration, drain func(context.Context, time.Duration) error) {
	closeVulkanResource(e.Close, poll, drain, func() { select {} })
}

func start(ctx context.Context, path string, check bool, out io.Writer) error {
	return startWithRuntimes(ctx, path, check, out, profileRuntimes{defaultVulkanProfileRuntime(), defaultCommunityRuntime()})
}
func startWithRuntime(ctx context.Context, path string, check bool, out io.Writer, profileRuntime vulkanProfileRuntime) error {
	return startWithRuntimes(ctx, path, check, out, profileRuntimes{profileRuntime, defaultCommunityRuntime()})
}
func startWithRuntimes(ctx context.Context, path string, check bool, out io.Writer, runtimes profileRuntimes) error {
	data, e := boundedFile(ctx, path, 64<<10)
	if e != nil {
		return e
	}
	cfg, e := parseConfig(data)
	if e != nil {
		return e
	}
	if !check && !cfg.AllowExecution {
		return fmt.Errorf("execution is not authorised by this configuration")
	}
	if !check {
		token := os.Getenv("SPEECHJOB_TOKEN")
		if len(token) < 32 || len(token) > 256 {
			return fmt.Errorf("SPEECHJOB_TOKEN required before model loading")
		}
		for _, b := range []byte(token) {
			if b < 33 || b > 126 {
				return fmt.Errorf("invalid token encoding")
			}
		}
	}
	// Read TLS configuration before expensive model work or opening the store.
	var tlsConfig *tls.Config
	if cfg.HTTP.TLSCert != "" {
		certBytes, e := boundedFile(ctx, cfg.HTTP.TLSCert, 1<<20)
		if e != nil {
			return fmt.Errorf("TLS certificate file rejected")
		}
		keyBytes, e := boundedFile(ctx, cfg.HTTP.TLSKey, 1<<20)
		if e != nil {
			return fmt.Errorf("TLS private key file rejected")
		}
		cert, e := tls.X509KeyPair(certBytes, keyBytes)
		if e != nil {
			return fmt.Errorf("TLS certificate/key rejected")
		}
		leaf, e := x509.ParseCertificate(cert.Certificate[0])
		if e != nil || time.Now().Before(leaf.NotBefore) || time.Now().After(leaf.NotAfter) {
			return fmt.Errorf("TLS certificate validity rejected")
		}
		for _, host := range cfg.HTTP.Hosts {
			name := host
			if h, _, e := net.SplitHostPort(host); e == nil {
				name = h
			}
			if e = leaf.VerifyHostname(name); e != nil {
				return fmt.Errorf("TLS certificate does not cover Host allowlist")
			}
		}
		tlsConfig = &tls.Config{MinVersion: tls.VersionTLS12, Certificates: []tls.Certificate{cert}, NextProtos: []string{"http/1.1"}}
	}
	if check {
		if _, e = buildProfile(ctx, cfg, false); e != nil {
			return e
		}
		if e = inspectCommunityMetadata(ctx, cfg); e != nil {
			return e
		}
		return json.NewEncoder(out).Encode(struct {
			MetadataChecked bool `json:"metadata_checked"`
			Loaded          bool `json:"model_loaded"`
			Listening       bool `json:"listening"`
		}{true, false, false})
	}
	if e = ctx.Err(); e != nil {
		return e
	}
	store, e := speechjob.Open(cfg.Store, speechjob.Limits{MaxJobs: cfg.Limits.Jobs, MaxUploadBytes: cfg.Limits.UploadBytes, MaxArtifactBytes: cfg.Limits.ArtifactBytes, MaxBytes: cfg.Limits.StoreBytes})
	if e != nil {
		return fmt.Errorf("store open rejected")
	}
	var budget *resourcebudget.Budget
	var resident *resourcebudget.Lease
	var workAdmission speechjob.Admission
	if r := cfg.Resources; r != nil {
		budget, e = resourcebudget.New(resourcebudget.Config{Capacity: resourcebudget.Resources{CPUSlots: r.CPUSlots, MemoryBytes: r.MemoryBytes}, MaxActive: 2, MaxWaiting: r.MaxWaiting})
		if e != nil {
			store.Close()
			return e
		}
		defer budget.Close()
		resident, e = budget.Acquire(ctx, resourcebudget.Resources{CPUSlots: cfg.Threads, MemoryBytes: r.LoadBytes})
		if e != nil {
			store.Close()
			return e
		}
		// All return paths after loading retain this lease until owned work drains.
		// Release concerns accounting; Go/native RSS need not fall immediately.
		defer resident.Release()
		workAdmission, e = budget.Admission(resourcebudget.Resources{CPUSlots: cfg.Threads, MemoryBytes: r.WorkBytes})
		if e != nil {
			store.Close()
			return e
		}
	}
	// Exclude another server process on this same store before allocating model
	// weights. This does not coordinate other stores or unrelated inference.
	profiles, e := buildProfileOwnedRuntimes(ctx, cfg, true, runtimes)
	if e != nil {
		store.Close()
		return e
	}
	if resident != nil {
		if e = resident.Shrink(resourcebudget.Resources{MemoryBytes: cfg.Resources.ResidentBytes}); e != nil {
			closeBuiltProfiles(profiles)
			store.Close()
			return e
		}
	}
	var queue *httpapi.QueueOptions
	runAdmission := workAdmission
	if cfg.Queue.Enable {
		if workAdmission == nil {
			workAdmission = speechjob.SerialAdmission()
		}
		queue = &httpapi.QueueOptions{Directory: cfg.Queue.Directory, MaxEntries: cfg.Queue.MaxEntries, MaxBytes: cfg.Queue.MaxBytes, JobTimeout: duration(cfg.Queue.JobSeconds), Admission: workAdmission}
		runAdmission = nil
	}
	handler, e := httpapi.New(httpapi.Config{Store: store, Profiles: profiles.Profiles, Token: os.Getenv("SPEECHJOB_TOKEN"), Hosts: cfg.HTTP.Hosts, Origin: cfg.HTTP.Origin, EnableUI: cfg.HTTP.EnableUI, MaxUploadBytes: cfg.Limits.UploadBytes, MaxConcurrentRequests: cfg.HTTP.MaxRequests, Queue: queue, RunAdmission: runAdmission})
	if e != nil {
		closeBuiltProfiles(profiles)
		store.Close()
		return e
	}
	listener, e := net.Listen("tcp", cfg.HTTP.Listen)
	if e != nil {
		handler.Shutdown(context.Background())
		closeBuiltProfiles(profiles)
		store.Close()
		return fmt.Errorf("listener bind failed")
	}
	if cfg.Queue.StartWorker {
		if e = handler.StartQueue(ctx); e != nil {
			listener.Close()
			handler.Shutdown(context.Background())
			closeBuiltProfiles(profiles)
			store.Close()
			return e
		}
	}
	// serveOwned does not return until handler-owned operations stop. The profiles'
	// model closures stay reachable through handler until then; no GC/close early.
	e = serveOwned(ctx, listener, handler, cfg.HTTP, tlsConfig, out)
	// Handler/queue drain precedes resident Vulkan encoder close; store/resource
	// release follows it. Quarantined owner intentionally blocks process exit.
	closeBuiltProfiles(profiles)
	return errors.Join(e, store.Close())
}

type handlerOwner interface {
	http.Handler
	Shutdown(context.Context) error
}

func serveOwned(ctx context.Context, listener net.Listener, handler handlerOwner, cfg HTTPSettings, tlsConfig *tls.Config, out io.Writer) error {
	if e := ctx.Err(); e != nil {
		listener.Close()
		handler.Shutdown(context.Background())
		return e
	}
	// Limit accepted transport resources as well as handler requests. Pending OS
	// listen backlog remains kernel-owned. HTTP/2 multiplexing is not enabled.
	bounded := net.Listener(newCappedListener(listener, cfg.MaxConnections))
	if tlsConfig != nil {
		bounded = tls.NewListener(bounded, tlsConfig)
	}
	base, cancel := context.WithCancel(context.Background())
	defer cancel()
	wrapped := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCtx, stop := context.WithTimeout(r.Context(), duration(cfg.RequestSeconds))
		defer stop()
		handler.ServeHTTP(w, r.WithContext(requestCtx))
	})
	srv := &http.Server{Handler: wrapped, ReadHeaderTimeout: duration(cfg.HeaderSeconds), ReadTimeout: duration(cfg.RequestSeconds), WriteTimeout: duration(cfg.RequestSeconds), IdleTimeout: duration(cfg.IdleSeconds), MaxHeaderBytes: 32 << 10, BaseContext: func(net.Listener) context.Context { return base }, ErrorLog: log.New(io.Discard, "", 0)}
	done := make(chan error, 1)
	go func() { done <- srv.Serve(bounded) }()
	// No secret, model path or store path is logged in startup status.
	if e := json.NewEncoder(out).Encode(struct {
		Listening string `json:"listening"`
		Execution string `json:"execution"`
	}{listener.Addr().String(), "explicit job requests; queue worker only when separately enabled"}); e != nil {
		cancel()
		srv.Close()
		handler.Shutdown(context.Background())
		<-done
		return fmt.Errorf("startup status write failed; listener closed")
	}
	var serveErr error
	select {
	case <-ctx.Done():
	case serveErr = <-done:
		done = nil
	}
	cancel()
	// Close connections promptly (including pre-header idle peers); handler drain
	// below proves no owned callback outlives return, even beyond grace timeout.
	_ = srv.Close()
	drainCtx, stop := context.WithTimeout(context.Background(), duration(cfg.ShutdownSeconds))
	drainErr := handler.Shutdown(drainCtx)
	stop()
	if drainErr != nil {
		_ = json.NewEncoder(out).Encode(struct {
			State string `json:"state"`
		}{"shutdown grace expired; waiting for owned work, resources retained"})
		// Cooperative callbacks cannot be safely abandoned. Operator SIGKILL remains
		// possible, but this path never claims a graceful drain while work survives.
		if e := handler.Shutdown(context.Background()); e != nil {
			return e
		}
	}
	if done != nil {
		serveErr = <-done
	}
	if serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) && !errors.Is(serveErr, net.ErrClosed) {
		return fmt.Errorf("listener stopped unexpectedly")
	}
	return nil
}

type cappedListener struct {
	net.Listener
	slots  chan struct{}
	closed chan struct{}
	once   sync.Once
}

func newCappedListener(l net.Listener, n int) *cappedListener {
	return &cappedListener{Listener: l, slots: make(chan struct{}, n), closed: make(chan struct{})}
}
func (l *cappedListener) Accept() (net.Conn, error) {
	select {
	case l.slots <- struct{}{}:
	case <-l.closed:
		return nil, net.ErrClosed
	}
	conn, e := l.Listener.Accept()
	if e != nil {
		<-l.slots
		return nil, e
	}
	return &countedConn{Conn: conn, release: func() { <-l.slots }}, nil
}
func (l *cappedListener) Close() error {
	l.once.Do(func() { close(l.closed) })
	return l.Listener.Close()
}

type countedConn struct {
	net.Conn
	once    sync.Once
	release func()
}

func (c *countedConn) Close() error { e := c.Conn.Close(); c.once.Do(c.release); return e }
