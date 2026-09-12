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

	"github.com/rcarmo/go-pherence/runtime/speechjob"
	"github.com/rcarmo/go-pherence/runtime/speechjob/httpapi"
)

func start(ctx context.Context, path string, check bool, out io.Writer) error {
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
	// Exclude another server process on this same store before allocating model
	// weights. This does not coordinate other stores or unrelated inference.
	profiles, e := buildProfile(ctx, cfg, true)
	if e != nil {
		store.Close()
		return e
	}
	handler, e := httpapi.New(httpapi.Config{Store: store, Profiles: profiles, Token: os.Getenv("SPEECHJOB_TOKEN"), Hosts: cfg.HTTP.Hosts, Origin: cfg.HTTP.Origin, EnableUI: cfg.HTTP.EnableUI, MaxUploadBytes: cfg.Limits.UploadBytes, MaxConcurrentRequests: cfg.HTTP.MaxRequests})
	if e != nil {
		store.Close()
		return e
	}
	listener, e := net.Listen("tcp", cfg.HTTP.Listen)
	if e != nil {
		handler.Shutdown(context.Background())
		store.Close()
		return fmt.Errorf("listener bind failed")
	}
	// serveOwned does not return until handler-owned operations stop. The profiles'
	// model closures stay reachable through handler until then; no GC/close early.
	e = serveOwned(ctx, listener, handler, cfg.HTTP, tlsConfig, out)
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
	}{listener.Addr().String(), "explicit synchronous requests"}); e != nil {
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
