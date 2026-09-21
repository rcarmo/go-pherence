// Command transcribe-web exposes the durable speechjobserve queue through a
// trusted-LAN, no-login browser interface. The child API remains bearer-protected
// on private loopback; the random token never crosses the public listener.
package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

//go:embed web/*
var web embed.FS

type options struct {
	listen, hosts, backend, child, config string
	startup                               time.Duration
}

func parseOptions(args []string) (options, error) {
	var o options
	f := flag.NewFlagSet("transcribe-web", flag.ContinueOnError)
	f.SetOutput(io.Discard)
	f.StringVar(&o.listen, "listen", "[::]:8093", "trusted-LAN listen address")
	f.StringVar(&o.hosts, "hosts", "sigma.local:8093,192.168.1.70:8093,127.0.0.1:8093,localhost:8093", "comma-separated Host allowlist")
	f.StringVar(&o.backend, "backend", "http://127.0.0.1:18093", "private speechjobserve origin")
	f.StringVar(&o.child, "speechjobserve", "", "absolute speechjobserve executable")
	f.StringVar(&o.config, "config", "", "absolute speechjobserve configuration")
	f.DurationVar(&o.startup, "startup-timeout", 5*time.Minute, "child readiness timeout")
	if err := f.Parse(args); err != nil || f.NArg() != 0 {
		return o, fmt.Errorf("invalid flags")
	}
	if o.child == "" || o.config == "" {
		return o, fmt.Errorf("--speechjobserve and --config are required")
	}
	if err := absoluteRegular(o.child, true); err != nil {
		return o, fmt.Errorf("speechjobserve executable rejected: %w", err)
	}
	if err := absoluteRegular(o.config, false); err != nil {
		return o, fmt.Errorf("configuration rejected: %w", err)
	}
	backend, err := url.Parse(o.backend)
	if err != nil || backend.Scheme != "http" || backend.Path != "" || backend.RawQuery != "" || backend.Fragment != "" || backend.User != nil || backend.Port() == "" {
		return o, fmt.Errorf("backend must be a private loopback HTTP origin")
	}
	ip := net.ParseIP(backend.Hostname())
	if ip == nil || !ip.IsLoopback() {
		return o, fmt.Errorf("backend must use a literal loopback address")
	}
	if o.startup < time.Second || o.startup > 30*time.Minute {
		return o, fmt.Errorf("invalid startup timeout")
	}
	if _, err := hostSet(o.hosts); err != nil {
		return o, err
	}
	return o, nil
}

func absoluteRegular(path string, executable bool) error {
	if !filepath.IsAbs(path) {
		return fmt.Errorf("path is not absolute")
	}
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || executable && info.Mode()&0111 == 0 {
		return fmt.Errorf("path is not an eligible regular file")
	}
	return nil
}

func hostSet(raw string) (map[string]bool, error) {
	result := map[string]bool{}
	for _, host := range strings.Split(raw, ",") {
		host = strings.TrimSpace(host)
		u, err := url.Parse("http://" + host)
		if err != nil || host == "" || u.Host != host || u.Hostname() == "" || u.Port() == "" || strings.ContainsAny(host, " /\\?#@\t\r\n") || result[host] {
			return nil, fmt.Errorf("invalid Host allowlist")
		}
		result[host] = true
	}
	if len(result) == 0 || len(result) > 16 {
		return nil, fmt.Errorf("invalid Host allowlist size")
	}
	return result, nil
}

func randomToken() (string, error) {
	var token [32]byte
	if _, err := io.ReadFull(rand.Reader, token[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(token[:]), nil
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := run(ctx, os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	o, err := parseOptions(args)
	if err != nil {
		return err
	}
	hosts, _ := hostSet(o.hosts)
	backend, _ := url.Parse(o.backend)
	listener, err := net.Listen("tcp", o.listen)
	if err != nil {
		return fmt.Errorf("LAN listener: %w", err)
	}
	defer listener.Close()
	token, err := randomToken()
	if err != nil {
		return fmt.Errorf("internal credential: %w", err)
	}
	child, ready, childDone, err := startChild(ctx, o, token, stderr)
	if err != nil {
		return err
	}
	var childErr error
	childStopped := false
	stopChild := func() error {
		if childStopped {
			return childErr
		}
		if child.Process != nil {
			_ = child.Process.Signal(syscall.SIGTERM)
		}
		select {
		case childErr = <-childDone:
			childStopped = true
			return childErr
		case <-time.After(2 * time.Minute):
			if child.Process != nil {
				_ = child.Process.Kill()
			}
			return fmt.Errorf("speechjobserve did not drain before deadline")
		}
	}
	select {
	case <-ctx.Done():
		return errors.Join(ctx.Err(), stopChild())
	case childErr = <-childDone:
		childStopped = true
		return fmt.Errorf("speechjobserve exited before readiness: %w", childErr)
	case <-ready:
	case <-time.After(o.startup):
		return errors.Join(fmt.Errorf("speechjobserve readiness timeout"), stopChild())
	}
	if err = recoverInterrupted(ctx, backend, token); err != nil {
		return errors.Join(fmt.Errorf("queue recovery: %w", err), stopChild())
	}
	reconcileCtx, stopReconcile := context.WithCancel(ctx)
	defer stopReconcile()
	go reconcileLoop(reconcileCtx, backend, token, stderr)
	handler := newHandler(hosts, backend, token)
	server := &http.Server{Handler: handler, ReadHeaderTimeout: 10 * time.Second, ReadTimeout: 8 * time.Hour, WriteTimeout: 8 * time.Hour, IdleTimeout: 30 * time.Second, MaxHeaderBytes: 32 << 10}
	serveDone := make(chan error, 1)
	go func() { serveDone <- server.Serve(listener) }()
	_, _ = fmt.Fprintf(stdout, "{\"listening\":%q,\"backend\":%q,\"authentication\":\"trusted LAN; no login\"}\n", listener.Addr().String(), backend.String())
	var cause error
	select {
	case <-ctx.Done():
		cause = ctx.Err()
	case childErr = <-childDone:
		childStopped = true
		cause = fmt.Errorf("speechjobserve exited: %w", childErr)
	case err := <-serveDone:
		if !errors.Is(err, http.ErrServerClosed) {
			cause = err
		}
	}
	shutdown, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	serverErr := server.Shutdown(shutdown)
	childErr = stopChild()
	if errors.Is(cause, context.Canceled) {
		cause = nil
	}
	return errors.Join(cause, serverErr, childErr)
}

func startChild(ctx context.Context, o options, token string, stderr io.Writer) (*exec.Cmd, <-chan struct{}, chan error, error) {
	cmd := exec.Command(o.child, "--config", o.config)
	cmd.Env = append(os.Environ(), "SPEECHJOB_TOKEN="+token)
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, nil, err
	}
	cmd.Stderr = stderr
	if err = cmd.Start(); err != nil {
		return nil, nil, nil, err
	}
	ready := make(chan struct{})
	var readyOnce bool
	go func() {
		scanner := bufio.NewScanner(out)
		for scanner.Scan() {
			line := scanner.Text()
			if !readyOnce && strings.Contains(line, `"listening"`) {
				readyOnce = true
				close(ready)
			}
			_, _ = fmt.Fprintln(stderr, line)
		}
	}()
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	_ = ctx
	return cmd, ready, done, nil
}

func backendRequest(ctx context.Context, client *http.Client, backend *url.URL, token, method, path string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, backend.String()+path, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	return client.Do(req)
}

func recoverInterrupted(ctx context.Context, backend *url.URL, token string) error {
	client := &http.Client{Timeout: 30 * time.Second, Transport: &http.Transport{Proxy: nil, DisableCompression: true}}
	response, err := backendRequest(ctx, client, backend, token, http.MethodGet, "/v1/queue")
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK || response.Header.Get("Content-Type") != "application/json" {
		return fmt.Errorf("queue inventory HTTP %d", response.StatusCode)
	}
	var inventory struct {
		Entries []struct {
			JobID  string `json:"job_id"`
			Status string `json:"status"`
		} `json:"entries"`
	}
	decoder := json.NewDecoder(io.LimitReader(response.Body, 4<<20))
	if err = decoder.Decode(&inventory); err != nil || decoder.Decode(new(any)) != io.EOF || len(inventory.Entries) > 128 {
		return fmt.Errorf("invalid queue inventory")
	}
	for _, entry := range inventory.Entries {
		if entry.Status != "interrupted" {
			continue
		}
		if len(entry.JobID) != 32 {
			return fmt.Errorf("invalid interrupted job ID")
		}
		response, err = backendRequest(ctx, client, backend, token, http.MethodPost, "/v1/jobs/"+entry.JobID+"/retry-queued")
		if err != nil {
			return err
		}
		_, copyErr := io.Copy(io.Discard, io.LimitReader(response.Body, 4<<20))
		closeErr := response.Body.Close()
		if response.StatusCode != http.StatusAccepted || copyErr != nil || closeErr != nil {
			return fmt.Errorf("interrupted job retry HTTP %d", response.StatusCode)
		}
	}
	return nil
}

func reconcileLoop(ctx context.Context, backend *url.URL, token string, diagnostics io.Writer) {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		if err := reconcileTerminal(ctx, backend, token); err != nil && !errors.Is(err, context.Canceled) {
			_, _ = fmt.Fprintln(diagnostics, "queue reconciliation:", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func reconcileTerminal(ctx context.Context, backend *url.URL, token string) error {
	client := &http.Client{Timeout: 30 * time.Second, Transport: &http.Transport{Proxy: nil, DisableCompression: true}}
	response, err := backendRequest(ctx, client, backend, token, http.MethodGet, "/v1/queue")
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK || response.Header.Get("Content-Type") != "application/json" {
		return fmt.Errorf("queue inventory HTTP %d", response.StatusCode)
	}
	var inventory struct {
		Entries []struct {
			JobID  string `json:"job_id"`
			Status string `json:"status"`
		} `json:"entries"`
	}
	decoder := json.NewDecoder(io.LimitReader(response.Body, 4<<20))
	if err = decoder.Decode(&inventory); err != nil || decoder.Decode(new(any)) != io.EOF || len(inventory.Entries) > 128 {
		return fmt.Errorf("invalid queue inventory")
	}
	for _, entry := range inventory.Entries {
		if len(entry.JobID) != 32 {
			return fmt.Errorf("invalid terminal job ID")
		}
		var actions [][2]string
		switch entry.Status {
		case "succeeded":
			actions = [][2]string{{http.MethodPost, "/v1/jobs/" + entry.JobID + "/release-media"}, {http.MethodDelete, "/v1/jobs/" + entry.JobID + "/queue"}}
		case "cancelled":
			actions = [][2]string{{http.MethodDelete, "/v1/jobs/" + entry.JobID}}
		default:
			continue
		}
		for _, action := range actions {
			response, err = backendRequest(ctx, client, backend, token, action[0], action[1])
			if err != nil {
				return err
			}
			_, copyErr := io.Copy(io.Discard, io.LimitReader(response.Body, 4<<20))
			closeErr := response.Body.Close()
			if response.StatusCode != http.StatusNoContent || copyErr != nil || closeErr != nil {
				return fmt.Errorf("terminal cleanup HTTP %d", response.StatusCode)
			}
		}
	}
	return nil
}

func newHandler(hosts map[string]bool, backend *url.URL, token string) http.Handler {
	proxy := &httputil.ReverseProxy{
		Rewrite: func(request *httputil.ProxyRequest) {
			request.SetURL(backend)
			request.Out.URL.Path = strings.TrimPrefix(request.In.URL.Path, "/api")
			request.Out.URL.RawPath = ""
			request.Out.Host = backend.Host
			request.Out.Header.Del("Origin")
			request.Out.Header.Del("Cookie")
			request.Out.Header.Del("Sec-Fetch-Site")
			request.Out.Header.Set("Authorization", "Bearer "+token)
		},
		Transport: &http.Transport{Proxy: nil, DisableCompression: true, MaxIdleConns: 8, MaxIdleConnsPerHost: 8, IdleConnTimeout: 30 * time.Second},
		ErrorHandler: func(w http.ResponseWriter, _ *http.Request, _ error) {
			http.Error(w, `{"error":"backend_unavailable"}`, http.StatusBadGateway)
		},
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		securityHeaders(w)
		if !hosts[r.Host] {
			http.Error(w, "host rejected", http.StatusMisdirectedRequest)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/") {
			origin := "http://" + r.Host
			values := r.Header.Values("Origin")
			if len(values) > 1 || len(values) == 1 && values[0] != origin {
				http.Error(w, "origin rejected", http.StatusForbidden)
				return
			}
			if site := r.Header.Get("Sec-Fetch-Site"); site != "" && site != "same-origin" && site != "none" {
				http.Error(w, "cross-site request rejected", http.StatusForbidden)
				return
			}
			proxy.ServeHTTP(w, r)
			return
		}
		serveStatic(w, r)
	})
}

func securityHeaders(w http.ResponseWriter) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Referrer-Policy", "no-referrer")
	w.Header().Set("X-Frame-Options", "DENY")
	w.Header().Set("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
	w.Header().Set("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'; object-src 'none'")
}

func serveStatic(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead || r.URL.RawQuery != "" || r.URL.RawPath != "" {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	var name, contentType string
	switch r.URL.Path {
	case "/", "/index.html":
		name, contentType = "web/index.html", "text/html; charset=utf-8"
	case "/app.js":
		name, contentType = "web/app.js", "text/javascript; charset=utf-8"
	case "/style.css":
		name, contentType = "web/style.css", "text/css; charset=utf-8"
	default:
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	data, err := web.ReadFile(name)
	if err != nil {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Content-Length", fmt.Sprint(len(data)))
	if r.Method == http.MethodGet {
		_, _ = w.Write(data)
	}
}
