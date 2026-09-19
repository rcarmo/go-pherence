//go:build linux && amd64

package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"time"

	"github.com/rcarmo/go-pherence/runtime/resourcebudget"
	"github.com/rcarmo/go-pherence/runtime/speechjob"
	"github.com/rcarmo/go-pherence/runtime/speechjob/httpapi"
)

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
