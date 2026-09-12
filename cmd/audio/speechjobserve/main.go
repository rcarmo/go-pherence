package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"
)

const usage = `speechjobserve --config /absolute/server.json [--check]
SPEECHJOB_TOKEN is required to serve (environment only, never a flag).
--check validates configuration/file identities and model metadata only; it does
not load tensor payloads, create a store or listen. Serving requires an explicit
allow_execution=true configuration, CPU-only startup environment and local pinned
assets. No downloads, existing-service changes or automatic job runs occur.
Read cmd/audio/speechjobserve/README.md before enabling execution.
`

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if e := command(ctx, os.Args[1:], os.Stdout); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
func command(ctx context.Context, args []string, out io.Writer) error {
	f := flag.NewFlagSet("speechjobserve", flag.ContinueOnError)
	f.SetOutput(io.Discard)
	config := f.String("config", "", "")
	check := f.Bool("check", false, "")
	if e := f.Parse(args); e != nil {
		if e == flag.ErrHelp {
			_, e = io.WriteString(out, usage)
			return e
		}
		return fmt.Errorf("invalid flags; use --help")
	}
	if f.NArg() != 0 || *config == "" {
		return fmt.Errorf("--config is required; use --help")
	}
	return start(ctx, *config, *check, out)
}
