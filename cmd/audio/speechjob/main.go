// speechjob is an HTTP client, not an inference server or model runner.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if e := execute(ctx, os.Args[1:], os.Getenv, os.Stdout); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}

const usage = `speechjob [--url https://host:port] [--timeout 30s] [--allow-loopback-http] COMMAND
Token: SPEECHJOB_TOKEN environment variable (no token command-line flag).
Commands:
  upload --profile ID --file FILE [--name NAME]
  list [--after JOB_ID]
  get JOB_ID
  inventory
  queue                   list durable queue tickets (if enabled)
  enqueue JOB_ID          durable intent only; worker startup is operator-owned
  retry-queued JOB_ID     explicit retry of failed/cancelled/interrupted ticket
  forget-queued JOB_ID    remove terminal ticket metadata, not media
  run JOB_ID               synchronous; choose an adequate --timeout
  cancel JOB_ID            acknowledgement means requested, not finished
  delete --confirm JOB_ID  irreversible; no automatic retry
  download --job JOB_ID --artifact NAME --out FILE
Download names: transcript, vtt, speaker-transcript, speaker-vtt.
Existing output files are never replaced. No command retries automatically.
Global flags precede COMMAND. HTTPS verifies system trust; plain HTTP requires
--allow-loopback-http and a literal loopback IP. Redirects/proxies are disabled.
`

func execute(ctx context.Context, args []string, env func(string) string, out io.Writer) error {
	fs := flag.NewFlagSet("speechjob", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	endpoint := fs.String("url", env("SPEECHJOB_URL"), "")
	timeout := fs.Duration("timeout", 30*time.Second, "")
	loopback := fs.Bool("allow-loopback-http", false, "")
	if len(args) == 0 {
		_, e := io.WriteString(out, usage)
		return e
	}
	if e := fs.Parse(args); e != nil {
		if e == flag.ErrHelp {
			_, e = io.WriteString(out, usage)
			return e
		}
		return fmt.Errorf("invalid flags; use --help")
	}
	if *timeout <= 0 || *timeout > 8*time.Hour {
		return fmt.Errorf("timeout must be positive and at most 8h")
	}
	args = fs.Args()
	if len(args) == 0 {
		return fmt.Errorf("missing command; use --help")
	}
	c, e := newClient(*endpoint, env("SPEECHJOB_TOKEN"), *loopback)
	if e != nil {
		return e
	}
	defer c.close()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	switch args[0] {
	case "get", "run", "cancel", "enqueue", "retry-queued", "forget-queued":
		if len(args) != 2 || !validID(args[1]) {
			return fmt.Errorf("command requires one lowercase job ID")
		}
		method, path, status := "GET", "/v1/jobs/"+args[1], 200
		if args[0] != "get" {
			method = "POST"
			path += "/" + args[0]
			if args[0] == "cancel" || args[0] == "enqueue" || args[0] == "retry-queued" {
				status = 202
			}
			if args[0] == "forget-queued" {
				method = "DELETE"
				path = "/v1/jobs/" + args[1] + "/queue"
				status = 204
			}
		}
		return c.jsonRequest(ctx, method, path, nil, 0, status, out)
	case "queue":
		if len(args) != 1 {
			return fmt.Errorf("queue takes no arguments")
		}
		return c.jsonRequest(ctx, "GET", "/v1/queue", nil, 0, 200, out)
	case "inventory":
		if len(args) != 1 {
			return fmt.Errorf("inventory takes no arguments")
		}
		return c.jsonRequest(ctx, "GET", "/v1/inventory", nil, 0, 200, out)
	case "list":
		f := quietFlags("list")
		after := f.String("after", "", "")
		if e = f.Parse(args[1:]); e != nil || f.NArg() != 0 || *after != "" && !validID(*after) {
			return fmt.Errorf("invalid list cursor")
		}
		path := "/v1/jobs"
		if *after != "" {
			path += "?after=" + *after
		}
		return c.jsonRequest(ctx, "GET", path, nil, 0, 200, out)
	case "upload":
		f := quietFlags("upload")
		profile := f.String("profile", "", "")
		file := f.String("file", "", "")
		name := f.String("name", "", "")
		if e = f.Parse(args[1:]); e != nil || f.NArg() != 0 {
			return fmt.Errorf("invalid upload flags")
		}
		return c.upload(ctx, *profile, *file, *name, out)
	case "delete":
		f := quietFlags("delete")
		confirmed := f.Bool("confirm", false, "")
		if e = f.Parse(args[1:]); e != nil || !*confirmed || f.NArg() != 1 || !validID(f.Arg(0)) {
			return fmt.Errorf("delete requires --confirm followed by one job ID")
		}
		return c.jsonRequest(ctx, "DELETE", "/v1/jobs/"+f.Arg(0), nil, 0, 204, out)
	case "download":
		f := quietFlags("download")
		job := f.String("job", "", "")
		artifact := f.String("artifact", "", "")
		file := f.String("out", "", "")
		if e = f.Parse(args[1:]); e != nil || f.NArg() != 0 {
			return fmt.Errorf("invalid download flags")
		}
		return c.download(ctx, *job, *artifact, *file, out)
	default:
		return fmt.Errorf("unknown command; use --help")
	}
}
func quietFlags(name string) *flag.FlagSet {
	f := flag.NewFlagSet(name, flag.ContinueOnError)
	f.SetOutput(io.Discard)
	return f
}
