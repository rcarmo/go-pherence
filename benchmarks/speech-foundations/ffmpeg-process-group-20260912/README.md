# FFmpeg process-group cancellation — 12 September 2026

This model-free checkpoint tightens the temporary FFmpeg/ffprobe runner on Linux.

## Contract

- Every default media subprocess starts in a fresh process group.
- Context cancellation sends `SIGKILL` to that owned group, covering descendants rather than only the direct child.
- Linux `Pdeathsig=SIGKILL` covers abrupt Go-parent death for the direct child; descendants remain governed by their isolated group while the parent is alive.
- The runner still waits for the direct child and reports the caller context error after cancellation.
- Injected test runners retain their existing interface and must provide equivalent cancellation/drain behavior.
- Non-Linux behavior remains the existing direct `os/exec` cancellation path.
- This is lifecycle containment, not hard CPU, RSS or disk quota enforcement; those remain launcher/cgroup responsibilities.

## Verification

- A real shell subprocess starts with PID equal to PGID.
- A real child that forks delayed work is cancelled; the descendant cannot write its delayed marker and is absent or zombie-only afterward.
- Existing media and speech-job cancellation/FFmpeg tests remain the regression boundary.

No media, model, GPU, service, deployment, dependency pin or default inference policy changed.
