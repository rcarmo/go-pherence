# Jevlike provenance

This document records the upstream origin and attribution of the Jevlike port in
`go-pherence`. It is not a replacement for the repository's top-level licence.

- Upstream: <https://github.com/taubinator/jevlike>
- Reference revision: `94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452`
- Upstream copyright holder: Minimal Labs
- Upstream licence: MIT

The Go implementation adapts the upstream byte tokenisation, data builders,
tiny encoder and variable-option attention scorer. Native gradients, training,
checkpoint handling and integration code are developed for `go-pherence`.
The upstream Python implementation is the behavioural reference; this record
alone does not claim complete feature or numerical parity.

The upstream copyright and permission notice is retained in full below to
satisfy the attribution requirements for adapted code.

## Upstream licence notice

```text
MIT License

Copyright (c) 2026 Minimal Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
