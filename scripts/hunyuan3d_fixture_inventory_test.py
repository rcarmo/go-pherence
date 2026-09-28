#!/usr/bin/env python3
"""Test that header-only inventory never consumes a full ignored-Range response."""

from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from hunyuan3d_fixture_inventory import fetch_bytes, resolve_url


class Response:
    def __init__(self, status: int, content_range: str, data: bytes):
        self.status = status
        self.headers = {"Content-Range": content_range}
        self.stream = io.BytesIO(data)
        self.read_sizes: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, size=-1):
        self.read_sizes.append(size)
        return self.stream.read(size)


class TestBoundedHeaderRange(unittest.TestCase):
    def test_pinned_resolve_url(self):
        self.assertEqual(
            resolve_url("tencent/Hunyuan3D-2mini", "sub/model.fp16.safetensors", "f90a0f7d"),
            "https://huggingface.co/tencent/Hunyuan3D-2mini/resolve/f90a0f7d/sub/model.fp16.safetensors",
        )

    def test_valid_range(self):
        response = Response(206, "bytes 8-11/99", b"abcd")
        with patch("urllib.request.urlopen", return_value=response):
            self.assertEqual(fetch_bytes("https://example.test/model", (8, 11)), b"abcd")
        self.assertEqual(response.read_sizes, [5])

    def test_ignored_range_does_not_read(self):
        response = Response(200, "", b"entire model")
        with patch("urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "not honoured"):
                fetch_bytes("https://example.test/model", (0, 7))
        self.assertEqual(response.read_sizes, [])

    def test_wrong_offset_does_not_read(self):
        response = Response(206, "bytes 0-7/99", b"abcdefgh")
        with patch("urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "not honoured"):
                fetch_bytes("https://example.test/model", (8, 15))
        self.assertEqual(response.read_sizes, [])

    def test_oversize_range_is_bounded(self):
        response = Response(206, "bytes 0-7/99", b"a" * 128)
        with patch("urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "returned 9 bytes"):
                fetch_bytes("https://example.test/model", (0, 7))
        self.assertEqual(response.read_sizes, [9])


if __name__ == "__main__":
    unittest.main()
