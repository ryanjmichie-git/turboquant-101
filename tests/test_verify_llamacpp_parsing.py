"""Fixtures for verify.py's llama.cpp KV-size parsing. The M4 validation
pass found the old flat sum matched the "KV self size" total AND its
same-line K/V components, reporting exactly 2x the real MiB. These pin
the category-priority behavior."""

from conftest import load_verify

verify = load_verify()

# Real-world shape (llama.cpp ~build 9960, -lv 4): summary line repeats
# the components, and a per-backend buffer line reports the same total.
M4_STYLE_LOG = [
    "llama_context: KV self size  = 1152.00 MiB, "
    "K (f16):  576.00 MiB, V (f16):  576.00 MiB",
    "load_tensors: Metal KV buffer size =  1152.00 MiB",
]


def test_no_double_counting_on_m4_style_log():
    assert verify._extract_kv_mib(M4_STYLE_LOG) == 1152.0


def test_buffer_lines_summed_across_devices():
    lines = [
        "llama_kv_cache: CPU KV buffer size =   128.00 MiB",
        "llama_kv_cache: Metal KV buffer size =  1024.00 MiB",
    ]
    assert verify._extract_kv_mib(lines) == 1152.0


def test_self_size_used_when_no_buffer_lines():
    lines = ["llama_new_context: KV self size  =  612.00 MiB, "
             "K (q8_0):  306.00 MiB, V (q8_0):  306.00 MiB"]
    assert verify._extract_kv_mib(lines) == 612.0


def test_component_fallback_when_no_totals():
    lines = ["init: K (q4_0):  162.00 MiB", "init: V (q4_0):  162.00 MiB"]
    assert verify._extract_kv_mib(lines) == 324.0


def test_no_matches_returns_none():
    assert verify._extract_kv_mib(["main: server is listening"]) is None
    assert verify._extract_kv_mib([]) is None
