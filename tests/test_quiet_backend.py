"""The vLLM backend is quiet by default and shuts its engine down on
request, so demo/benchmark results print AFTER vLLM's teardown output.
Uses a fake `vllm` module -- no GPU needed."""

import sys
import types

import pytest

import niah.backends as backends


class _FakeEngine:
    def __init__(self):
        self.shutdowns = 0

    def shutdown(self):
        self.shutdowns += 1


class _FakeLLM:
    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.llm_engine = _FakeEngine()
        _FakeLLM.last = self

    def get_tokenizer(self):
        return types.SimpleNamespace(encode=lambda text: text.split())


QUIET_VARS = ("HF_HUB_VERBOSITY", "TRANSFORMERS_VERBOSITY",
              "FLASHINFER_LOGGING_LEVEL", "TQDM_DISABLE")


@pytest.fixture
def fake_vllm(monkeypatch):
    mod = types.ModuleType("vllm")
    mod.LLM = _FakeLLM
    mod.SamplingParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", mod)
    for var in ("VLLM_LOGGING_LEVEL", *QUIET_VARS):
        monkeypatch.delenv(var, raising=False)
    return mod


def test_quiet_by_default(fake_vllm):
    backends.VLLMBackend()
    import os
    assert os.environ["VLLM_LOGGING_LEVEL"] == "ERROR"


def test_verbose_restores_info(fake_vllm):
    backends.VLLMBackend(verbose=True)
    import os
    assert os.environ["VLLM_LOGGING_LEVEL"] == "INFO"


def test_user_setting_wins(fake_vllm, monkeypatch):
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "DEBUG")
    backends.VLLMBackend()
    import os
    assert os.environ["VLLM_LOGGING_LEVEL"] == "DEBUG"


def test_close_shuts_engine_down_once(fake_vllm):
    b = backends.VLLMBackend()
    engine = _FakeLLM.last.llm_engine
    b.close()
    b.close()  # idempotent
    assert engine.shutdowns == 1
    assert b.llm is None


def test_close_swallows_teardown_errors(fake_vllm):
    b = backends.VLLMBackend()

    def boom():
        raise ImportError("libnvrtc.so.13: cannot open shared object file")

    _FakeLLM.last.llm_engine.shutdown = boom
    b.close()  # must not raise: results are already in hand


def test_llamacpp_backend_has_close():
    backends.LlamaServerBackend().close()


def test_quiet_mode_silences_non_vllm_noise(fake_vllm):
    """HF_TOKEN notice, FlashInfer autotuner lines and progress bars ignore
    VLLM_LOGGING_LEVEL; quiet mode must reach them too."""
    import os
    backends.VLLMBackend()
    assert os.environ["HF_HUB_VERBOSITY"] == "error"
    assert os.environ["FLASHINFER_LOGGING_LEVEL"] == "error"
    assert os.environ["TQDM_DISABLE"] == "1"
    assert _FakeLLM.last.kwargs["use_tqdm_on_load"] is False


def test_verbose_leaves_other_noise_alone(fake_vllm):
    import os
    backends.VLLMBackend(verbose=True)
    for var in QUIET_VARS:
        assert var not in os.environ
    assert "use_tqdm_on_load" not in _FakeLLM.last.kwargs


def test_user_can_override_tqdm_flag(fake_vllm):
    backends.VLLMBackend(extra_engine_kwargs={"use_tqdm_on_load": True})
    assert _FakeLLM.last.kwargs["use_tqdm_on_load"] is True
