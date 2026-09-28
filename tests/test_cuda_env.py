"""niah.cuda_env puts pip's CUDA 13 lib folder on LD_LIBRARY_PATH (where
libnvrtc.so.13 lives but the system loader doesn't look), without
duplicating it and without touching anything when the folder is absent."""

import sysconfig

import niah.cuda_env as cuda_env


def _fake_purelib(monkeypatch, root):
    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"purelib": str(root)})


def test_prepends_once(monkeypatch, tmp_path):
    lib = tmp_path / "nvidia" / "cu13" / "lib"
    lib.mkdir(parents=True)
    _fake_purelib(monkeypatch, tmp_path)
    env = {"LD_LIBRARY_PATH": "/usr/local/lib"}
    cuda_env.with_cu13_libs(env)
    cuda_env.with_cu13_libs(env)  # idempotent
    assert env["LD_LIBRARY_PATH"] == f"{lib}:/usr/local/lib"


def test_sets_when_unset(monkeypatch, tmp_path):
    lib = tmp_path / "nvidia" / "cu13" / "lib"
    lib.mkdir(parents=True)
    _fake_purelib(monkeypatch, tmp_path)
    env = {}
    cuda_env.with_cu13_libs(env)
    assert env["LD_LIBRARY_PATH"] == str(lib)


def test_noop_without_cu13_folder(monkeypatch, tmp_path):
    _fake_purelib(monkeypatch, tmp_path)
    env = {}
    cuda_env.with_cu13_libs(env)
    assert "LD_LIBRARY_PATH" not in env
