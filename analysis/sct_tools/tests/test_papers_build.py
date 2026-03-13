"""Regression tests for the LaTeX build bootstrap."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[3]
BUILD_SCRIPT = ROOT / "papers" / "build.py"


def _load_build_module():
    spec = importlib.util.spec_from_file_location("sct_papers_build", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_find_fontconfig_file_prefers_existing_env_value(tmp_path, monkeypatch):
    build = _load_build_module()
    config = tmp_path / "fonts.conf"
    config.write_text("<fontconfig/>", encoding="utf-8")
    monkeypatch.setenv("FONTCONFIG_FILE", str(config))
    monkeypatch.setattr(build, "_WINDOWS_FONTCONFIG_CANDIDATES", ())
    assert build.find_fontconfig_file() == str(config)


def test_find_fontconfig_file_falls_back_to_known_windows_candidate(tmp_path, monkeypatch):
    build = _load_build_module()
    config = tmp_path / "fonts.conf"
    config.write_text("<fontconfig/>", encoding="utf-8")
    monkeypatch.delenv("FONTCONFIG_FILE", raising=False)
    monkeypatch.setattr(build, "_IS_WINDOWS", True)
    monkeypatch.setattr(build, "_WINDOWS_FONTCONFIG_CANDIDATES", (config,))
    assert build.find_fontconfig_file() == str(config)


def test_compile_tex_passes_fontconfig_env_to_subprocess(tmp_path):
    build = _load_build_module()
    tex = tmp_path / "smoke.tex"
    tex.write_text(r"\documentclass{article}\begin{document}ok\end{document}", encoding="utf-8")
    mock_proc = MagicMock(returncode=0, stdout="", stderr="")

    with (
        patch.object(
            build, "build_tectonic_env", return_value={"FONTCONFIG_FILE": "C:/fonts.conf"}
        ),
        patch.object(build.subprocess, "run", return_value=mock_proc) as mock_run,
    ):
        ok, _, _ = build.compile_tex(tex, "tectonic")

    assert ok is True
    assert mock_run.call_args.kwargs["env"]["FONTCONFIG_FILE"] == "C:/fonts.conf"
