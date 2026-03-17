"""Regression tests for README master-function wording and asymptotics."""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
FIGURE_SCRIPT = ROOT / "docs" / "figures" / "generate_readme_figures.py"
README = ROOT / "README.md"
B4_TEX = ROOT / "theory" / "consistency-checks" / "B4_ghost_inevitability.tex"


def _load_figure_module():
    spec = importlib.util.spec_from_file_location("sct_readme_figures", FIGURE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_master_function_figure_states_limit_direction_explicitly():
    module = _load_figure_module()

    fig, ax = module.build_master_function_figure()
    text_fragments = [text.get_text() for text in ax.texts]
    plt.close(fig)

    assert any(r"x \to -\infty" in text for text in text_fragments)
    assert any(r"\sqrt{\pi/|x|}" in text for text in text_fragments)
    assert any(r"x \to +\infty" in text for text in text_fragments)
    assert any(r"2/x" in text for text in text_fragments)


def test_readme_explains_left_and_right_master_function_branches():
    readme = README.read_text(encoding="utf-8")

    assert "left branch" in readme
    assert "right branch" in readme
    assert "x < 0" in readme
    assert "x > 0" in readme
    assert "x -> -infinity" in readme
    assert "x -> +infinity" in readme


def test_b4_uses_correct_negative_axis_prefactor():
    content = B4_TEX.read_text(encoding="utf-8")

    assert r"\sqrt{\pi/|z|}" in content
    assert r"\sqrt{2\pi/|z|}" not in content
