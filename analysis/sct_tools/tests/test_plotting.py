"""Tests for sct_tools.plotting — publication-quality plotting defaults."""

import os
import tempfile

import matplotlib

matplotlib.use('Agg')  # non-interactive backend for tests
import matplotlib.pyplot as plt
import numpy as np
import pytest

from sct_tools import plotting


class TestStyleInit:
    def test_init_style_runs(self):
        """init_style() should not raise."""
        plotting.init_style()

    def test_init_style_sets_font_size(self):
        plotting.init_style(font_size=12)
        assert plt.rcParams['font.size'] == 12

    def test_init_style_sets_dpi(self):
        plotting.init_style(dpi=150)
        assert plt.rcParams['figure.dpi'] == 150

    def test_init_style_usetex_false(self):
        plotting.init_style(usetex=False)
        assert plt.rcParams['text.usetex'] is False

    def test_init_style_line_width(self):
        plotting.init_style()
        assert plt.rcParams['lines.linewidth'] == 1.5

    def test_scienceplots_detected(self):
        """SciencePlots availability flag should be boolean."""
        assert isinstance(plotting._SCIENCEPLOTS_OK, bool)


class TestSCTColors:
    def test_all_colors_exist(self):
        expected_keys = ['scalar', 'dirac', 'vector', 'total',
                         'prediction', 'data', 'theory_band', 'reference']
        for key in expected_keys:
            assert key in plotting.SCT_COLORS

    def test_colors_are_hex(self):
        for name, color in plotting.SCT_COLORS.items():
            assert color.startswith('#'), f"{name} color not hex: {color}"
            assert len(color) == 7, f"{name} color wrong length: {color}"

    def test_scalar_is_blue(self):
        assert plotting.SCT_COLORS['scalar'] == '#2196F3'

    def test_dirac_is_red(self):
        assert plotting.SCT_COLORS['dirac'] == '#F44336'

    def test_vector_is_green(self):
        assert plotting.SCT_COLORS['vector'] == '#4CAF50'


class TestCreateFigure:
    def test_single_column(self):
        fig, ax = plotting.create_figure()
        w, h = fig.get_size_inches()
        assert w == pytest.approx(3.4, abs=0.1)
        plt.close(fig)

    def test_double_column(self):
        fig, axes = plotting.create_figure(1, 2)
        w, h = fig.get_size_inches()
        assert w == pytest.approx(7.0, abs=0.1)
        plt.close(fig)

    def test_custom_figsize(self):
        fig, ax = plotting.create_figure(figsize=(5, 4))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(5.0, abs=0.1)
        assert h == pytest.approx(4.0, abs=0.1)
        plt.close(fig)

    def test_grid_layout(self):
        fig, axes = plotting.create_figure(2, 3, figsize=(7, 5))
        assert axes.shape == (2, 3)
        plt.close(fig)

    def test_squeeze_single(self):
        fig, ax = plotting.create_figure(1, 1, squeeze=True)
        # Single axes, not array
        assert not isinstance(ax, np.ndarray)
        plt.close(fig)


class TestSaveFigure:
    def test_save_pdf(self):
        plotting.init_style()
        fig, ax = plotting.create_figure()
        ax.plot([1, 2, 3], [1, 4, 9])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = plotting.save_figure(fig, "test_plot", fmt="pdf",
                                            directory=tmpdir)
            assert os.path.exists(filepath)
            assert filepath.suffix == '.pdf'
        plt.close(fig)

    def test_save_png(self):
        fig, ax = plotting.create_figure()
        ax.plot([1, 2], [1, 2])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = plotting.save_figure(fig, "test_png", fmt="png",
                                            directory=tmpdir)
            assert os.path.exists(filepath)
            assert filepath.suffix == '.png'
        plt.close(fig)

    def test_save_svg(self):
        fig, ax = plotting.create_figure()
        ax.plot([0], [0])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = plotting.save_figure(fig, "test_svg", fmt="svg",
                                            directory=tmpdir)
            assert os.path.exists(filepath)
        plt.close(fig)

    def test_default_directory(self):
        """Default save directory should be analysis/figures/."""
        fig, ax = plotting.create_figure()
        ax.plot([0], [0])
        filepath = plotting.save_figure(fig, "_test_default_dir")
        assert filepath.parent == plotting._FIGURES_DIR
        # Clean up
        if filepath.exists():
            filepath.unlink()
        plt.close(fig)


class TestPlotFormFactors:
    def test_returns_fig_and_axes(self):
        fig, (ax_C, ax_R) = plotting.plot_form_factors(n_points=10)
        assert fig is not None
        assert ax_C is not None
        assert ax_R is not None
        plt.close(fig)

    def test_has_labels(self):
        fig, (ax_C, ax_R) = plotting.plot_form_factors(n_points=10)
        # Both axes should have x and y labels
        assert ax_C.get_xlabel() != ''
        assert ax_C.get_ylabel() != ''
        assert ax_R.get_xlabel() != ''
        assert ax_R.get_ylabel() != ''
        plt.close(fig)

    def test_custom_range(self):
        fig, _ = plotting.plot_form_factors(x_range=(1, 10), n_points=5)
        plt.close(fig)  # Just check no crash


class TestPlotSpectralDimension:
    def test_returns_fig_ax(self):
        fig, ax = plotting.plot_spectral_dimension(n_points=10)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_xscale_log(self):
        fig, ax = plotting.plot_spectral_dimension(n_points=10)
        assert ax.get_xscale() == 'log'
        plt.close(fig)


class TestPlotResiduals:
    def test_without_errors(self):
        x = [1, 2, 3]
        y_data = [1.1, 2.0, 2.9]
        y_model = [1.0, 2.0, 3.0]
        fig, ax = plotting.plot_residuals(x, y_data, y_model)
        assert fig is not None
        plt.close(fig)

    def test_with_errors(self):
        x = [1, 2, 3]
        y_data = [1.1, 2.0, 2.9]
        y_model = [1.0, 2.0, 3.0]
        yerr = [0.1, 0.1, 0.1]
        fig, ax = plotting.plot_residuals(x, y_data, y_model, yerr=yerr)
        assert fig is not None
        plt.close(fig)


class TestPlotComparisonTable:
    def test_basic_comparison(self):
        labels = ["obs1", "obs2"]
        sct = [1.0, 2.0]
        competitors = {"LQG": [0.9, 2.1]}
        data = [1.05, 1.95]
        fig, ax = plotting.plot_comparison_table(labels, sct, competitors, data)
        assert fig is not None
        plt.close(fig)

    def test_with_errors(self):
        labels = ["a", "b"]
        sct = [1.0, 2.0]
        competitors = {"AS": [1.1, 1.9]}
        data = [1.0, 2.0]
        errors = [0.1, 0.15]
        fig, ax = plotting.plot_comparison_table(labels, sct, competitors,
                                                  data, data_errors=errors)
        assert fig is not None
        plt.close(fig)


class TestAnnotateHelpers:
    def test_annotate_theory_version(self):
        fig, ax = plotting.create_figure()
        ax.plot([0, 1], [0, 1])
        plotting.annotate_theory_version(ax, version="SCT v0.7")
        # No crash = pass
        plt.close(fig)

    def test_annotate_prediction(self):
        fig, ax = plotting.create_figure()
        ax.plot([0, 10], [0, 10])
        plotting.annotate_prediction(ax, 5, 5, "Test prediction")
        plt.close(fig)

    def test_annotate_with_custom_date(self):
        fig, ax = plotting.create_figure()
        plotting.annotate_theory_version(ax, date="2026-03-11")
        plt.close(fig)
