from pathlib import Path

import numpy as np
import pytest
import plotly.graph_objects as go

from cgap_sbl_jadce.gui.charts import _line_style

playwright = pytest.importorskip('playwright.sync_api')


@pytest.mark.parametrize('log', [False, True])
def test_browser_line_zoom_is_unrestricted_and_reset_restores_default_view(log):
    with playwright.sync_playwright() as driver:
        if not Path(driver.chromium.executable_path).exists():
            pytest.skip('Chromium is not installed')
        try:
            browser = driver.chromium.launch(headless=True, args=['--no-sandbox'])
        except playwright.Error as error:
            pytest.skip(f'Chromium is unavailable: {error.message.splitlines()[0]}')
        try:
            page = browser.new_page(viewport={'width': 1000, 'height': 800})
            fig = _line_style(go.Figure(go.Scatter(x=[4, 8, 12], y=[.001, .01, .1], mode='lines+markers')),
                              'Test', 'SNR', 'NMSE', log=log, xlim=(4, 24))
            html = fig.to_html(include_plotlyjs=True, full_html=False, div_id='chart',
                               config={'scrollZoom': True, 'displaylogo': False})
            page.set_content('<div data-testid="stPlotlyChart">' + html + '</div>')
            page.wait_for_function("document.getElementById('chart')._fullLayout?.yaxis?.range")
            ranges = "() => {const f = document.getElementById('chart')._fullLayout; return [f.xaxis.range, f.yaxis.range];}"
            initial = page.evaluate(ranges)
            np.testing.assert_allclose(initial[0], [4, 24])
            page.locator('[data-title="Zoom out"]').click()
            for current, default in zip(page.evaluate(ranges), initial):
                assert current[0] < default[0] < default[1] < current[1]
            page.locator('[data-title="Reset axes"]').click()
            np.testing.assert_allclose(page.evaluate(ranges), initial)
            page.locator('[data-title="Zoom in"]').click()
            for current, default in zip(page.evaluate(ranges), initial):
                assert default[0] < current[0] < current[1] < default[1]
            page.locator('[data-title="Reset axes"]').click()
            np.testing.assert_allclose(page.evaluate(ranges), initial)
        finally:
            browser.close()


def test_browser_square_matrix_cells_at_different_widths():
    from cgap_sbl_jadce.gui.charts import _square_matrix, _style
    with playwright.sync_playwright() as driver:
        if not Path(driver.chromium.executable_path).exists():
            pytest.skip('Chromium is not installed')
        try:
            browser = driver.chromium.launch(headless=True, args=['--no-sandbox'])
        except playwright.Error as error:
            pytest.skip(f'Chromium is unavailable: {error.message.splitlines()[0]}')
        try:
            page = browser.new_page()
            heatmaps = [_square_matrix(_style(go.Figure(go.Heatmap(z=np.eye(n))), 'Square'), (n, n))
                        for n in (6, 20)]
            for width in (400, 1000):
                page.set_viewport_size({'width': width, 'height': 800})
                for fig in heatmaps:
                    page.set_content(fig.to_html(include_plotlyjs=True, full_html=False, div_id='chart'))
                    page.wait_for_function("document.getElementById('chart')._fullLayout?.yaxis?._length")
                    width_px, height_px = page.evaluate("() => {const f = document.getElementById('chart')._fullLayout; return [f.xaxis._length, f.yaxis._length];}")
                    assert width_px == pytest.approx(height_px, abs=.1)
        finally:
            browser.close()
