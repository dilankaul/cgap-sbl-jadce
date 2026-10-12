from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser
from cgap_sbl_jadce.figures import common  # applies shared typography


def glyph_fonts(label, family=None):
    prop = FontProperties(math_fontfamily=family) if family else FontProperties()
    return [(g[0].family_name, g[0].style_name)
            for g in MathTextParser('path').parse(label, prop=prop).glyphs]


def test_bold_latin_uses_native_computer_modern():
    for label in [r'$\mathbf{Z}$', r'$\mathbf{H}$', r'$\mathbf{C}$', r'$\mathbf{z}$']:
        assert glyph_fonts(label) == [('cmb10', 'Regular')]


def test_regular_math_stays_computer_modern():
    for label in [r'$123$', r'$x+y=P_i$', r'$\alpha\beta\tau\sigma$', r'$C_{i,j}$']:
        assert all(family.startswith('cm') for family, style in glyph_fonts(label))


def test_greek_boldsymbol_uses_true_stix_bold():
    assert glyph_fonts(r'$\boldsymbol{\Theta}$', 'stix') == [('STIXGeneral', 'Bold')]
    assert glyph_fonts(r'$\boldsymbol{\theta}$', 'stix') == [('STIXGeneral', 'Bold Italic')]
