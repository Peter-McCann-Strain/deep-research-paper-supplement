"""Shared matplotlib style for every figure in the paper.

Added 2026-07-27 after an adversarial figure-quality review found the figure
scripts had silently forked into two generations: five/six scripts already used
a colourblind-safe Okabe-Ito palette and print-legible font sizes, three others
(disentanglement, e5-dose-response, judge-gold) used ad hoc non-Okabe-Ito hex
colours and undersized fonts, and every script fell through to matplotlib's
bundled DejaVu Serif rather than a Times-family face matching the paper body
(newtxtext/TeXGyreTermesX). This module is the single source of truth going
forward; per-script rcParams overrides should only ever raise font sizes above
the defaults here to compensate for a specific figure's print scale factor
(wide two-panel figures shrink harder when placed at \\linewidth), never
introduce a different colour or font family.

Usage:
    from plot_style import apply_style, OKABE_ITO
    apply_style(base_size=13)   # pass a larger base_size for figures that are
                                # scaled down harder by their \\includegraphics width
"""
import matplotlib

# Okabe & Ito (2008) colourblind-safe palette. Only the first three are used as
# the paper's standing category convention (cluster / other / local-7B); the
# rest are available for figures needing more than three categories.
OKABE_ITO = {
    "blue": "#0072B2",
    "bluish_green": "#009E73",
    "vermillion": "#D55E00",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "yellow": "#F0E442",
    "purple": "#CC79A7",
    "black": "#000000",
}

# Standing category convention used across fig1_money / fig_cost / fig_oracle /
# fig_vintage / fig_stratification / fig_cd_clean: keep any new figure on this
# mapping rather than inventing a new one.
C_CLUSTER = OKABE_ITO["blue"]
C_OTHER = OKABE_ITO["bluish_green"]
C_LOCAL = OKABE_ITO["vermillion"]

# Times-family face matching the paper body (newtxtext/TeXGyreTermesX). Nimbus
# Roman is metrically compatible and is what TeX Gyre Termes is itself built
# from; falling back to DejaVu Serif only if neither is installed keeps the
# script from hard-crashing on a machine without these fonts, though the
# embedded PDF will then mismatch the paper body again.
FONT_SERIF_STACK = ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"]

# Every figure must clear this at its ACTUAL print scale (see each script's own
# scale-factor comment for the true shrink from includegraphics width vs the
# canvas size drawn at), not at the nominal on-canvas rcParams size.
PRINT_FLOOR_PT = 9


def apply_style(base_size=13, axes_labelsize=None, tick_labelsize=None,
                 legend_fontsize=None, axes_linewidth=0.9):
    """Set the shared rcParams. Pass a larger base_size for figures whose
    \\includegraphics width or wide canvas shrinks text harder in print (see
    each figure script's own header comment for its measured scale factor)."""
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": FONT_SERIF_STACK,
        "font.size": base_size,
        "axes.linewidth": axes_linewidth,
        "axes.labelsize": axes_labelsize or base_size,
        "xtick.labelsize": tick_labelsize or (base_size - 1),
        "ytick.labelsize": tick_labelsize or (base_size - 1),
        "legend.fontsize": legend_fontsize or (base_size - 2),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # matplotlib's default mathtext font (DejaVu Sans) does not follow
        # font.serif, so every "$...$" span (numbers, CIs, Greek letters) would
        # otherwise embed a sans-serif face even with font.family set above.
        # STIX was designed to blend with Times-family text, matching the
        # paper body (newtxtext) far better than the "cm" (Computer Modern)
        # default.
        "mathtext.fontset": "stix",
    })
