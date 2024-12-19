"""Visualization utilities."""

import colorsys
from pathlib import Path

import IPython.display as ipy_display
import matplotlib as mpl
import matplotlib.colors
import matplotlib.font_manager
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import HTML, SVG, display

_FONT_NAME = "FiraCode Nerd Font"
_FONT_URL = "https://github.com/chemcognition-lab/graphs_primer/blob/main/data/FiraCode-Regular.ttf?raw=true"


def set_visualization_style():
    if not mpl.font_manager.fontManager.findfont(_FONT_NAME):
        try:
            import pyfonts

            font = pyfonts.load_font(font_url=_FONT_URL)
            mpl.font_manager.fontManager.addfont(mpl.font_manager.findfont(font))
            mpl.rcParams["font.sans-serif"] = [font.get_name()]
            print(f"Loaded {_FONT_NAME}.")
        except ValueError:
            print(f"Could not load {_FONT_NAME}.")

    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.pad_inches"] = 0.1
    mpl.rcParams["savefig.transparent"] = True
    mpl.rcParams["axes.linewidth"] = 2.5
    mpl.rcParams["legend.markerscale"] = 1.0
    mpl.rcParams["legend.fontsize"] = "small"
    # seaborn color palette
    sns.set_palette("colorblind")
    sns.set_style("whitegrid", {"axes.grid": False})
    np.set_printoptions(precision=3)


def save_figure(name, adir="."):
    fig = plt.gcf()
    for ext in ["svg", "png"]:
        path = Path(adir) / f"{name}.{ext}"
        fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0)


def header(text, n=3):
    display(HTML(f"<h{n}>{text}</h{n}>"))


def display_svg(svg_text):
    ipy_display.display(SVG(svg_text))


def shift_color(c, mult=0.5):
    """Shift luminosity of a color, mult < 1 is lighter, mult > 1 is darker.

    Input can be matplotlib color string, hex string, or RGB tuple.
    """
    h, light, s = colorsys.rgb_to_hls(*mpl.colors.to_rgb(c))
    new_light = float(np.clip(1 - mult * (1 - light), 0, 1.0))
    return colorsys.hls_to_rgb(h, new_light, s)


def lighten_color(c, ammount):
    return shift_color(c, np.clip(ammount, 0, 1.0))


def darken_color(c, ammount):
    return shift_color(c, 1 + ammount)


def plot_color_swatches(colors, labels=None, edge: bool = False):
    """Plots a series of color swatches."""
    n = len(colors)
    fig, ax = plt.subplots(figsize=(n, 1))  # Adjust figure size as needed
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Remove margins

    for i, color in enumerate(colors):
        edge_kwargs = {"linewidth": 1, "edgecolor": "black"} if edge else {}
        rect = patches.Rectangle((i / n, 0), 1 / n, 1, facecolor=color, **edge_kwargs)
        ax.add_patch(rect)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if labels is not None:
        delta = 1 / (n * 2)
        ax.set_xticks(np.arange(n) / n + delta)
        ax.set_xticklabels(labels)
    else:
        ax.set_xticks([])
    ax.set_yticks([])


def light_color_cmap(color, name="test"):
    start_color = (1, 1, 1)
    end_color = color
    colors = [start_color, end_color]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(name, colors, N=256)
    return cmap


def rgb2hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        int(255 * rgb[0]), int(255 * rgb[1]), int(255 * rgb[2])
    )


def print_colors(col_list):
    for i, c in enumerate(col_list):
        hex_str = rgb2hex(c)
        print(f"{i:2d} - ({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) - {hex_str}")


def generate_similar_colors(base_color, n, deviation_factor=0.4, hue_variation=0.02):
    """Generates a list of colors similar to a base color."""
    h, l, s = colorsys.rgb_to_hls(*base_color)
    colors = []
    max_deviation = min(n * 0.05, deviation_factor)
    deviations = np.linspace(-max_deviation, max_deviation, n)
    for i in range(n):
        new_h = h + np.random.uniform(-hue_variation, hue_variation)
        new_h = new_h % 1
        new_l = max(0, min(1.0, l + deviations[i]))
        new_s = max(0, min(1, s + deviations[i]))
        new_rgb = colorsys.hls_to_rgb(new_h, new_l, new_s)
        colors.append(new_rgb)
    return colors
