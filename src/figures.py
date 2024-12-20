"""Concrete functions to draw plots and figures."""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from . import numerics, paths_and_constants, vis

Colors = paths_and_constants.Colors


def draw_graph_legend(G, name, node_name, edge_name, fontsize=22, linespacing=1.2):
    fig = plt.gcf()
    ax = plt.gca()
    bbox = ax.get_position()
    upper_left_x = bbox.x1
    upper_left_y = bbox.y1

    fig.text(
        bbox.x0,
        bbox.y1,
        name,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=fontsize,
        transform=fig.transFigure,
    )

    text1 = plt.text(
        upper_left_x,
        upper_left_y,
        f"{len(G.nodes):3d} nodes ({node_name})",
        fontsize=fontsize,
        horizontalalignment="right",
        verticalalignment="top",
        transform=fig.transFigure,
    )

    # Get the height of a single line of text in figure coordinates
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_text = text1.get_window_extent(
        renderer=renderer
    )  # use text1 instead of creating a new one
    text_height_pixels = bbox_text.height
    text_height_figure = text_height_pixels / fig.dpi / fig.get_size_inches()[1]
    second_line_y = upper_left_y - text_height_figure * linespacing

    # Second text
    ax.text(
        upper_left_x,
        second_line_y,
        f"{len(G.edges):3d} edges ({edge_name})",
        horizontalalignment="right",
        verticalalignment="top",
        transform=fig.transFigure,
        fontsize=fontsize,
    )


def draw_tensor(
    values,
    col,
    labels=None,
    cmaps=None,
    rsize=0.5,
    rotate_text=True,
    draw_text=True,
    shape_title=True,
):
    assert values.ndim == 2, "Olny 2D tensors!"
    n, k = values.shape
    if labels is None:
        labels = [str(i) for i in range(k)]
    if cmaps is None:
        cmaps = [vis.light_color_cmap(col)] * k

    width, height = n * rsize, k * rsize
    fig, ax = plt.subplots(figsize=(width, height))
    for i in range(n):
        for j in range(k):
            pos = i * rsize, height - (j + 1) * rsize
            color = cmaps[j](values[i, j])
            rect = plt.Rectangle(pos, rsize, rsize, facecolor=color)
            ax.add_patch(rect)

    # Set axis limits and labels
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.yaxis.tick_left()  # Move x-ticks to the top
    ax.set_xticks([])
    ax.set_xticklabels([])
    if draw_text:
        ax.set_yticks(np.arange(rsize / 2, height, rsize))
        ax.set_yticklabels(labels[::-1], fontsize=18)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
    if shape_title:
        plt.title(
            f"{values.shape[0]}x{values.shape[1]}",
            fontdict={"color": col, "fontsize": 18},
        )

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor(col)  # Or any color you prefer
    return


def draw_embedding(values, cmap, border: bool):
    values = values.flatten().reshape(-1, 1)
    assert np.max(values) <= 1, "Expecting embeddings in range [0, 1]"
    assert np.min(values) >= 0, "Expecting embeddings in range [0, 1]"
    plt.imshow(values, cmap=cmap)
    plt.xticks([], [])
    plt.yticks([], [])
    if border:
        for axis in ["top", "bottom", "left", "right"]:
            plt.gca().spines[axis].set_linewidth(3)
            plt.gca().spines[axis].set_color("k")
    plt.tight_layout()


def draw_vector(values, cmap):
    values = values.flatten()
    plt.figure(figsize=(8, 2))
    k = len(values)
    index = np.arange(k)
    colors = [cmap(v) for v in values]
    ax = plt.gca()
    rect = patches.Rectangle(
        (-0.5, 0.0),
        k,
        1.0,
        facecolor=Colors.background,
        edgecolor=Colors.background_edge,
    )
    ax.add_patch(rect)
    plt.bar(index, values, width=0.95, linewidth=1, edgecolor=cmap(1.0), color=colors)
    plt.axis("off")
    plt.ylim([0.0, 1])


def build_graph_parts(g, spec_list, main_color, extra_colors, extract_fn):
    tensors = {}
    names = []
    cmaps = []
    n_colors = sum(feat.dim if feat.is_discrete() else 1 for feat in spec_list)
    pal = extra_colors[:n_colors]
    cmaps = [vis.light_color_cmap(c) for c in pal]
    for i, feat in enumerate(spec_list):
        name = feat.name
        value_dict = extract_fn(g, name)
        first = i == 0
        if not value_dict:
            raise ValueError(f"No attribute {name} in graph")
        if feat.data_type == "categorical":
            id_map = {v: i for i, v in enumerate(feat.values)}
            ids = [id_map[v] for v in value_dict.values()]
            tensor = numerics.one_hot_encode(ids, len(feat.values))
            tensors[name] = tensor
            names.extend(feat.values)
            if first:
                colors = [cmaps[i](1.0) for i in np.argmax(tensor, axis=1)]
        elif feat.data_type == "continuous":
            if feat.dim == 1:
                tensor = numerics.cast_as_2d(list(value_dict.values()))
            else:
                tensor = np.hstack(list(value_dict.values()))
            tensors[name] = numerics.normalize_values(
                tensor, feat.values[0], feat.values[1]
            )
            names.append(name)
            if first:
                colors = [cmaps[0](vi) for vi in tensor[:, 0]]
        else:
            raise ValueError(f"Unknown data type {feat.data_type}")

    full_tensor = np.hstack(list(tensors.values()))
    return full_tensor, names, colors, cmaps
