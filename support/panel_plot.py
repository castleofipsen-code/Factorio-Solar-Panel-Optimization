import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_solar_array(size, solar_coords, acc_coords, subs_coords, robo_coords, pole_coords = None):

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    ax.grid(visible=True)
    ax.set_xticks(np.linspace(0, size, size + 1))
    ax.set_yticks(np.linspace(0, size, size + 1))
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_axisbelow(True)
    ax.set_aspect('equal')
    ax.add_patch(patches.Rectangle((0, 0), size, size,  linewidth = 2, edgecolor = 'k', facecolor = 'none'))


    add_structure(ax, solar_coords, 3, (0.4666, 0.7098, 0.9961))
    add_structure(ax, acc_coords, 2, (0.2, 0.7, 0.3))
    add_structure(ax, subs_coords, 2, (0.6, 0.65, 0.7))
    add_structure(ax, robo_coords, 4, (0.6, 0.1, 0.3) )

    if pole_coords is not None:
        add_structure(ax, pole_coords, 1, (1.0, 0.7, 0.3) )    

    plt.show()

    return fig, ax

def plot_solar_array_periodic(size, solar_coords, acc_coords, subs_coords, robo_coords, pole_coords=None, electric_network=None, plot_electric = False):

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    ax.grid(visible=True)
    ax.set_xticks(np.linspace(0, size, size + 1))
    ax.set_yticks(np.linspace(0, size, size + 1))
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.tick_params(axis='both', which='both', length=0)

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_axisbelow(True)
    ax.set_aspect('equal')

    # Base boundary
    ax.add_patch(
        patches.Rectangle(
            (0, 0), size, size,
            linewidth=2,
            edgecolor='k',
            facecolor='none'
        )
    )

    # Periodic / wrapped structures
    add_structure_periodic(ax, solar_coords, 3, (0.4666, 0.7098, 0.9961), size)
    add_structure_periodic(ax, acc_coords,   2, (0.2, 0.7, 0.3), size)
    add_structure_periodic(ax, subs_coords,  2, (0.6, 0.65, 0.7), size)
    add_structure_periodic(ax, robo_coords,  4, (0.6, 0.1, 0.3), size)

    if plot_electric:
        electric_network_sub = subs_coords - np.array([8,8])
        electric_network_mps = pole_coords - np.array([3,3])
        add_structure_periodic(ax, electric_network_sub, 18, (0.75, 0.75, 0.75) , size, border=False)
        add_structure_periodic(ax, electric_network_mps, 7, (0.75, 0.75, 0.75) , size, border=False)

    if pole_coords is not None:
        add_structure_periodic(ax, pole_coords, 1, (1.0, 0.7, 0.3) , size)

    if electric_network is not None:
        add_structure_periodic(ax, electric_network, 1, (0.8, 0.9, 1.0) , size, border=False)


    return fig, ax


def add_inset_border_band(ax, position, size, inset, width, color):
    """Draw a rectangular border band without filling its interior."""
    x, y = position
    span = size - 2 * inset

    strips = (
        ((x + inset, y + inset), span, width),
        ((x + inset, y + size - inset - width), span, width),
        ((x + inset, y + inset + width), width, span - 2 * width),
        ((x + size - inset - width, y + inset + width), width, span - 2 * width),
    )

    for strip_position, strip_width, strip_height in strips:
        ax.add_patch(
            patches.Rectangle(
                strip_position,
                strip_width,
                strip_height,
                linewidth=0,
                facecolor=(*color, 1.0),
            )
        )


def add_structure(
    ax,
    anchor,
    size,
    color,
    border,
    linewidth=2,
    solid_border=False,
    inside_border_width=0.1,
    fill_alpha=0.5,
    gap=0.0,
):

    anchor = np.asarray(anchor)   # â† fix

    black = (0,0,0)
    edge_color = tuple(0.6*x + 0.4*y for x, y in zip(color, black))
    white = (1, 1, 1)
    inner_edge_color = tuple(
        0.7 * fill + 0.3 * highlight
        for fill, highlight in zip(color, white)
    )
    interior_fill_color = tuple(
        0.25 * fill + 0.75 * inner
        for fill, inner in zip(color, inner_edge_color)
    )

    for i in range(anchor.shape[0]):
        position = (
            anchor[i, 0] + gap / 2,
            anchor[i, 1] + gap / 2,
        )
        drawing_size = size - gap

        if solid_border:
            if border:
                if not 0 < inside_border_width < drawing_size / 4:
                    raise ValueError(
                        "inside_border_width must be positive and the two "
                        "border bands must fit inside the structure"
                    )

                # Draw the translucent fill first so it cannot soften either
                # border during antialiasing.
                fill_inset = 2 * inside_border_width
                fill_position = (
                    position[0] + fill_inset,
                    position[1] + fill_inset,
                )
                fill_size = drawing_size - 2 * fill_inset
                ax.add_patch(
                    patches.Rectangle(
                        fill_position,
                        fill_size,
                        fill_size,
                        linewidth=0,
                        facecolor=(*interior_fill_color, fill_alpha),
                    )
                )

                # These are true border rings: neither is drawn below the fill.
                add_inset_border_band(
                    ax,
                    position,
                    drawing_size,
                    inset=0,
                    width=inside_border_width,
                    color=edge_color,
                )
                add_inset_border_band(
                    ax,
                    position,
                    drawing_size,
                    inset=inside_border_width,
                    width=inside_border_width,
                    color=inner_edge_color,
                )
            else:
                ax.add_patch(
                    patches.Rectangle(
                        position,
                        drawing_size,
                        drawing_size,
                        linewidth=0,
                        facecolor=(*interior_fill_color, fill_alpha),
                    )
                )
        else:
            ax.add_patch(
                patches.Rectangle(
                    position,
                    drawing_size,
                    drawing_size,
                    linewidth=linewidth if border else 0,
                    edgecolor=edge_color,
                    facecolor=color,
                    alpha=0.5,
                )
            )

    return

def add_structure_periodic(ax, coords, structure_size, color, domain_size, border=True):
    for dx in (-domain_size, 0, domain_size):
        for dy in (-domain_size, 0, domain_size):
            shifted_coords = [(x + dx, y + dy) for x, y in coords]
            add_structure(ax, shifted_coords, structure_size, color, border)
