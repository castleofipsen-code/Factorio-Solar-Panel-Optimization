import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

transparent_figure = False
show_network_score = True #False to show empty tiles
permanent_robo = True

grid = 100


if grid == 50:
    tile_area = 2500
    max_area = tile_area
    if permanent_robo:
        max_area -= 16
    ratio = 0.84672

    solar_max = np.ceil((max_area)/(9 + 4*0.84672))
    substation_min, substation_max = 5, 10
    solar_min = solar_max - 5
elif grid == 100:
    tile_area = 10000
    max_area = tile_area
    if permanent_robo:
        max_area -= 32
    ratio = 0.84672

    solar_max = np.ceil((max_area)/(9 + 4*0.84672)) - 5
    substation_min, substation_max = 22, 32
    solar_min = solar_max - 10
else:
    raise ValueError(f"grid must be either 50 or 100; got {grid}")

n_solar = np.arange(solar_min, solar_max + 1)
n_substation = np.arange(substation_min, substation_max + 1)


n_accumulators = np.zeros((len(n_solar), len(n_substation)))
free_area = np.zeros((len(n_solar), len(n_substation)))
max_power =  np.zeros((len(n_solar), len(n_substation)))

for i in range(len(n_solar)):
    for j in range(len(n_substation)):

        acc_best = np.ceil(n_solar[i]*0.84672)
        n_accumulators[i, j] = acc_best
        while True:
            total_area =  n_accumulators[i, j]*4 + n_solar[i]*9 + n_substation[j]*4
            if total_area <= max_area:
                if show_network_score:
                    free_area[i, j] = tile_area/(max_area - total_area + n_substation[j]*4)
                else:
                    free_area[i, j] = max_area - total_area
                break
            n_accumulators[i, j] -= 1

        max_power[i, j] = min(42.0*n_solar[i], 5000/100.8*n_accumulators[i,j])

print(n_solar)

fig, ax = plt.subplots()
figure_scale = 1.5 if grid == 100 else 1.0
fig.set_size_inches(11.52 * figure_scale, 8.64 * figure_scale)
fig.subplots_adjust(right=0.76)
outside_color = "white" if transparent_figure else "black"
background_alpha = 0 if transparent_figure else 1
fig.patch.set_alpha(background_alpha)
ax.patch.set_alpha(background_alpha)

im = ax.imshow(
    max_power.T,
    origin="lower",
    aspect="auto",
    cmap="winter",
    extent=[
        n_solar[0] - 0.5,
        n_solar[-1] + 0.5,
        n_substation[0] - 0.5,
        n_substation[-1] + 0.5,
    ]
)
ax.set_box_aspect(1)

ax.set_xticks(n_solar)
ax.set_yticks(n_substation)
ax.tick_params(axis="both", labelsize=15, colors=outside_color)
plt.setp(ax.get_xticklabels(), fontfamily="DejaVu Sans")
plt.setp(ax.get_yticklabels(), fontfamily="DejaVu Sans")
for spine in ax.spines.values():
    spine.set_color(outside_color)

# Grid lines between cells
ax.set_xticks(n_solar - 0.5, minor=True)
ax.set_yticks(n_substation - 0.5, minor=True)

ax.grid(which="minor", color="black", linewidth=1)

ax.set_xlabel("Solar panels", fontsize=16, color=outside_color)
ax.set_ylabel("Substations", fontsize=16, color=outside_color)

label_colors = {
    "n_accumulators": (0.00, 0.00, 0.00),
    "free_area": (0.00, 0.00, 0.00),
    "max_power": (0.00, 0.00, 0.00),
}

def draw_label(text, x, y, y_offset, color):
    ax.annotate(
        text,
        (x, y),
        xytext=(0, y_offset),
        textcoords="offset points",
        ha="center",
        va="center",
        color=color,
        fontsize=12,
        fontfamily="DejaVu Sans",
        fontweight="normal",
        zorder=3
    )

for i, solar in enumerate(n_solar):
    for j, substation in enumerate(n_substation):

        cell_labels = [
            (f"{int(n_accumulators[i, j])}", label_colors["n_accumulators"], 14),
            (f"{max_power[i, j]:.0f}", label_colors["max_power"], 0),
            (f"{free_area[i, j]:.1f}", label_colors["free_area"], -14),
        ]

        for text, color, y_offset in cell_labels:
            draw_label(text, solar, substation, y_offset, color)

cbar = fig.colorbar(im, ax=ax)
cbar.ax.patch.set_alpha(0)
cbar.outline.set_edgecolor(outside_color)
cbar.set_label("Max. Continuous Power", fontsize=16, color=outside_color)
cbar.ax.tick_params(labelsize=15, colors=outside_color)
plt.setp(cbar.ax.get_yticklabels(), fontfamily="DejaVu Sans")
for spine in cbar.ax.spines.values():
    spine.set_color(outside_color)

legend_handles = [
    Line2D([0], [0], marker="^", linestyle="None",
           markersize=12,
           markerfacecolor=outside_color, markeredgecolor=outside_color,
           markeredgewidth=1.2,
           label=r"top. $N_a$"),
    Line2D([0], [0], marker="o", linestyle="None",
           markersize=12,
           markerfacecolor=outside_color, markeredgecolor=outside_color,
           markeredgewidth=1.2,
           label=r"mid. $[P]$"),
    Line2D([0], [0], marker="v", linestyle="None",
           markersize=12,
           markerfacecolor=outside_color, markeredgecolor=outside_color,
           markeredgewidth=1.2,
           label=r"bot. $\mathrm{max}(N_m)$"),
]

legend = ax.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(0.70, 0.88),
    bbox_transform=fig.transFigure,
    frameon=False,
    fontsize=16
)
for text in legend.get_texts():
    text.set_color(outside_color)

output_dir = Path("solved")
output_dir.mkdir(exist_ok=True)
background = "transparent" if transparent_figure else "opaque"
output_path = output_dir / f"solar_vs_substation_rp_{background}_1200dpi.png"
fig.savefig(
    output_path,
    dpi=1200,
    transparent=transparent_figure,
    facecolor=fig.get_facecolor(),
)
print(f"Saved figure to {output_path.resolve()}")

plt.show()
