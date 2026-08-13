import argparse
import pickle
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D


REGULAR_FONT_PATH = Path(r"C:\Windows\Fonts\times.ttf")


def regular_font(size):
    if REGULAR_FONT_PATH.exists():
        return FontProperties(fname=str(REGULAR_FONT_PATH), size=size, weight="normal")
    return FontProperties(family="Times New Roman", size=size, weight="normal")


mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.weight": "normal",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 18,
    "figure.titleweight": "normal",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


DEFAULT_P_GRID_INPUTS = (
    "results/fig3_vector_decay_theta_no_intercept_results.mat",
    "results/fig3_vector_decay_theta_no_intercept_results_50.mat",
    # "results/fig3_vector_decay_theta_no_intercept_results_50_.mat",
)
DEFAULT_N_GRID_INPUT = "results/fig3_vector_decay_theta_no_intercept_N_grid_p11_results.mat"
DEFAULT_FIGURE = "outputs/Figure3.pdf"

P_SPECS = (
    (1, ("p_2",)),
    (10, ("p_11",)),
    (50, ("p_50", "p_51")),
)

N_SPECS = (
    (25, ("N_25",)),
    (50, ("N_50",)),
    (100, ("N_100",)),
)

METHODS = (
    ("structured_transport", "Our method", "blue"),
    ("GOT_noisy_scalar", "Zhu & Müller (2026)", "purple"),
    ("TW2025_PGFR", "Tucker & Wu (2025)", "green"),
)

N_GRID_METHODS = (
    "structured_transport",
    "TW2025_PGFR",
)


def load_torch_saved_dict(path):
    """Read torch.save outputs containing ordinary Python dict/list/float data."""
    path = Path(path)
    with zipfile.ZipFile(str(path), "r") as archive:
        data_names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if not data_names:
            raise ValueError("No data.pkl entry found in {}".format(path))
        return pickle.loads(archive.read(data_names[0]))


def load_merged_results(input_paths):
    merged = {}
    for input_path in input_paths:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(str(input_path))
        for group_key, method_results in load_torch_saved_dict(input_path).items():
            merged.setdefault(group_key, {}).update(method_results)
    return merged


def load_results(input_path):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    results = load_torch_saved_dict(input_path)
    if not isinstance(results, dict):
        raise TypeError("Expected top-level dict, got {}".format(type(results).__name__))
    return results


def resolve_key(results, keys):
    for key in keys:
        if key in results:
            return key
    return None


def make_boxplot(ax, data_list, positions, width, color):
    if not data_list:
        return None
    bp = ax.boxplot(
        data_list,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor="none", edgecolor=color, linewidth=1.6),
        whiskerprops=dict(color=color, linewidth=1.4),
        capprops=dict(color=color, linewidth=1.4),
        medianprops=dict(color=color, linewidth=1.8),
        flierprops=dict(
            marker="o",
            markersize=3.5,
            markerfacecolor=color,
            markeredgecolor=color,
            alpha=0.55,
        ),
    )
    for box in bp["boxes"]:
        box.set_facecolor("none")
        box.set_edgecolor(color)
        box.set_linewidth(1.6)
    for artist_name in ("whiskers", "caps", "medians"):
        for artist in bp[artist_name]:
            artist.set_color(color)
            artist.set_linewidth(1.8 if artist_name == "medians" else 1.4)
    for flier in bp["fliers"]:
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)
        flier.set_alpha(0.55)
    return bp


def collect_box_data(results, specs, method_key, allowed_methods=None):
    if allowed_methods is not None and method_key not in allowed_methods:
        return [], []

    data_list = []
    positions = []
    active_methods = [item for item in METHODS if allowed_methods is None or item[0] in allowed_methods]
    method_offsets = np.linspace(-0.27, 0.27, len(active_methods))
    method_index = [item[0] for item in active_methods].index(method_key)
    x = np.arange(len(specs), dtype=float)

    for group_index, (_, keys) in enumerate(specs):
        group_key = resolve_key(results, keys)
        if group_key is None:
            continue
        record = results.get(group_key, {}).get(method_key)
        if not record:
            continue
        w1 = np.asarray(record.get("w1", []), dtype=float)
        w1 = w1[np.isfinite(w1)]
        if w1.size == 0:
            continue
        data_list.append(w1)
        positions.append(float(x[group_index] + method_offsets[method_index]))

    return data_list, positions


def style_axis(ax, xtick_labels, xlabel):
    ax.set_xticklabels(xtick_labels, fontproperties=regular_font(18))
    ax.set_xlabel(xlabel, fontproperties=regular_font(18))
    ax.set_ylabel(r"$d_{\mathcal{W}}(\hat{\nu},\nu)$", fontproperties=regular_font(18))
    ax.set_ylim(0.08, 0.40)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(regular_font(18))
    ax.grid(False)


def draw_panel(ax, results, specs, allowed_methods, xtick_labels, xlabel, title):
    x = np.arange(len(specs), dtype=float)
    active_methods = [item for item in METHODS if item[0] in allowed_methods]
    offsets = np.linspace(-0.27, 0.27, len(active_methods))
    width = 0.15 if len(active_methods) == 4 else 0.17

    group_halfspan = float(np.max(np.abs(offsets)) + width / 2.0 + 0.08)
    for xi in x:
        ax.axvspan(xi - group_halfspan, xi + group_halfspan, color="gray", alpha=0.06, zorder=0)

    for method_key, _, color in active_methods:
        data_list, positions = collect_box_data(
            results,
            specs,
            method_key,
            allowed_methods=allowed_methods,
        )
        make_boxplot(ax, data_list, positions, width, color)

    ax.set_xticks(x)
    ax.set_xlim(-0.55, len(specs) - 0.45)
    ax.set_title(title, fontproperties=regular_font(18), pad=8)
    style_axis(ax, xtick_labels=xtick_labels, xlabel=xlabel)


def plot_combined_boxplot(p_results, n_results, save_path=DEFAULT_FIGURE, show=False):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), sharey=False)

    draw_panel(
        axes[0],
        p_results,
        P_SPECS,
        allowed_methods=tuple(method_key for method_key, _, _ in METHODS),
        xtick_labels=[r"$d={}$".format(d_label) for d_label, _ in P_SPECS],
        xlabel="",
        title=r"Varying $d$ ($N=100$)",
    )
    draw_panel(
        axes[1],
        n_results,
        N_SPECS,
        allowed_methods=N_GRID_METHODS,
        xtick_labels=[r"$N={}$".format(N_label) for N_label, _ in N_SPECS],
        xlabel="",
        title=r"Varying $N$ ($d=10$)",
    )
    legend_handles = [
        Line2D([0], [0], color=color, lw=2.2, label=label)
        for _, label, color in METHODS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=True,
        prop=regular_font(18),
        handlelength=1.1,
        columnspacing=0.9,
        handletextpad=0.35,
        bbox_to_anchor=(0.5, 1.10),
    )

    plt.tight_layout(rect=[0, 0, 1, 1.0])
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def get_time(results, specs, method_key, group_label):
    for label, keys in specs:
        if label != group_label:
            continue
        group_key = resolve_key(results, keys)
        if group_key is None:
            return ""
        record = results.get(group_key, {}).get(method_key)
        if record and "time" in record:
            return float(record["time"])
    return ""


def format_time(value):
    if value == "":
        return "--"
    return "{:.6g}".format(float(value))


def print_time_table(p_results, n_results):
    columns = [
        "Method",
        "d=1",
        "d=10",
        "d=50",
        "N=25",
        "N=50",
        "N=100",
    ]
    rows = []
    for method_key, method_label, _ in METHODS:
        row = [method_label]
        for d_label, _ in P_SPECS:
            row.append(format_time(get_time(p_results, P_SPECS, method_key, d_label)))
        for N_label, _ in N_SPECS:
            if method_key not in N_GRID_METHODS:
                row.append("--")
            else:
                row.append(format_time(get_time(n_results, N_SPECS, method_key, N_label)))
        rows.append(row)

    widths = [
        max(len(str(row[i])) for row in ([columns] + rows))
        for i in range(len(columns))
    ]

    print("Total computation time in seconds")
    print("Varying d columns use N=100; varying N columns use d=10.")
    print("Zhu & Müller (2026) is omitted from the N-grid experiment due to computational cost.")
    print()

    header = "  ".join(str(columns[i]).ljust(widths[i]) for i in range(len(columns)))
    divider = "  ".join("-" * widths[i] for i in range(len(columns)))
    print(header)
    print(divider)
    for row in rows:
        print("  ".join(str(row[i]).ljust(widths[i]) for i in range(len(columns))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine p-grid and N-grid fig3 vector no-intercept W1 boxplots."
    )
    parser.add_argument("--p-grid-inputs", nargs="+", default=list(DEFAULT_P_GRID_INPUTS))
    parser.add_argument("--n-grid-input", default=DEFAULT_N_GRID_INPUT)
    parser.add_argument("--figure", default=DEFAULT_FIGURE)
    parser.add_argument("--show", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    p_results = load_merged_results(args.p_grid_inputs)
    n_results = load_results(args.n_grid_input)

    plot_combined_boxplot(p_results, n_results, save_path=args.figure, show=args.show)
    print("Saved figure: {}".format(Path(args.figure).resolve()))
    print()
    print_time_table(p_results, n_results)


if __name__ == "__main__":
    main()
