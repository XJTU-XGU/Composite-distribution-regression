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
    "legend.fontsize": 16,
    "figure.titlesize": 18,
    "figure.titleweight": "normal",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


DEFAULT_INPUT = "results_null_scalar/fig3_scalar_irrelevant_assume_d10_results.mat"
DEFAULT_FIGURE = "outputs/Figure4.pdf"

METHODS = (
    ("structured_transport", "Our method", "blue"),
    ("GP2022_condfree_OT", "Ghodrati & Panaretos (2022)", "darkorange"),
)


def load_torch_saved_dict(path):
    """Read torch.save outputs containing ordinary Python dict/list/float data."""
    path = Path(path)
    with zipfile.ZipFile(str(path), "r") as archive:
        data_names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if not data_names:
            raise ValueError("No data.pkl entry found in {}".format(path))
        return pickle.loads(archive.read(data_names[0]))


def load_results(input_path):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    results = load_torch_saved_dict(input_path)
    if not isinstance(results, dict):
        raise TypeError("Expected top-level dict, got {}".format(type(results).__name__))
    return results


def resolve_setting(results, setting_key=None):
    if setting_key is not None:
        if setting_key not in results:
            raise KeyError("Setting key {} not found. Available keys: {}".format(
                setting_key, list(results.keys())
            ))
        return setting_key
    if len(results) != 1:
        raise ValueError("Please pass --setting-key. Available keys: {}".format(list(results.keys())))
    return next(iter(results.keys()))


def make_boxplot(ax, data_list, positions, width, color):
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
            markersize=3.8,
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


def method_w1(record):
    w1 = np.asarray(record.get("w1", []), dtype=float)
    return w1[np.isfinite(w1)]


def draw_w1_boxplot(ax, setting_results):
    positions = np.arange(len(METHODS), dtype=float)
    width = 0.32

    for idx, (method_key, _, color) in enumerate(METHODS):
        record = setting_results.get(method_key)
        if not record:
            continue
        w1 = method_w1(record)
        if w1.size == 0:
            continue
        make_boxplot(ax, [w1], [positions[idx]], width, color)

    ax.set_xticks(positions)
    ax.set_xticklabels(["" for _ in METHODS])
    ax.set_ylabel(r"$d_{\mathcal{W}}(\hat{\nu},\nu)$", fontproperties=regular_font(18))
    ax.set_title("Wasserstein distance", fontproperties=regular_font(18), pad=8)
    ax.grid(False)
    ax.set_ylim([0.08,0.35])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(regular_font(18))


def draw_time_barplot(ax, setting_results):
    positions = np.arange(len(METHODS), dtype=float)
    values = []
    colors = []
    for method_key, _, color in METHODS:
        record = setting_results.get(method_key, {})
        values.append(float(record.get("time", 0.0)))
        colors.append(color)

    ax.bar(
        positions,
        values,
        width=0.3,
        color="none",
        edgecolor=colors,
        linewidth=1.8,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(["" for _ in METHODS])
    ax.set_ylabel("Time (seconds)", fontproperties=regular_font(18))
    ax.set_title("Computational time", fontproperties=regular_font(18), pad=8)
    ax.grid(False)
    ax.set_xlim([-0.5,1.5])
    ax.set_ylim([0,15])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(regular_font(18))


def print_summary(setting_key, setting_results):
    print("Summary for {}:".format(setting_key))
    for method_key, method_label, _ in METHODS:
        record = setting_results.get(method_key, {})
        w1 = method_w1(record)
        if w1.size == 0:
            print("  {}: no data".format(method_label))
            continue
        sd = float(np.std(w1, ddof=1)) if w1.size > 1 else 0.0
        print(
            "  {}: w1_len={}, mean={:.6g}, sd={:.6g}, time={:.6g}s".format(
                method_label,
                int(w1.size),
                float(np.mean(w1)),
                sd,
                float(record.get("time", 0.0)),
            )
        )


def plot_summary(results, setting_key, save_path=DEFAULT_FIGURE, show=False):
    setting_results = results[setting_key]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.55))

    draw_w1_boxplot(axes[0], setting_results)
    draw_time_barplot(axes[1], setting_results)

    legend_handles = [
        Line2D([0], [0], color=color, lw=2.2, label=label)
        for _, label, color in METHODS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=True,
        prop=regular_font(18),
        handlelength=1.2,
        columnspacing=1.0,
        handletextpad=0.35,
        bbox_to_anchor=(0.5, 0.96),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.82])
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot scalar-irrelevant misspecification experiment results."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--setting-key", default=None)
    parser.add_argument("--figure", default=DEFAULT_FIGURE)
    parser.add_argument("--show", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    results = load_results(args.input)
    setting_key = resolve_setting(results, args.setting_key)
    plot_summary(results, setting_key, save_path=args.figure, show=args.show)
    print("Saved figure: {}".format(Path(args.figure).resolve()))
    print()
    print_summary(setting_key, results[setting_key])


if __name__ == "__main__":
    main()
