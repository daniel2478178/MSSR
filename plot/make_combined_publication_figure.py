"""Create a publication-style two-panel comparison figure from the source workbooks."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT = BASE_DIR / "combined_publication_figure.png"
PDF_OUTPUT = BASE_DIR / "combined_publication_figure.pdf"
TASK_IDS = [f"P{number:02d}" for number in range(1, 60)]
TEST_COLUMNS = {
    0.00: "noise_0_R2",
    0.01: "noise_001_R2",
    0.03: "noise_003_R2",
    0.05: "noise_005_R2",
    0.10: "noise_01_R2",
}
TEST_NOISE = np.array(list(TEST_COLUMNS))
FAILURE_LIMIT = -1e50

COLORS = {
    "MDSR": "#1F4E79",
    "MDSR with parameters": "#D58A00",
    "MDSR, train noise 0": "#1F4E79",
    "MDSR, train noise 0.01": "#2E8B57",
    "MDSR, train noise 0.03": "#C44E52",
    "LLM-MDSR, train noise 0": "#D58A00",
    "LLM-MDSR, train noise 0.01": "#2E8B57",
    "LLM-MDSR, train noise 0.03": "#C44E52",
    "Perfect equations (clean)": "#7A5195",
}
DISTRIBUTION_COLORS = {
    "MDSR without parameters": "#1F4E79",
    "MDSR with parameters": "#D58A00",
    "LLM-MDSR, train noise 0": "#2E8B57",
}


def read_frame(path, columns, complete=False):
    frame = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    required = {"ID", *columns}
    missing = required - set(frame.columns)
   
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    frame = frame.set_index("ID")
   
    return frame.reindex(TASK_IDS) if complete else frame


def clean_r2(values):
    values = pd.to_numeric(values, errors="coerce")
    return values.where(np.isfinite(values) & values.gt(FAILURE_LIMIT))


def r2_counts(values):
    values = clean_r2(values)
    valid = values.dropna().to_numpy()
    counts = np.histogram(valid, bins=[-np.inf, 0, 0.9, 0.99, 0.9999, 1, np.inf])[0]
    
    return np.r_[values.isna().sum(), counts]


def structure_counts(values):
    values = pd.to_numeric(values, errors="coerce")
    valid = values[np.isfinite(values) & values.ge(0)]
    counts = np.histogram(valid, bins=[0, 45, 75, 90, 100.0000001])[0]
    return np.r_[values.size - valid.size, counts]


def success_rates(frame):
    frame = frame.reindex(TASK_IDS)
    values = frame[list(TEST_COLUMNS.values())].apply(clean_r2)
    return 100 * values.ge(0.9).sum().to_numpy() / len(TASK_IDS)


def perfect_success_rates(frame):
    values = frame[list(TEST_COLUMNS.values())].apply(pd.to_numeric, errors="coerce")
    outliers = values.lt(0).all(axis=1)
    values = values.loc[~outliers]
    return 100 * values.ge(0.9).sum().to_numpy() / len(values)


def robustness_reliability(frame, common_ids):
    values = frame.loc[common_ids, list(TEST_COLUMNS.values())]
    values = values.apply(clean_r2)
    return 100 * values.ge(0.9).sum().to_numpy() / len(TASK_IDS)


def add_bar_labels(ax, bars, counts):
    for bar, count in zip(bars, counts):
        if not count:
            continue
        x = bar.get_x() + bar.get_width() / 2
        if count >= 4:
            ax.text(
                x,
                count / 2,
                str(int(count)),
                ha="center",
                va="center",
                color="white",
                fontsize=7.5,
                fontweight="bold",
            )
        else:
            ax.annotate(
                str(int(count)),
                (x, count),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#222222",
            )


def load_inputs():
    mdsr = read_frame(
        BASE_DIR / "不带参数.xlsx",
        ["shared_fit_r2_0_7", "structure_similarity_score", *TEST_COLUMNS.values()],
        complete=True,
    )
    mdsr_parameters = read_frame(
        BASE_DIR / "带参数统计表格(1).xlsx",
        ["shared_fit_r2_0_7", "structure_similarity_score"],
    )
    llm = {
        noise: read_frame(
            BASE_DIR / "LLM-SR" / filename,
            [*TEST_COLUMNS.values(), "SimilarityScore"],
        )
        for noise, filename in {
            0.00: "merged_statistics_没有噪声.xlsx",
            0.01: "merged_statistics_noise_001.xlsx",
            0.03: "merged_statistics_noise_003.xlsx",
        }.items()
    }
    perfect = read_frame(
        BASE_DIR.parent
        / "equation_verfication"
        / "physicsMDSR_Range_GenerationFormula_noise_metrics.xlsx",
        list(TEST_COLUMNS.values()),
        complete=True,
    )
    robustness = {
        noise: read_frame(
            BASE_DIR / filename,
            list(TEST_COLUMNS.values()),
        )
        for noise, filename in {
            0.00: "不带参数.xlsx",
            0.01: "噪声001统计表(1).xlsx",
            0.03: "噪声003统计表.xlsx",
        }.items()
    }
    return mdsr, mdsr_parameters, llm, perfect, robustness


def plot_distributions(ax, mdsr, mdsr_parameters, llm):
    r2_labels = [
        "Failed",
        r"$R^2 < 0$",
        r"$0 \leq R^2 < 0.9$",
        r"$0.9 \leq R^2 < 0.99$",
        r"$0.99 \leq R^2 < 0.9999$",
        r"$0.9999 \leq R^2 < 1$",
        r"$R^2 = 1$",
    ]
    structure_labels = [
        "Failed\n$S < 0$",
        "Low\n$0$–$44$",
        "Medium\n$45$–$74$",
        "Relatively high\n$75$–$89$",
        "High\n$90$–$100$",
    ]
    labels = r2_labels + structure_labels
    x = np.r_[np.arange(len(r2_labels)), np.arange(len(r2_labels) + 1, len(labels) + 1)]
    counts = {
        "MDSR without parameters": np.r_[
            r2_counts(mdsr["shared_fit_r2_0_7"]),
            structure_counts(mdsr["structure_similarity_score"]),
        ],
        "MDSR with parameters": np.r_[
            r2_counts(mdsr_parameters["shared_fit_r2_0_7"]),
            structure_counts(mdsr_parameters["structure_similarity_score"]),
        ],
        "LLM-MDSR, train noise 0": np.r_[
            r2_counts(llm[0.00]["noise_0_R2"]),
            structure_counts(llm[0.00]["SimilarityScore"]),
        ],
    }
    offsets = [-0.27, 0, 0.27]
    width = 0.25
    for offset, (label, values) in zip(offsets, counts.items()):
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=DISTRIBUTION_COLORS[label],
            edgecolor="white",
            linewidth=0.35,
        )
        add_bar_labels(ax, bars, values)

    ax.axvline(6.5, color="#888888", linewidth=0.8, linestyle="--")
    ax.text(3, 1.02, "Shared-fit $R^2$ distribution", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(10, 1.02, "Structural similarity distribution", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Number of tasks")
    ax.set_title("Distribution comparison", pad=28)
    ax.set_ylim(0, 42)
    ax.set_xlim(-0.75, x[-1] + 0.75)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def plot_noise_comparison(ax, mdsr, llm, perfect, robustness):
    success_series = [
        ("MDSR", success_rates(mdsr), "s", "--"),
        ("LLM-MDSR, train noise 0", success_rates(llm[0.00]), "^", "-"),
        ("LLM-MDSR, train noise 0.01", success_rates(llm[0.01]), "^", (0, (6, 2))),
        ("LLM-MDSR, train noise 0.03", success_rates(llm[0.03]), "^", ":"),
        ("Perfect equations (clean)", perfect_success_rates(perfect), "*", "--"),
    ]
    for label, values, marker, linestyle in success_series:
        ax.plot(
            TEST_NOISE,
            values,
            color=COLORS[label],
            marker=marker,
            linestyle=linestyle,
            linewidth=2.2,
            markersize=6,
            markerfacecolor="white" if label == "Perfect equations (clean)" else COLORS[label],
            markeredgewidth=1.5 if label == "Perfect equations (clean)" else 0.8,
            label=label,
            zorder=3,
        )

    common_ids = sorted(
        set.intersection(*(set(frame.dropna(how="all").index) for frame in robustness.values())),
        key=lambda value: int(value[1:]),
    )
    reliability_series = [
        ("MDSR, train noise 0", robustness_reliability(robustness[0.00], common_ids), COLORS["MDSR, train noise 0"], "s"),
        ("MDSR, train noise 0.01", robustness_reliability(robustness[0.01], common_ids), COLORS["MDSR, train noise 0.01"], "s"),
        ("MDSR, train noise 0.03", robustness_reliability(robustness[0.03], common_ids), COLORS["MDSR, train noise 0.03"], "s"),
    ]
    perfect_values = perfect[list(TEST_COLUMNS.values())].apply(pd.to_numeric, errors="coerce")
    perfect_outliers = perfect_values.lt(0).all(axis=1)
    perfect_clean = perfect_values.loc[~perfect_outliers]
    reliability_series.append(
        ("Perfect equations (clean)", 100 * perfect_clean.ge(0.9).mean().to_numpy(), COLORS["Perfect equations (clean)"], "*")
    )
    for label, values, color, marker in reliability_series:
        ax.plot(
            TEST_NOISE,
            values,
            color=color,
            marker=marker,
            linestyle=(0, (3, 2)),
            linewidth=1.5,
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.1,
            alpha=0.8,
            zorder=2,
        )

    ax.set_title("(A) Success rate and high-accuracy reliability")
    ax.set_xlabel("Testing noise")
    ax.set_ylabel(r"Success rate / fraction with $R^2 \geq 0.9$ (%)")
    ax.set_xticks(TEST_NOISE, ["0", "0.01", "0.03", "0.05", "0.10"])
    ax.set_ylim(55, 105)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    legend_handles = [
        Line2D([0], [0], color=COLORS[label], marker=marker, linestyle=linestyle,
               linewidth=2, markersize=5, markerfacecolor="white" if label == "Perfect equations (clean)" else COLORS[label],
               label=label)
        for label, _, marker, linestyle in success_series
    ]
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linestyle=(0, (3, 2)),
            linewidth=1.5,
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.1,
            label=label,
        )
        for label, _, color, marker in reliability_series[:-1]
    )
    ax.legend(handles=legend_handles, loc="lower left", ncol=2, fontsize=7.5, frameon=True)


def plot_typical_performance(ax, perfect, robustness):
    common_ids = sorted(
        set.intersection(*(set(frame.dropna(how="all").index) for frame in robustness.values())),
        key=lambda value: int(value[1:]),
    )
    series = [
        (0.00, "MDSR, train noise 0", COLORS["MDSR, train noise 0"]),
        (0.01, "MDSR, train noise 0.01", COLORS["MDSR, train noise 0.01"]),
        (0.03, "MDSR, train noise 0.03", COLORS["MDSR, train noise 0.03"]),
    ]
    for noise, label, color in series:
        values = robustness[noise].loc[common_ids, list(TEST_COLUMNS.values())]
        values = values.apply(clean_r2)
        quantiles = values.quantile([0.25, 0.50, 0.75])
        ax.plot(
            TEST_NOISE,
            quantiles.loc[0.50].to_numpy(),
            color=color,
            marker="s",
            linewidth=2,
            markersize=5,
            label=label,
        )
        ax.fill_between(
            TEST_NOISE,
            quantiles.loc[0.25].to_numpy(),
            quantiles.loc[0.75].to_numpy(),
            color=color,
            alpha=0.14,
        )

    perfect_values = perfect[list(TEST_COLUMNS.values())].apply(
        pd.to_numeric, errors="coerce"
    )
    perfect_outliers = perfect_values.lt(0).all(axis=1)
    perfect_clean = perfect_values.loc[~perfect_outliers]
    perfect_quantiles = perfect_clean.quantile([0.25, 0.50, 0.75])
    ax.plot(
        TEST_NOISE,
        perfect_quantiles.loc[0.50].to_numpy(),
        color=COLORS["Perfect equations (clean)"],
        marker="*",
        markerfacecolor="white",
        markeredgewidth=1.2,
        linewidth=2,
        markersize=8,
        linestyle="--",
        label="Perfect equations (clean)",
    )
    ax.fill_between(
        TEST_NOISE,
        perfect_quantiles.loc[0.25].to_numpy(),
        perfect_quantiles.loc[0.75].to_numpy(),
        color=COLORS["Perfect equations (clean)"],
        alpha=0.10,
    )
    ax.set_title("(B) Typical performance")
    ax.set_xlabel("Testing noise")
    ax.set_ylabel(r"$R^2$ (median; band = IQR)")
    ax.set_xticks(TEST_NOISE, ["0", "0.01", "0.03", "0.05", "0.10"])
    ax.set_ylim(0.82, 1.01)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(loc="lower left", ncol=2, fontsize=7.5, frameon=True)


def main():
    mdsr, mdsr_parameters, llm, perfect, robustness = load_inputs()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )
    fig = plt.figure(figsize=(14.0, 10.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.45, wspace=0.24)
    distribution_ax = fig.add_subplot(grid[0, :])
    typical_ax = fig.add_subplot(grid[1, 0])
    comparison_ax = fig.add_subplot(grid[1, 1])
    plot_distributions(distribution_ax, mdsr, mdsr_parameters, llm)
    plot_typical_performance(typical_ax, perfect, robustness)
    plot_noise_comparison(comparison_ax, mdsr, llm, perfect, robustness)
    distribution_ax.legend(loc="upper left", ncol=3, frameon=True, fontsize=8.5)
    for ax in (distribution_ax, typical_ax, comparison_ax):
        ax.set_axisbelow(True)
        ax.tick_params(which="both", top=True, right=True)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.07, top=0.96, hspace=0.45, wspace=0.24)
    fig.savefig(OUTPUT, dpi=800, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_OUTPUT, format="pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved {OUTPUT}")
    print(f"Saved {PDF_OUTPUT}")


if __name__ == "__main__":
    main()
