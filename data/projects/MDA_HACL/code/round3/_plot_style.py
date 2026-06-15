"""Talk-ready matplotlib defaults for the BNN2 round-3 pipeline."""
import matplotlib as mpl


def apply_talk_style() -> None:
    """Apply bold, talk-ready matplotlib rcParams.

    Idempotent — touches only rcParams. Per-call kwargs that override these
    (e.g. explicit ``linewidth=``) still win at the call site.
    """
    mpl.rcParams.update({
        "font.size": 16,
        "font.weight": "bold",
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 16,
        "axes.labelweight": "bold",
        "axes.linewidth": 2.0,
        "axes.edgecolor": "black",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "lines.linewidth": 2.5,
        "lines.markersize": 9,
        "lines.markeredgewidth": 1.6,
        "lines.markeredgecolor": "black",
        "patch.linewidth": 1.5,
        "patch.edgecolor": "black",
        "legend.fontsize": 13,
        "legend.title_fontsize": 14,
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.dpi": 120,
    })
