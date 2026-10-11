"""ORCA plotting primitives. Inputs must already have explicit, audited semantics.

No automatic exclusions, imputation, inference tests or synthetic observations.
Use Theme(project_style_path), then the functions below in a task-specific plot.py.
"""
from __future__ import annotations

import copy
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, Patch
import numpy as np


DEFAULT_STYLE = Path(__file__).resolve().parents[1] / "assets" / "orca-style.json"
if not DEFAULT_STYLE.exists():
    DEFAULT_STYLE = Path(__file__).with_name("orca-style.json")


def values(data, name="data", allow_nan=False):
    """Reject invalid observations instead of silently losing them."""
    a = np.asarray(data, dtype=float)
    if a.ndim != 1 or len(a) == 0:
        raise ValueError(f"{name}: expected a nonempty one-dimensional vector")
    bad = np.isinf(a) if allow_nan else ~np.isfinite(a)
    if bad.any() or not np.isfinite(a).any():
        raise ValueError(f"{name}: invalid/missing values require explicit audited handling")
    return a.copy()


def flag_iqr(data):
    """Return flags only. An IQR flag does not authorize removal."""
    a = values(data)
    q1, q3 = np.quantile(a, [0.25, 0.75])
    return (a < q1 - 1.5 * (q3 - q1)) | (a > q3 + 1.5 * (q3 - q1))


class Theme:
    def __init__(self, project_style=None, *, defaults=None, font_paths=()):
        self.path = Path(project_style) if project_style else None
        source = self.path if self.path and self.path.exists() else Path(defaults or DEFAULT_STYLE)
        self.config = json.loads(source.read_text(encoding="utf-8"))
        self.notes = []
        for path in font_paths:
            fm.fontManager.addfont(str(path))
        available = {f.name for f in fm.fontManager.ttflist}
        candidates = self.config["fonts"]["latin"]
        self.font = next((f for f in candidates if f in available), None)
        if self.font is None:
            raise RuntimeError("No configured Latin font available; register a local font explicitly")
        self.cjk = next((f for f in self.config["fonts"]["cjk"] if f in available), None)
        if self.font != candidates[0]:
            self.notes.append(f"Font fallback: {candidates[0]} -> {self.font}")
        old = self.config.get("actual_font")
        if old and old != self.font:
            self.notes.append(f"Project font changed in this environment: {old} -> {self.font}")
        self.config["actual_font"] = self.font
        self.config["actual_cjk_font"] = self.cjk
        self.save()

    @property
    def c(self):
        return self.config["colors"]

    @property
    def g(self):
        return self.config["geometry"]

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def style(self, group):
        if not isinstance(group, str) or not group:
            raise ValueError("Group IDs must be nonempty strings")
        styles = self.config["group_styles"]
        if group not in styles:
            i = len(styles)
            if i >= len(self.config["categorical"]):
                raise ValueError("Palette exhausted: define an explicit distinguishable group style or facet")
            styles[group] = {"color": self.config["categorical"][i], "marker": self.config["markers"][i]}
        if group not in self.config["group_order"]:
            self.config["group_order"].append(group)
        self.save()
        return styles[group]

    def order(self, groups):
        groups = list(groups)
        for group in groups:
            self.style(group)
        return [g for g in self.config["group_order"] if g in groups]

    def rc(self):
        t, g, c = self.config["text"], self.g, self.c
        family = [self.font] + ([self.cjk] if self.cjk else [])
        return {
            "font.family": family, "font.weight": self.config["fonts"]["weight"],
            "font.size": t["tick_pt"], "axes.labelsize": t["label_pt"],
            "axes.labelweight": self.config["fonts"]["weight"],
            "xtick.labelsize": t["tick_pt"], "ytick.labelsize": t["tick_pt"],
            "legend.fontsize": t["legend_pt"], "legend.frameon": False,
            "text.color": c["ink"], "axes.labelcolor": c["ink"],
            "axes.edgecolor": c["ink"], "xtick.color": c["ink"], "ytick.color": c["ink"],
            "axes.linewidth": g["axis_width"], "lines.linewidth": g["line_width"],
            "axes.spines.top": False, "axes.spines.right": False,
            "xtick.direction": "out", "ytick.direction": "out",
            "xtick.major.width": g["axis_width"], "ytick.major.width": g["axis_width"],
            "axes.grid": False, "figure.facecolor": self.config["figure"]["background"],
            "axes.facecolor": self.config["figure"]["background"],
            "svg.fonttype": "path", "pdf.fonttype": 42, "ps.fonttype": 42,
        }

    def context(self):
        return matplotlib.rc_context(self.rc())

    def figure(self, *, width=None, height=None):
        f = self.config["figure"]
        return plt.subplots(figsize=(width or f["width_in"], height or f["height_in"]), layout="constrained")

    def legend(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)


def _points(ax, x, y, theme, group):
    s = theme.style(group)
    return ax.scatter(x, y, s=theme.g["point_area"], c=s["color"], marker=s["marker"],
                      alpha=theme.g["point_alpha"], edgecolors=theme.c["ink"],
                      linewidths=theme.g["point_edge"], zorder=3)


def grouped(ax, groups, theme, *, kind="box", xlabel="", ylabel="Value"):
    """Raw independent-unit values. bar/dot use mean + sample SD, never SEM."""
    if kind not in {"bar", "dot", "box", "violin"}:
        raise ValueError("Unknown grouped chart")
    order = theme.order(groups)
    if not order:
        raise ValueError("No groups")
    arrays = {g: values(groups[g], g) for g in order}
    if kind == "violin" and any(len(a) < 10 or np.ptp(a) == 0 for a in arrays.values()):
        raise ValueError("Violin requires >=10 nonconstant values per group; use box/dot")
    summary = []
    for i, name in enumerate(order):
        a = arrays[name]
        color = theme.style(name)["color"]
        mean = float(np.mean(a))
        sd = float(np.std(a, ddof=1)) if len(a) > 1 else None
        q1, med, q3 = np.quantile(a, [0.25, 0.5, 0.75])
        summary.append({"group": name, "n": len(a), "mean": mean, "sd": sd,
                        "q1": float(q1), "median": float(med), "q3": float(q3)})
        if kind in {"bar", "dot"}:
            if kind == "bar":
                ax.bar(i, mean, width=theme.g["bar_width"], facecolor=color, alpha=0.28,
                       edgecolor=theme.c["ink"], linewidth=theme.g["axis_width"], zorder=1)
            if sd is not None:
                ax.errorbar(i, mean, yerr=sd, fmt="none", color=theme.c["ink"],
                            elinewidth=1.5, capsize=theme.g["error_cap"], zorder=4)
            if kind == "dot":
                ax.scatter([i], [mean], s=90, c=theme.c["ink"], marker="D", zorder=5)
        elif kind == "box":
            box = ax.boxplot([a], positions=[i], widths=theme.g["box_width"], patch_artist=True,
                             manage_ticks=False, showfliers=False, whis=1.5,
                             medianprops={"color": theme.c["ink"], "linewidth": 2},
                             whiskerprops={"color": theme.c["ink"], "linewidth": 1.5},
                             capprops={"color": theme.c["ink"], "linewidth": 1.5},
                             boxprops={"edgecolor": theme.c["ink"], "linewidth": 1.5})
            box["boxes"][0].set_facecolor(color)
            box["boxes"][0].set_alpha(0.35)
        else:
            violin = ax.violinplot([a], positions=[i], widths=theme.g["violin_width"],
                                   showextrema=False, bw_method="scott", points=100)
            violin["bodies"][0].set_facecolor(color)
            violin["bodies"][0].set_edgecolor(theme.c["ink"])
            violin["bodies"][0].set_alpha(0.35)
            ax.vlines(i, q1, q3, color=theme.c["ink"], linewidth=4, zorder=4)
            ax.scatter([i], [med], c="white", s=35, edgecolors=theme.c["ink"], zorder=5)
        # Stable group-specific jitter, independent of other groups and measured values.
        digest = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
        rng = np.random.default_rng(theme.config["seed"] + digest)
        jitter = rng.uniform(-theme.g["jitter_width"], theme.g["jitter_width"], len(a))
        _points(ax, i + jitter, a, theme, name)
    ax.set_xticks(range(len(order)), order)
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(-0.6, len(order) - 0.4))
    if kind == "bar":
        low, high = ax.get_ylim()
        ax.set_ylim(min(0, low), max(0, high))
    return summary


def scatter(ax, x, y, theme, *, group="Data", xlabel="x", ylabel="y"):
    x, y = values(x, "x"), values(y, "y")
    if len(x) != len(y):
        raise ValueError("x and y lengths differ")
    artist = _points(ax, x, y, theme, group)
    artist.set_label(group)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    return {"n": len(x)}


def line(ax, x, y, theme, *, group="Data", xlabel="x", ylabel="y"):
    x, y = values(x, "x"), values(y, "y", allow_nan=True)
    if len(x) != len(y) or np.any(np.diff(x) <= 0):
        raise ValueError("Line x must be strictly increasing, one y per x; handle replicates explicitly")
    s = theme.style(group)
    ax.plot(x, y, color=s["color"], marker=s["marker"], markersize=6,
            linewidth=theme.g["line_width"], label=group)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    return {"n": int(np.isfinite(y).sum()), "gaps": int(np.isnan(y).sum())}


def heatmap(ax, matrix, theme, *, rows, columns, label, center=None, vmin=None, vmax=None):
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape != (len(rows), len(columns)) or not a.size:
        raise ValueError("Matrix shape and labels must match")
    if np.isinf(a).any() or not np.isfinite(a).any():
        raise ValueError("Heatmap cannot contain infinity or only missing values")
    lo = float(np.nanmin(a)) if vmin is None else float(vmin)
    hi = float(np.nanmax(a)) if vmax is None else float(vmax)
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("Supply meaningful distinct color limits for constant/invalid range")
    if center is None:
        colors = [theme.c["lavender100"], theme.c["indigo500"], theme.c["indigo900"]]
        norm = Normalize(lo, hi)
    else:
        if not lo < center < hi:
            raise ValueError("Diverging center must lie inside explicit color limits")
        colors = [theme.c["indigo700"], "#FFFEFC", theme.c["coral400"]]
        norm = TwoSlopeNorm(vmin=lo, vcenter=center, vmax=hi)
    cmap = LinearSegmentedColormap.from_list("orca", colors)
    cmap.set_bad(theme.c["missing"])
    im = ax.imshow(np.ma.masked_invalid(a), cmap=cmap, norm=norm, aspect="auto", interpolation="none")
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(rows)), rows)
    low_clip, high_clip = np.nanmin(a) < lo, np.nanmax(a) > hi
    extend = "both" if low_clip and high_clip else "min" if low_clip else "max" if high_clip else "neither"
    ax.figure.colorbar(im, ax=ax, label=label, extend=extend, fraction=0.05, pad=0.04)
    if np.isnan(a).any():
        ax.legend(handles=[Patch(facecolor=theme.c["missing"], label="Missing")],
                  loc="upper left", bbox_to_anchor=(0, -0.12), borderaxespad=0)
    return {"shape": list(a.shape), "missing": int(np.isnan(a).sum()),
            "vmin": lo, "vmax": hi, "center": center, "colorbar_extend": extend}


def volcano(ax, log2fc, p, theme, *, p_label, p_threshold, fc_threshold, display_cap=None):
    fc, p = values(log2fc, "log2FC"), values(p, p_label)
    if len(fc) != len(p) or np.any((p < 0) | (p > 1)):
        raise ValueError("Require matched log2FC and p values in [0,1]")
    if not 0 < p_threshold < 1 or not np.isfinite(fc_threshold) or fc_threshold <= 0:
        raise ValueError("Explicit positive effect and probability thresholds are required")
    if (p == 0).any() and display_cap is None:
        raise ValueError("p=0 requires an explicit documented display_cap; input remains unchanged")
    y = np.full(len(p), np.inf)
    np.log10(p, out=y, where=p > 0)
    y[p > 0] *= -1
    capped = np.zeros(len(p), dtype=bool)
    if display_cap is not None:
        if not np.isfinite(display_cap) or display_cap <= -np.log10(p_threshold):
            raise ValueError("display_cap must exceed the significance threshold")
        capped = y > display_cap
        y = np.minimum(y, display_cap)
    down = (fc <= -fc_threshold) & (p <= p_threshold)
    up = (fc >= fc_threshold) & (p <= p_threshold)
    masks = {"Below thresholds": ~(down | up), "Negative": down, "Positive": up}
    colors = [theme.c["missing"], theme.c["indigo700"], theme.c["coral400"]]
    for (name, mask), color in zip(masks.items(), colors):
        ax.scatter(fc[mask & ~capped], y[mask & ~capped], c=color, s=theme.g["point_area"],
                   alpha=0.8, edgecolors=theme.c["ink"], linewidths=0.4, label=name)
        if (mask & capped).any():
            ax.scatter(fc[mask & capped], y[mask & capped], c=color, marker="^", s=60,
                       edgecolors=theme.c["ink"], linewidths=0.6)
    if capped.any():
        ax.scatter([], [], marker="^", c=theme.c["ink"], label="Display capped")
    for x in [-fc_threshold, fc_threshold]:
        ax.axvline(x, color=theme.c["muted"], linewidth=1.25, linestyle="--", zorder=0)
    ax.axhline(-np.log10(p_threshold), color=theme.c["muted"], linewidth=1.25, linestyle="--", zorder=0)
    limit = max(float(np.max(np.abs(fc))), fc_threshold) * 1.08
    ax.set(xlabel="log₂ fold change", ylabel=f"−log₁₀({p_label})", xlim=(-limit, limit))
    ax.set_ylim(bottom=0)
    theme.legend(ax)
    return {"n": len(p), "p_label": p_label, "p_threshold": p_threshold,
            "fc_threshold": fc_threshold, "counts": {k: int(m.sum()) for k, m in masks.items()},
            "capped": int(capped.sum()), "display_cap": display_cap}


def venn(ax, sets, theme):
    """Exact mutually exclusive counts; schematic circles are NOT area proportional."""
    if len(sets) not in (2, 3) or any(not isinstance(s, (set, frozenset)) for s in sets.values()):
        raise ValueError("Supply 2–3 sets of audited, unique IDs; use UpSet for more")
    order = theme.order(sets)
    n = len(order)
    union = set().union(*(sets[k] for k in order))
    counts = {format(i, f"0{n}b"): 0 for i in range(1, 2 ** n)}
    for item in union:
        bits = "".join("1" if item in sets[k] else "0" for k in order)
        counts[bits] += 1
    if n == 2:
        centers = [(-0.45, 0), (0.45, 0)]
        labels = [(-0.65, 1.15), (0.65, 1.15)]
        positions = {"10": (-0.85, 0), "11": (0, 0), "01": (0.85, 0)}
    else:
        centers = [(-0.45, 0.30), (0.45, 0.30), (0, -0.48)]
        labels = [(-0.80, 1.48), (0.80, 1.48), (0, -1.65)]
        positions = {"100": (-0.87, 0.57), "010": (0.87, 0.57), "001": (0, -1.0),
                     "110": (0, 0.73), "101": (-0.48, -0.35), "011": (0.48, -0.35), "111": (0, 0.08)}
    for group, xy, label_xy in zip(order, centers, labels):
        ax.add_patch(Circle(xy, 1, facecolor=theme.style(group)["color"], alpha=0.28,
                            edgecolor=theme.c["ink"], linewidth=1.5))
        ax.text(*label_xy, group, ha="center", va="center", fontsize=theme.config["text"]["label_pt"])
    for bits, xy in positions.items():
        ax.text(*xy, str(counts[bits]), ha="center", va="center", fontsize=theme.config["text"]["emphasis_pt"])
    ax.set(xlim=(-1.7, 1.7), ylim=(-1.9, 1.8), aspect="equal")
    ax.axis("off")
    return {"set_order": order, "region_counts": counts, "union_n": len(union),
            "area_proportional": False, "caption_note": "Circles are schematic; area does not encode counts."}


def histogram(ax, groups, theme, *, bins, xlabel="Value", density=False):
    edges = values(bins, "bin edges")
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("Provide explicit strictly increasing shared bin edges")
    arrays = {name: values(a, name) for name, a in groups.items()}
    if not arrays:
        raise ValueError("No groups")
    if any(np.min(a) < edges[0] or np.max(a) > edges[-1] for a in arrays.values()):
        raise ValueError("Bins omit observations; include full range or audit an explicit subset")
    counts = {}
    for name in theme.order(arrays):
        a = arrays[name]
        h, _ = np.histogram(a, bins=edges, density=density)
        ax.stairs(h, edges, color=theme.style(name)["color"], linewidth=2, label=name)
        counts[name] = h.tolist()
    ax.set(xlabel=xlabel, ylabel="Density" if density else "Count", ylim=(0, None))
    theme.legend(ax)
    return {"bins": edges.tolist(), "density": density, "heights": counts}


def ecdf(ax, groups, theme, *, xlabel="Value"):
    if not groups:
        raise ValueError("No groups")
    output = {}
    for name in theme.order(groups):
        a = values(groups[name], name)
        x, count = np.unique(a, return_counts=True)
        y = np.cumsum(count) / len(a)
        ax.step(np.r_[x[0], x], np.r_[0, y], where="post", color=theme.style(name)["color"], label=name)
        output[name] = {"n": len(a), "x": x.tolist(), "cumulative": y.tolist()}
    ax.set(xlabel=xlabel, ylabel="Cumulative proportion", ylim=(0, 1.03))
    theme.legend(ax)
    return output


def export(fig, directory, theme, *, caption, alt_text, provenance, stem="figure"):
    """Export fixed-canvas PNG/SVG/PDF plus metadata; never overwrites outputs."""
    if not caption.strip() or not alt_text.strip() or not provenance:
        raise ValueError("A caption, alt text and provenance are required")
    if Path(stem).name != stem:
        raise ValueError("stem must be a filename, not a path")
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"png": out / f"{stem}@2x.png", "svg": out / f"{stem}.svg", "pdf": out / f"{stem}.pdf",
             "caption": out / "caption.md", "alt": out / "alt-text.txt", "manifest": out / "figure-manifest.json",
             "style": out / "orca-figure-style.json"}
    if any(p.exists() for p in paths.values()):
        raise FileExistsError("Use a new/versioned output directory")
    f = theme.config["figure"]
    dpi = f["display_width_px"] * f["export_scale"] / fig.get_figwidth()
    with theme.context():
        fig.canvas.draw()
        for fmt in ("png", "svg", "pdf"):
            fig.savefig(paths[fmt], format=fmt, dpi=dpi, facecolor=f["background"], bbox_inches=None)
    paths["caption"].write_text(caption.rstrip() + "\n", encoding="utf-8")
    paths["alt"].write_text(alt_text.rstrip() + "\n", encoding="utf-8")
    paths["style"].write_text(json.dumps(theme.config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"template_version": theme.config["template_version"], "font": theme.font,
                "cjk_font": theme.cjk, "notes": theme.notes,
                "runtime": {"python": platform.python_version(), "matplotlib": matplotlib.__version__, "numpy": np.__version__},
                "export": {"width_in": fig.get_figwidth(), "height_in": fig.get_figheight(), "dpi": dpi,
                           "display_width_px": f["display_width_px"], "svg_text": "paths", "pdf_fonttype": 42},
                "provenance": copy.deepcopy(provenance)}
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {k: str(p) for k, p in paths.items()}
