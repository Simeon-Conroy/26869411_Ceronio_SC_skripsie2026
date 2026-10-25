"""Compact six-figure comparison of all filter-sweep H5 results."""
from pathlib import Path
import re
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "filter_results"
OUT = DATA / "comparison_plots"
FILTERS = ["Complementary", "Madgwick", "Mahony"]
METRICS = [("RMSE", "RMSE_mean"), ("MAE", "MAE_mean")]
METRIC_KEY = dict(METRICS)
DATASET_NAMES = {"Standstill":"Standstill", "Short_no_WP":"Short movement — no wrist/palm", "Short_WP":"Short movement — with wrist/palm", "Long_no_WP":"Long movement — no wrist/palm", "Long_WP":"Long movement — with wrist/palm"}

def read(path):
    with h5py.File(path, "r") as f:
        return {k: f[k][:] for k in f.keys()}

def discover():
    out = []
    for path in DATA.glob("*.h5"):
        m = re.match(r"^(Complementary|Madgwick|Mahony)_(.+?)_(no_noise|noise)\.h5$", path.name)
        if m:
            out.append({"filter":m.group(1), "run":m.group(2), "noise":m.group(3)=="noise", "data":read(path)})
    return out

def gain_info(item):
    d, f = item["data"], item["filter"]
    if f == "Complementary": values, name = np.asarray(d["Alpha"]), "Alpha"
    elif f == "Madgwick": values, name = np.asarray(d["Beta"]), "Beta"
    else: values, name = np.asarray(d["Ki"]), "Ki (Kp=0.1)"
    order = np.argsort(values)
    values = values[order]
    return np.arange(1, len(values) + 1), name, values, order

def gain_key(items):
    lines = []
    for filt in FILTERS:
        item = next((x for x in items if x["filter"] == filt), None)
        if item is None: continue
        _, name, values, _ = gain_info(item)
        lines.append(f"{filt}: " + ", ".join(f"{i}={v:g}" for i, v in enumerate(values, 1)) + (" (Kp=0.1)" if filt == "Mahony" else ""))
    return "\n".join(lines)

def label(run): return DATASET_NAMES.get(run, run.replace("_", " "))
def deg(x): return np.asarray(x, float) * 180 / np.pi

def overall(items, noisy, movement):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=False)
    for ax, (metric, key) in zip(axes.flat, METRICS):
        for item in sorted((x for x in items if x["noise"] == noisy and movement.lower() in x["run"].lower()), key=lambda x:(FILTERS.index(x["filter"]), x["run"])):
            x, xlabel, _, order = gain_info(item); y = deg(item["data"][key])[order]
            ax.plot(x, y, marker="o", linewidth=1.5, label=f"{label(item['run'])} — {item['filter']}")
        ax.set_title(metric); ax.set_xlabel("Gain setting index"); ax.set_ylabel("Degrees"); ax.grid(alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.105), ncol=3, fontsize=8)
    group = [x for x in items if x["noise"] == noisy and movement.lower() in x["run"].lower()]
    fig.suptitle(f"Overall filter metrics — {movement} — {'Gaussian noise' if noisy else 'no noise'}",y=0.92)
    fig.text(0.5, 0.045, "Gain key\n" + gain_key(group), ha="center", va="center", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="0.55", linewidth=0.8))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.23, wspace=0.18)
    fig.savefig(OUT / f"overall_{movement.lower()}_{'noise' if noisy else 'no_noise'}.svg"); plt.close(fig)

def per_joint(items, noisy, metrics, movement):
    OUT.mkdir(parents=True, exist_ok=True)
    available = [x for x in items if x["noise"] == noisy and movement.lower() in x["run"].lower()]
    if not available: return
    names = available[0]["data"].get("joint_names")
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    n_joints = len(names)
    cols = 5
    rows = int(np.ceil(n_joints / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3.1 * rows + 1.4), squeeze=False, constrained_layout=False)
    for j, name in enumerate(names):
        ax = axes.flat[j]
        for item in available:
            x, _, _, order = gain_info(item); d=item["data"]
            tag = f"{label(item['run'])} — {item['filter']}"
            for metric_index, (_, key) in enumerate(metrics):
                style = "-" if metric_index == 0 else "-"
                ax.plot(x, deg(d[key + "_pj_mean"][order, j]), marker=".", linewidth=1, linestyle=style, label=tag)
            ax.set_title(name, fontsize=9); ax.grid(alpha=.2); ax.set_xlabel("Gain index", fontsize=8); ax.set_ylabel("deg", fontsize=8)
    for ax in axes.flat[n_joints:]: ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.175), ncol=4, fontsize=8)
    suffix = metrics[0][0].lower() if len(metrics) == 1 else "combined"
    fig.suptitle(f"Per-joint {suffix.upper()} - {movement} — {'Gaussian noise' if noisy else 'no noise'}"  ,y=0.92)
    fig.text(0.5, 0.155, "Gain key\n" + gain_key(available), ha="center", va="center", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="0.55", linewidth=0.8))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.23, hspace=0.45, wspace=0.18)
    fig.savefig(OUT / f"per_joint_{suffix}_{movement.lower()}_{'noise' if noisy else 'no_noise'}.svg"); plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True); items=discover()
    if not items: raise RuntimeError(f"No *_noise.h5 or *_no_noise.h5 files found in {DATA}")
    for movement in ("Long", "Short", "Standstill"):
        for noisy in (False, True):
            overall(items, noisy, movement)
            per_joint(items, noisy, [("RMSE", "RMSE")], movement)
            per_joint(items, noisy, [("MAE", "MAE")], movement)
    print(f"Wrote movement-separated SVG figures to {OUT}")
if __name__ == "__main__": main()
