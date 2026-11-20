"""Joint power, low-friction robustness, and how they (and push robustness)
depend on the gait, for the wide-range study.

Reads, per run (task__run.npz):
- logs/study_results/power_wide/: grid at normal friction, 16 robots, 6 s
  settle + 6 s measured (study_eval.py --results power_wide ...);
- logs/study_results/friction0.3_wide/: the same at contact friction 0.3,
  below the 0.5-1.0 training range;
- logs/study_results/runs/: the full grid evaluation (64 robots), whose gait
  shares label each cell for the push and friction comparisons;
- logs/study_results/push_wide/: the push test (push_eval.py).

By setting: per-run means over the grid (power, cost of transport, power
fluctuation, power unevenness, RL power share, falls and speed at friction
0.3), mean +- SD over seeds, exact permutation p against the baseline.

By gait: gaits are grouped into families (2-leg, 2-leg + part-time 3rd, hop,
bound, pair + single, 3-beat, periodic, irregular). Power is compared per
robot (each robot's own gait label). Push and friction results have no
per-robot gait, so they are compared per (run, cell): the cell's gait is the
run's majority family there in the full grid evaluation (at least half of the
robots; else 'mixed'). Gaits differ by speed and frequency, and by reward
setting, so two adjusted comparisons are given besides the raw mean:
- cell-adjusted: value minus the mean of its (speed, frequency) cell over
  every run, i.e. compared with other gaits at the same speed and frequency;
- within-setting: value minus the mean of its cell over the runs of its own
  setting, i.e. compared with other seeds of the same reward at the same cell
  (gait differences there come from the seed, not the reward).
95% intervals are bootstrapped over runs.

Writes results/study/energy_wide/: energy_settings.csv, energy_by_gait.csv,
energy_settings.png, energy_by_gait.png.

Usage:
    uv run --frozen scripts/energy_summary.py
"""

import csv
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import study_summary
from grid_eval import PERIODIC
from study_summary import DATA, FIGURES, INK, MUTED, permutation_p, style

warnings.filterwarnings("ignore", category=RuntimeWarning)
study_summary.use_wide()
PREFIX, SETTINGS = study_summary.PREFIX, study_summary.STUDY_SETTINGS
OUT = FIGURES / "energy_wide"
RNG = np.random.default_rng(0)

FAMILIES = [
    "2-leg",
    "2-leg + part-time 3rd",
    "hop",
    "bound",
    "pair + single",
    "3-beat",
    "periodic",
    "irregular",
]
FAMILY_COLORS = {
    "2-leg": "#a3172f",
    "2-leg + part-time 3rd": "#f4a3b0",
    "hop": "#1f5fae",
    "bound": "#7d5bd0",
    "pair + single": "#eb6834",
    "3-beat": "#1baf7a",
    "periodic": "#c9a227",
    "irregular": "#a9a8a0",
    "mixed": "#6b6a64",
}


def family(name):
    if name in ("2-leg, together", "2-leg, alternating"):
        return "2-leg"
    if name in ("2-leg + sporadic 3rd", "2-leg hop + 3rd-leg support"):
        return "2-leg + part-time 3rd"
    if name in ("hop", "hop, staggered landing", "rear-first hop"):
        return "hop"
    if name == "bound":
        return "bound"
    if name.startswith("3-beat"):
        return "3-beat"
    if " | " in name:
        return "pair + single"
    if name == PERIODIC:
        return "periodic"
    return "irregular"  # irregular, not stepping


def load(folder):
    """{(setting, run): npz dict}."""
    out = {}
    for path in sorted((DATA / folder).glob(f"{PREFIX}*.npz")):
        m = re.fullmatch(rf"{PREFIX}(.+?)__(.+?)\.npz", path.name)
        if m and m[1] in SETTINGS and "__it" not in m[2]:
            out[(m[1], m[2])] = dict(np.load(path, allow_pickle=True))
    return out


def rl_index(d):
    return list(d["leg_names"]).index("RL")


# ---------------------------------------------------------------- by setting

SETTING_METRICS = [
    # key, label, fn(power npz, friction npz)
    ("power", "joint power [W]", lambda p, f: np.nanmean(p["power"])),
    ("cot", "cost of transport", lambda p, f: np.nanmean(p["cost_of_transport"])),
    (
        "cot_1_3",
        "cost of transport, 1-3 m/s",
        lambda p, f: np.nanmean(
            p["cost_of_transport"][:, (p["speeds"] >= 1) & (p["speeds"] <= 3)]
        ),
    ),
    (
        "power_fluctuation",
        "power fluctuation [SD/mean]",
        lambda p, f: np.nanmean(p["power_fluctuation"]),
    ),
    ("peak_power", "peak power [W]", lambda p, f: np.nanmean(p["peak_power"])),
    (
        "power_unevenness",
        "power unevenness across legs [CV]",
        lambda p, f: np.nanmean(p["power_unevenness"]),
    ),
    (
        "rl_power",
        "RL share of joint power",
        lambda p, f: np.nanmean(p["leg_power_shares"][..., rl_index(p)]),
    ),
    ("fall_nominal", "falls, normal friction", lambda p, f: np.mean(p["fall_rate"])),
    ("fall_f03", "falls, friction 0.3", lambda p, f: np.mean(f["fall_rate"])),
    (
        "speed_f03",
        "speed ratio, friction 0.3",
        lambda p, f: np.nanmean(f["speed_ratio"]),
    ),
]


def by_setting(power, friction):
    per = defaultdict(list)
    for key in sorted(power):
        if key in friction:
            per[key[0]].append(
                {k: float(fn(power[key], friction[key])) for k, _, fn in SETTING_METRICS}
            )
    rows, base = [], per.get("baseline", [])
    for setting in SETTINGS:
        group = per.get(setting)
        if not group:
            continue
        row = {"setting": setting, "seeds": len(group)}
        for k, _, _ in SETTING_METRICS:
            v = np.array([g[k] for g in group])
            row[f"{k}_mean"], row[f"{k}_sd"] = v.mean(), v.std(ddof=1)
            if base and setting != "baseline":
                ref = np.array([g[k] for g in base])
                row[f"{k}_p"], _ = permutation_p(
                    v, ref, lambda a, b: np.mean(a) - np.mean(b)
                )
        rows.append(row)
    return rows


def print_settings(rows):
    keys = ["power", "cot", "cot_1_3", "power_fluctuation", "power_unevenness",
            "rl_power", "fall_nominal", "fall_f03", "speed_f03"]
    pct = {"rl_power", "fall_nominal", "fall_f03"}
    heads = ["power W", "CoT", "CoT 1-3", "pwr fluct", "pwr uneven", "RL power",
             "falls", "falls f0.3", "speed f0.3"]
    print("\nBy setting: mean±SD over seeds (* p < 0.05 against baseline)")
    print(f"{'setting':10s} n " + "".join(f"{h:>15s}" for h in heads))
    for r in rows:
        cells = []
        for k in keys:
            m, s, p = r[f"{k}_mean"], r[f"{k}_sd"], r.get(f"{k}_p", 1.0)
            if k in pct:
                t = f"{m:.1%}±{s:.1%}"
            elif k == "power":
                t = f"{m:.0f}±{s:.0f}"
            else:
                t = f"{m:.2f}±{s:.2f}"
            cells.append(t + ("*" if p < 0.05 else " "))
        print(f"{r['setting']:10s} {r['seeds']} " + "".join(f"{c:>15s}" for c in cells))


def plot_settings(rows, path):
    keys = ["power", "cot_1_3", "power_fluctuation", "power_unevenness", "rl_power",
            "fall_f03"]
    labels = dict((k, l) for k, l, _ in SETTING_METRICS)
    names = [r["setting"] for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, k in zip(axes.ravel(), keys):
        style(ax)
        m = np.array([r[f"{k}_mean"] for r in rows])
        s = np.array([r[f"{k}_sd"] for r in rows])
        sig = [r.get(f"{k}_p", 1.0) < 0.05 for r in rows]
        colors = [INK if n == "baseline" else ("#2a78d6" if g else "#b9cfe9")
                  for n, g in zip(names, sig)]
        ax.bar(range(len(names)), m, yerr=s, color=colors, capsize=2,
               edgecolor="white", linewidth=1)
        ax.axhline(m[names.index("baseline")], color=INK, lw=0.8, ls="--")
        ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
        ax.set_title(labels[k], fontsize=10, color=INK)
        if k in ("rl_power", "fall_f03"):
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle(
        "Wide study: joint power and low-friction falls by reward setting "
        "(mean ± SD over seeds; dark blue: p < 0.05 against baseline, dashed line)",
        color=INK, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)


# ---------------------------------------------------------------- by gait


def adjusted(records, value):
    """Add raw, cell-adjusted and within-setting values to records (dicts
    with setting, run, cell and `value`)."""
    v = np.array([r[value] for r in records], float)
    cells = [r["cell"] for r in records]
    groups = [(r["setting"], r["cell"]) for r in records]
    for name, keys in (("cell", cells), ("within", groups)):
        sums, counts = defaultdict(float), defaultdict(int)
        for k, x in zip(keys, v):
            if np.isfinite(x):
                sums[k] += x
                counts[k] += 1
        for r, k, x in zip(records, keys, v):
            r[f"{value}:{name}"] = x - sums[k] / counts[k] if counts[k] else np.nan
    return records


def family_stats(records, value, families):
    """Per family: n, raw mean, cell-adjusted and within-setting means with
    95% bootstrap intervals over runs."""
    runs = sorted({(r["setting"], r["run"]) for r in records})
    index = {run: i for i, run in enumerate(runs)}
    out = {}
    for fam in families:
        sel = [r for r in records if r["family"] == fam and np.isfinite(r[value])]
        if len(sel) < 5:
            continue
        res = {"n": len(sel), "runs": len({(r["setting"], r["run"]) for r in sel})}
        for suffix in ("", ":cell", ":within"):
            key = value + suffix
            sums = np.zeros(len(runs))
            counts = np.zeros(len(runs))
            for r in sel:
                if np.isfinite(r[key]):
                    i = index[(r["setting"], r["run"])]
                    sums[i] += r[key]
                    counts[i] += 1
            mean = sums.sum() / counts.sum()
            boot = []
            for _ in range(1000):
                pick = RNG.integers(0, len(runs), len(runs))
                c = counts[pick].sum()
                if c:
                    boot.append(sums[pick].sum() / c)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            res[suffix or "raw"] = (mean, lo, hi)
        out[fam] = res
    return out


def robot_records(power):
    """Per robot that stayed up in the normal-friction power pass."""
    recs = []
    for (setting, run), d in power.items():
        names = list(d["gait_names"])
        rl = rl_index(d)
        labels = d["robot_label"]
        for (i, j, k), lab in np.ndenumerate(labels):
            if lab < 0:
                continue
            recs.append({
                "setting": setting,
                "run": run,
                "cell": (i, j),
                "family": family(names[lab]),
                "cot": d["robot_cost_of_transport"][i, j, k],
                "power_fluctuation": d["robot_power_fluctuation"][i, j, k],
                "power_unevenness": d["robot_power_unevenness"][i, j, k],
                "rl_power": d["robot_leg_power_shares"][i, j, k, rl],
            })
    return recs


def cell_family(d, i, j):
    counts = d["gait_counts"][i, j]
    if counts.sum() == 0:
        return None
    fams = defaultdict(float)
    for name, c in zip(d["gait_names"], counts):
        fams[family(str(name))] += c
    best = max(fams, key=fams.get)
    return best if fams[best] >= 0.5 * counts.sum() else "mixed"


def push_records(push, main):
    recs = []
    for key, d in push.items():
        if key not in main:
            continue
        g = main[key]
        for a, omega in enumerate(d["omegas"]):
            for b, speed in enumerate(d["speeds"]):
                i = int(np.flatnonzero(np.isclose(g["omegas"], omega))[0])
                j = int(np.flatnonzero(np.isclose(g["speeds"], speed))[0])
                fam = cell_family(g, i, j)
                if fam is None:
                    continue
                kept = ~d["fell_before"][a, b]
                mag = d["magnitude"][a, b]
                fell = d["fell"][a, b]
                big = kept & (mag >= 1.5)
                mid = kept & (mag >= 1.0) & (mag <= 1.5) & ~fell
                rec = d["recovery_time"][a, b][mid]
                recs.append({
                    "setting": key[0],
                    "run": key[1],
                    "cell": (i, j),
                    "family": fam,
                    "push_fall": fell[big].mean() if big.any() else np.nan,
                    "push_recovery": np.nanmedian(rec) if np.isfinite(rec).any()
                    else np.nan,
                })
    return recs


def friction_records(friction, power, main):
    recs = []
    for key, f in friction.items():
        if key not in power or key not in main:
            continue
        p, g = power[key], main[key]
        for (i, j), fall in np.ndenumerate(f["fall_rate"]):
            fam = cell_family(g, i, j)
            if fam is None:
                continue
            recs.append({
                "setting": key[0],
                "run": key[1],
                "cell": (i, j),
                "family": fam,
                "friction_extra_falls": fall - p["fall_rate"][i, j],
                "friction_speed_loss": p["speed_ratio"][i, j] - f["speed_ratio"][i, j],
            })
    return recs


GAIT_METRICS = [
    # key, label, records, percent
    ("cot", "cost of transport", "robot", False),
    ("power_fluctuation", "power fluctuation over time [SD/mean]", "robot", False),
    ("power_unevenness", "power unevenness across legs [CV]", "robot", False),
    ("rl_power", "RL share of joint power", "robot", True),
    ("push_fall", "push falls (1.5-2 m/s pushes)", "push", True),
    ("push_recovery", "push recovery time, 1-1.5 m/s [s]", "push", False),
    ("friction_extra_falls", "extra falls at friction 0.3", "friction", True),
    ("friction_speed_loss", "speed-ratio loss at friction 0.3", "friction", False),
]


def by_gait(power, friction, main, push):
    sources = {
        "robot": robot_records(power),
        "push": push_records(push, main),
        "friction": friction_records(friction, power, main),
    }
    results, rows = {}, []
    for key, label, src, pct in GAIT_METRICS:
        recs = adjusted(sources[src], key)
        stats = family_stats(recs, key, FAMILIES + ["mixed"])
        results[key] = stats
        total = sum(s["n"] for s in stats.values())
        for fam, s in stats.items():
            row = {"metric": key, "unit": src, "family": fam, "n": s["n"],
                   "share": s["n"] / total, "runs": s["runs"]}
            for name in ("raw", ":cell", ":within"):
                m, lo, hi = s[name]
                tag = name.strip(":")
                row[f"{tag}"], row[f"{tag}_lo"], row[f"{tag}_hi"] = m, lo, hi
            rows.append(row)
    return results, rows


def print_gait(results):
    print("\nBy gait family. raw mean | cell-adjusted | within-setting (difference "
          "from the cell mean; [95% interval over runs]); n = robots or run-cells")
    for key, label, src, pct in GAIT_METRICS:
        f = (lambda x: f"{x:+.1%}") if pct else (lambda x: f"{x:+.2f}")
        g = (lambda x: f"{x:.1%}") if pct else (lambda x: f"{x:.2f}")
        print(f"\n{label}  ({'per robot' if src == 'robot' else 'per run-cell'})")
        for fam, s in results[key].items():
            raw = s["raw"][0]
            c, w = s[":cell"], s[":within"]
            print(f"  {fam:22s} n={s['n']:6d}  {g(raw):>7s} | {f(c[0]):>7s} "
                  f"[{f(c[1])}, {f(c[2])}] | {f(w[0]):>7s} [{f(w[1])}, {f(w[2])}]")


def plot_gait(results, path):
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    for ax, (key, label, src, pct) in zip(axes.ravel(), GAIT_METRICS):
        style(ax)
        stats = results[key]
        fams = [f for f in FAMILIES + ["mixed"] if f in stats]
        x = np.arange(len(fams))
        for off, name, alpha, mk in ((-0.18, ":cell", 1.0, "o"),
                                     (0.18, ":within", 0.55, "s")):
            m = np.array([stats[f][name][0] for f in fams])
            lo = np.array([stats[f][name][1] for f in fams])
            hi = np.array([stats[f][name][2] for f in fams])
            for xi, f, mi, l, h in zip(x, fams, m, lo, hi):
                ax.plot([xi + off] * 2, [l, h], color=FAMILY_COLORS[f], lw=2,
                        alpha=alpha)
                ax.plot(xi + off, mi, mk, color=FAMILY_COLORS[f], ms=7, alpha=alpha,
                        markeredgecolor="white", markeredgewidth=1.5)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x, [f"{f}\n(n={stats[f]['n']})" for f in fams], fontsize=7,
                      rotation=35, ha="right")
        ax.set_title(label + "\nvs. others at the same cell", fontsize=9, color=INK)
        if pct:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    handles = [
        plt.Line2D([], [], marker="o", color=INK, lw=2, label="cell-adjusted (all runs)"),
        plt.Line2D([], [], marker="s", color=INK, lw=2, alpha=0.55,
                   label="within setting (same reward, other seeds)"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(
        "Wide study by gait family: difference from other gaits at the same speed "
        "and frequency (95% intervals over runs)", color=INK, fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=120)


def write_csv(rows, path):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    power, friction = load("power_wide"), load("friction0.3_wide")
    main_runs, push = load("runs"), load("push_wide")
    print(f"runs: power {len(power)}, friction {len(friction)}, grid {len(main_runs)}, "
          f"push {len(push)}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = by_setting(power, friction)
    print_settings(rows)
    write_csv(rows, OUT / "energy_settings.csv")
    plot_settings(rows, OUT / "energy_settings.png")
    results, gait_rows = by_gait(power, friction, main_runs, push)
    print_gait(results)
    write_csv(gait_rows, OUT / "energy_by_gait.csv")
    plot_gait(results, OUT / "energy_by_gait.png")
    print(f"\nSaved {OUT}/energy_settings.csv/.png, energy_by_gait.csv/.png")


if __name__ == "__main__":
    main()
